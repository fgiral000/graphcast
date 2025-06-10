#!/usr/bin/env python3
# ---------------------------------------------------------------------------
#  Train GenCast with Flax-NNX + Grain + Orbax
# ---------------------------------------------------------------------------
#  Quick start (single host, CPU/GPU/TPU) **
#
#  pip install "jax>=0.4.26" "flax>=0.8" optax \
#              gcsfs xarray zarr orbax-checkpoint grain
#
#  python train_gencast.py \
#       --gcs_zarr gs://graphcast-demo/mini_batch_2018.zarr \
#       --norm_stats gs://graphcast-demo/normalization/era5_1979_2017.nc \
#       --batch_size 8 --num_epochs 2 --lr 1e-4 --save_dir ~/checkpoints
# ---------------------------------------------------------------------------
from __future__ import annotations
import argparse, functools, os, pathlib, time, warnings
from typing import Any, Dict, Iterator

import gcsfs, xarray as xr
import jax, jax.numpy as jnp
import optax
import flax.nnx as nnx
import orbax.checkpoint as ocp
import numpy as np
import grain
from grain import transforms as G

# ---------------------------------------------------------------------------#
#  GenCast + wrappers (your repo checkout must be on PYTHONPATH)             #
# ---------------------------------------------------------------------------#
from gencast import gencast          # your NNX port
from graphcast import graphcast
from gencast import nan_cleaning
from common import normalization

# ---------------------------------------------------------------------------#
#  --------------------------  DATA  --------------------------------------- #
# ---------------------------------------------------------------------------#
def open_gcs_zarr(path: str) -> xr.Dataset:
    """Open a Zarr store on Google Cloud Storage with Xarray + GCSFS."""
    fs = gcsfs.GCSFileSystem(token="cloud")          # anonymous, public bucket
    mapper = gcsfs.mapping.GCSMap(path, gcs=fs)      # fsspec mapper
    return xr.open_zarr(mapper, consolidated=True)

class WeatherRecord(G.MapTransform):
    """Grain transform: integer index → dict(inputs, targets, forcings)."""

    def __init__(self, ds: xr.Dataset, task: graphcast.TaskConfig):
        super().__init__()
        self._ds   = ds
        self._task = task

    def map(self, idx: int) -> Dict[str, xr.Dataset]:
        sample = self._ds.isel(sample=idx)
        inputs   = sample[self._task.input_variables ].to_dataset("variable")
        targets  = sample[self._task.target_variables].to_dataset("variable")
        forcings = None
        if self._task.forcing_variables:
            forcings = sample[list(self._task.forcing_variables)].to_dataset("variable")
        return dict(inputs=inputs, targets=targets, forcings=forcings)

def build_dataloader(
        ds: xr.Dataset,
        task: graphcast.TaskConfig,
        batch_size: int,
        num_epochs: int,
        shuffle: bool = True,
        seed: int = 0,
        worker_count: int = 0,
    ) -> grain.DataLoader:
    """Return a Grain DataLoader that yields batched dictionaries."""
    source   = grain.RangeDataSource(len(ds.sample))
    sampler  = grain.IndexSampler(
        num_records=len(source),
        num_epochs=num_epochs,
        shard_options=grain.NoSharding(),   # single host – change for multi-host
        shuffle=shuffle,
        seed=seed,
    )
    ops = [
        WeatherRecord(ds, task),
        G.Batch(batch_size, drop_remainder=True),
    ]
    return grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=ops,
        worker_count=worker_count)

# ---------------------------------------------------------------------------#
#  -----------------------  MODEL + WRAPPERS  ------------------------------ #
# ---------------------------------------------------------------------------#
def build_core_model(rngs: nnx.Rngs) -> gencast.GenCast:
    """Bare GenCast instantiation – edit hyper-parameters as needed."""
    den_cfg   = gencast.denoiser.DenoiserArchitectureConfig()
    return gencast.GenCast(
        task_config               = gencast.TASK,
        denoiser_architecture_config = den_cfg,
        sampler_config            = gencast.SamplerConfig(),
        noise_config              = gencast.NoiseConfig(),
        noise_encoder_config      = gencast.denoiser.NoiseEncoderConfig(),
        rngs                      = rngs,
    )

def build_full_model(rngs: nnx.Rngs, norm_stats_path: str) -> nan_cleaning.NanCleaner:
    """GenCast wrapped with Normalizer → NanCleaner."""
    core = build_core_model(rngs)
    norm_stats = xr.open_dataset(norm_stats_path)   # means / stds / diffs …
    normalizer = normalization.Normalizer(norm_stats, task_config=gencast.TASK)
    model_norm = normalizer.wrap(core)
    return nan_cleaning.NanCleaner(model_norm)

# ---------------------------------------------------------------------------#
#  -----------------------  CHECKPOINTING  --------------------------------- #
# ---------------------------------------------------------------------------#
def make_ckpt_manager(ckpt_dir: pathlib.Path) -> ocp.CheckpointManager:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    handler   = ocp.StandardCheckpointHandler()
    options   = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    return ocp.CheckpointManager(str(ckpt_dir), handler, options)

def save_step(
    step: int,
    model: nnx.Module,
    optim: nnx.Optimizer,
    loader_iter: grain.DataLoaderIterator,
    manager: ocp.CheckpointManager,
) -> None:
    """Save <model state, optimiser state, data-iterator> in one Orbax ckpt."""
    _, state = nnx.split(model)
    state_dict   = nnx.to_pure_dict(state)
    optim_dict   = nnx.to_pure_dict(nnx.split(optim)[1])
    target_tree  = dict(model=state_dict, optim=optim_dict)
    save_args    = ocp.args.Composite(
        model = ocp.args.PyTreeSave(state_dict),
        optim = ocp.args.PyTreeSave(optim_dict),
        data_iter = grain.PyGrainCheckpointSave(loader_iter)
    )
    manager.save(step, target_tree, save_args=save_args)   # async by default
    manager.wait_until_finished()

def restore_if_possible(
    manager: ocp.CheckpointManager,
    rngs: nnx.Rngs,
    norm_stats_path: str,
    optim_ctor: optax.GradientTransformation,
    loader_iter: grain.DataLoaderIterator,
) -> tuple[int, nan_cleaning.NanCleaner, nnx.Optimizer]:
    """Return (start_step, model, optimiser); create fresh ones if no ckpt."""
    latest = manager.latest_step()
    if latest is None:
        # Nothing to restore – create fresh objects.
        model = build_full_model(rngs, norm_stats_path)
        optim = nnx.Optimizer(model, optim_ctor)
        return 0, model, optim

    # ----- Prepare abstract structures for Orbax --------------------------
    abstract_model   = nnx.eval_shape(lambda: build_full_model(rngs, norm_stats_path))
    graphdef, abs_state = nnx.split(abstract_model)
    abs_model_dict   = nnx.to_pure_dict(abs_state)
    abs_optim_dict   = nnx.to_pure_dict(nnx.split(nnx.Optimizer(abstract_model, optim_ctor))[1])

    restore_args = ocp.args.Composite(
        model = abs_model_dict,
        optim = abs_optim_dict,
        data_iter = grain.PyGrainCheckpointRestore(loader_iter)
    )
    restored = manager.restore(latest, restore_args=restore_args)

    # ----- Re-hydrate the model & optimiser -------------------------------
    nnx.replace_by_pure_dict(abs_state, restored["model"])
    model = nnx.merge(graphdef, abs_state)
    optim = nnx.Optimizer(model, optim_ctor)        # bind to *live* model
    nnx.replace_by_pure_dict(nnx.split(optim)[1], restored["optim"])

    print(f"✔️  Restored checkpoint @ step {latest}")
    return latest + 1, model, optim

# ---------------------------------------------------------------------------#
#  -------------------------  TRAIN & EVAL FNs  ---------------------------- #
# ---------------------------------------------------------------------------#
def make_train_fn(model: nan_cleaning.NanCleaner, optim: nnx.Optimizer):
    @nnx.jit
    def _step(batch: Dict[str, xr.Dataset]) -> jnp.ndarray:
        def loss_fn(m):
            loss, _ = m.loss(batch["inputs"], batch["targets"], batch["forcings"])
            return jnp.mean(loss)
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optim.update(grads)
        return loss
    return _step

def make_eval_fn(model: nan_cleaning.NanCleaner):
    @nnx.jit
    def _step(batch: Dict[str, xr.Dataset]) -> jnp.ndarray:
        preds = model(batch["inputs"], batch["targets"], batch["forcings"])
        mse   = (preds - batch["targets"]).to_array().data ** 2
        return jnp.mean(mse)
    return _step

# ---------------------------------------------------------------------------#
#  ------------------------------  MAIN  ----------------------------------- #
# ---------------------------------------------------------------------------#
def main() -> None:
    parser = argparse.ArgumentParser(description="Train GenCast (NNX + Grain)")
    parser.add_argument("--gcs_zarr",   required=True, help="gs://… path to mini ERA5 batch")
    parser.add_argument("--norm_stats", required=True, help="gs://… normalization .nc")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--save_dir",   type=str,   default="checkpoints")
    parser.add_argument("--workers",    type=int,   default=0)
    args = parser.parse_args()

    # ---- DATA ------------------------------------------------------------
    print("⇢ Opening dataset on GCS …")
    ds = open_gcs_zarr(args.gcs_zarr)
    loader = build_dataloader(
        ds, gencast.TASK, args.batch_size,
        num_epochs=args.num_epochs,
        worker_count=args.workers)
    loader_iter = iter(loader)

    # ---- RNG / model / optimiser ----------------------------------------
    rngs = nnx.Rngs(0)
    opt_tx = optax.adamw(args.lr)
    ckpt_mgr = make_ckpt_manager(pathlib.Path(args.save_dir))
    step, model, optim = restore_if_possible(
        ckpt_mgr, rngs, args.norm_stats, opt_tx, loader_iter)

    train_step = make_train_fn(model, optim)
    eval_step  = make_eval_fn(model)

    # ---- LOOP -----------------------------------------------------------
    for epoch in range(step, args.num_epochs):
        t0, losses = time.time(), []
        for batch in loader:
            loss = float(train_step(batch))
            losses.append(loss)
        print(f"epoch {epoch:03d} | train-loss {np.mean(losses):.4e} | {time.time()-t0:.1f}s")

        # quick validation: reuse first N batches
        val_losses, loader_iter_val = [], iter(loader)
        for _ in range(10):                           # cheap sanity check
            try:
                batch = next(loader_iter_val)
            except StopIteration:
                break
            val_losses.append(float(eval_step(batch)))
        print(f"            |   val-mse {np.mean(val_losses):.4e}")

        save_step(epoch, model, optim, loader_iter, ckpt_mgr)

    print("✔︎ Finished training.")

# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    # Grain likes to be quiet – drop some verbose Orbax warnings in interactive runs
    warnings.filterwarnings("ignore", category=UserWarning, module="orbax")
    main()

