# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wrappers that take care of casting."""

import contextlib
from typing import Any, Mapping, Tuple

import chex
from common import predictor_base
import jax
import jax.numpy as jnp
import numpy as np
import xarray


PyTree = Any


class Bfloat16Cast(predictor_base.Predictor):
  """Wrapper that casts all inputs to bfloat16 and outputs to targets dtype."""

  def __init__(self, predictor: predictor_base.Predictor, enabled: bool = True):
    """Inits the wrapper.

    Args:
      predictor: predictor being wrapped.
      enabled: disables the wrapper if False, for simpler hyperparameter scans.

    """
    self._enabled = enabled
    self._predictor = predictor

  def __call__(self,
               inputs: xarray.Dataset,
               targets_template: xarray.Dataset,
               forcings: xarray.Dataset,
               **kwargs
               ) -> xarray.Dataset:
    if not self._enabled:
      return self._predictor(inputs, targets_template, forcings, **kwargs)

    # Cast all inputs to bfloat16 before passing to the predictor.
    # The predictor is assumed to handle its internal precision.
    predictions = self._predictor(
        *_all_inputs_to_bfloat16(inputs, targets_template, forcings),
        **kwargs,)

    predictions_dtype = infer_floating_dtype(predictions)
    # The wrapped predictor should ideally output bfloat16 if inputs were bfloat16
    # and it's designed for mixed precision.
    if predictions_dtype != jnp.bfloat16:
      raise ValueError(f'Expected bfloat16 output, got {predictions_dtype}')

    targets_dtype = infer_floating_dtype(targets_template)
    return tree_map_cast(
        predictions, input_dtype=jnp.bfloat16, output_dtype=targets_dtype)

  def loss(self,
           inputs: xarray.Dataset,
           targets: xarray.Dataset,
           forcings: xarray.Dataset,
           **kwargs,
           ) -> predictor_base.LossAndDiagnostics:
    if not self._enabled:
      return self._predictor.loss(inputs, targets, forcings, **kwargs)

    # Cast all inputs to bfloat16 before passing to the predictor.
    loss, scalars = self._predictor.loss(
        *_all_inputs_to_bfloat16(inputs, targets, forcings), **kwargs)

    if loss.dtype != jnp.bfloat16:
      raise ValueError(f'Expected bfloat16 loss, got {loss.dtype}')

    targets_dtype = infer_floating_dtype(targets)

    # Note that casting back the loss to e.g. float32 should not affect data
    # types of the backwards pass, because the first thing the backwards pass
    # should do is to go backwards the casting op and cast back to bfloat16
    # (and xprofs seem to confirm this).
    return tree_map_cast((loss, scalars),
                         input_dtype=jnp.bfloat16, output_dtype=targets_dtype)

  def loss_and_predictions(  # pytype: disable=signature-mismatch  # jax-ndarray
      self,
      inputs: xarray.Dataset,
      targets: xarray.Dataset,
      forcings: xarray.Dataset,
      **kwargs,
      ) -> Tuple[predictor_base.LossAndDiagnostics,
                 xarray.Dataset]:
    if not self._enabled:
      return self._predictor.loss_and_predictions(inputs, targets, forcings,  # pytype: disable=bad-return-type  # jax-ndarray
                                                  **kwargs)

    # Cast all inputs to bfloat16 before passing to the predictor.
    (loss, scalars), predictions = self._predictor.loss_and_predictions(
        *_all_inputs_to_bfloat16(inputs, targets, forcings), **kwargs)

    if loss.dtype != jnp.bfloat16:
      raise ValueError(f'Expected bfloat16 loss, got {loss.dtype}')

    predictions_dtype = infer_floating_dtype(predictions)
    if predictions_dtype != jnp.bfloat16:
      raise ValueError(f'Expected bfloat16 output, got {predictions_dtype}')

    targets_dtype = infer_floating_dtype(targets)
    return tree_map_cast(((loss, scalars), predictions),
                         input_dtype=jnp.bfloat16, output_dtype=targets_dtype)


def infer_floating_dtype(data_vars: Mapping[str, chex.Array]) -> np.dtype:
  """Infers a floating dtype from an input mapping of data."""
  dtypes = {
      v.dtype
      for k, v in data_vars.items() if jnp.issubdtype(v.dtype, np.floating)}
  if len(dtypes) != 1:
    dtypes_and_shapes = {
        k: (v.dtype, v.shape)
        for k, v in data_vars.items() if jnp.issubdtype(v.dtype, np.floating)}
    raise ValueError(
        f'Did not found exactly one floating dtype {dtypes} in input variables:'
        f'{dtypes_and_shapes}')
  return list(dtypes)[0]


def _all_inputs_to_bfloat16(
    inputs: xarray.Dataset,
    targets: xarray.Dataset,
    forcings: xarray.Dataset,
    ) -> Tuple[xarray.Dataset,
               xarray.Dataset,
               xarray.Dataset]:
  return (inputs.astype(jnp.bfloat16),
          jax.tree.map(lambda x: x.astype(jnp.bfloat16), targets),
          forcings.astype(jnp.bfloat16))


def tree_map_cast(inputs: PyTree, input_dtype: np.dtype, output_dtype: np.dtype,
                  ) -> PyTree:
  def cast_fn(x):
    if isinstance(x, (jnp.ndarray, np.ndarray)) and x.dtype == input_dtype:
      return x.astype(output_dtype)
    return x # Return x unchanged if not a numerical array or not the specified input_dtype
  return jax.tree.map(cast_fn, inputs)
