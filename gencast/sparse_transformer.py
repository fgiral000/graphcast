# Copyright 2024 DeepMind Technologies Limited.
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
"""Transformer with either dense or sparse attention.

The sparse attention implemented here is for nodes to attend only to themselves
and their neighbours on the graph). It assumes that the adjacency matrix has a
banded structure, and is implemented with dense operations computing with only
the diagonal, super diagonal, and subdiagonal blocks of the tri-block-diagonal
attention matrix.

The basic model structure of the transformer and some functions were adapted
from xlm's transformer_simple.py.
"""

import dataclasses
import logging
from typing import Any, Callable, Literal, Optional, Tuple

from common import mlp as mlp_builder
from gencast import sparse_transformer_utils as utils

import flax.nnx as nnx
import jax
from jax.experimental.pallas.ops.tpu import splash_attention
import jax.numpy as jnp
import numpy as np
import scipy as sp


@dataclasses.dataclass
class _ModelConfig:
  """Transformer config."""
  # Depth, or num transformer blocks. One 'layer' is attn + ffw.
  num_layers: int
  # Primary width, the number of channels on the carrier path.
  d_model: int
  # Number of heads for self-attention.
  num_heads: int
  # Mask block size.
  mask_block_size: int
  # Attention type - 'mha' or 'triblockdiag_mha'
  attention_type: str = 'triblockdiag_mha'
  block_q: Optional[int] = None
  block_kv: Optional[int] = None
  block_kv_compute: Optional[int] = None
  block_q_dkv: Optional[int] = None
  block_kv_dkv: Optional[int] = None
  block_kv_dkv_compute: Optional[int] = None
  # mask type if splash attention being used - 'full' or 'lazy'
  mask_type: Optional[str] = 'full'
  # Number of channels per-head for self-attn QK computation.
  key_size: Optional[int] = None
  # Number of channels per-head for self-attn V computation.
  value_size: Optional[int] = None
  # Activation to use, any in jax.nn.
  activation: str = 'gelu'
  # Init scale for ffw layers (divided by num_layers)
  ffw_winit_mult: float = 2.0
  # Init scale for final ffw layer (divided by depth)
  ffw_winit_final_mult: float = 2.0
  # Init scale for mha proj (divided by depth).
  attn_winit_mult: float = 2.0
  # Init scale for mha w (divided by depth).
  attn_winit_final_mult: float = 2.0
  # Number of hidden units in the MLP blocks. Defaults to 4 * d_model.
  ffw_hidden: Optional[int] = None

  def __post_init__(self):
    if self.ffw_hidden is None:
      self.ffw_hidden = 4 * self.d_model
    # Compute key_size and value_size from d_model // num_heads.
    if self.key_size is None:
      if self.d_model % self.num_heads != 0:
        raise ValueError('num_heads has to divide d_model exactly')
      self.key_size = self.d_model // self.num_heads
    if self.value_size is None:
      if self.d_model % self.num_heads != 0:
        raise ValueError('num_heads has to divide d_model exactly')
      self.value_size = self.d_model // self.num_heads


################################################# Attention utils ################################################

def get_mask_block_size(mask: sp.sparse.csr_matrix) -> int:
  """Get blocksize of the adjacency matrix (attn mask) for the permuted mesh."""
  # sub-diagonal bandwidth
  lbandwidth = (
      np.arange(mask.shape[0]) - (mask != 0).argmax(axis=0) + 1).max()
  # super-diagonal bandwidth
  ubandwidth = (
      (mask.shape[0]-1) - np.argmax(mask[::-1,:] != 0, axis=0
                                    ) - np.arange(mask.shape[0]) + 1).max()
  block_size = np.maximum(lbandwidth, ubandwidth)
  return block_size



def triblockdiag_softmax(logits: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
                         ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Softmax given the diag, upper diag, and lower diag logit blocks."""

  logits_d, logits_u, logits_l = logits

  m = jnp.max(jnp.stack([
      jax.lax.stop_gradient(logits_d.max(-1, keepdims=True)),
      jax.lax.stop_gradient(logits_u.max(-1, keepdims=True)),
      jax.lax.stop_gradient(logits_l.max(-1, keepdims=True))]), axis=0)

  unnormalized_d = jnp.exp(logits_d - m)
  unnormalized_u = jnp.exp(logits_u - m)
  unnormalized_l = jnp.exp(logits_l - m)

  denom = (
      unnormalized_d.sum(-1, keepdims=True)
      + unnormalized_u.sum(-1, keepdims=True)
      + unnormalized_l.sum(-1, keepdims=True)
  )

  logits_d = unnormalized_d / denom
  logits_u = unnormalized_u / denom
  logits_l = unnormalized_l / denom

  return (logits_d, logits_u, logits_l)


def _make_splash_mha(
    mask,
    mask_type: str,
    num_heads: int,
    block_q: Optional[int] = None,
    block_kv: Optional[int] = None,
    block_kv_compute: Optional[int] = None,
    block_q_dkv: Optional[int] = None,
    block_kv_dkv: Optional[int] = None,
    block_kv_dkv_compute: Optional[int] = None,
    tanh_soft_cap: Optional[float] = None,
) -> Callable[..., jnp.ndarray]:
  """Construct attention kernel."""
  if mask_type == 'full':
    mask = np.broadcast_to(mask[None],
                           (num_heads, *mask.shape)).astype(np.bool_)

  block_sizes = splash_attention.BlockSizes(
      block_q=block_q,
      block_kv=block_kv,
      block_kv_compute=block_kv_compute,
      block_q_dkv=block_q_dkv,
      block_kv_dkv=block_kv_dkv,
      block_kv_dkv_compute=block_kv_dkv_compute,
      use_fused_bwd_kernel=True,
  )
  attn = splash_attention.make_splash_mha(mask, block_sizes=block_sizes,
                                          head_shards=1,
                                          q_seq_shards=1,
                                          attn_logits_soft_cap=tanh_soft_cap,
                                          )
  return attn



def mask_block_diags(mask: sp.sparse.csr_matrix,
                     num_padding_nodes: int,
                     block_size: int) -> jnp.ndarray:
  """Pad and reshape mask diag, super-siag and sub-diag blocks."""
  # add zero padding to mask
  mask_padding_rows = sp.sparse.csr_matrix(
      (num_padding_nodes, mask.shape[1]), dtype=jnp.int32)
  mask = sp.sparse.vstack([mask, mask_padding_rows])
  mask_padding_cols = sp.sparse.csr_matrix(
      (mask.shape[0], num_padding_nodes), dtype=jnp.int32)
  mask = sp.sparse.hstack([mask, mask_padding_cols])

  assert (mask.shape[-1] % block_size) == 0
  mask_daig_blocks = jnp.stack(
      [jnp.array(mask[i * block_size : (i + 1) * block_size,
                      i * block_size : (i + 1) * block_size,
                      ].toarray())
       for i in range(mask.shape[0] // block_size)])
  mask_upper_diag_blocks = jnp.stack(
      [jnp.array(mask[i * block_size : (i + 1) * block_size,
                      (i + 1) * block_size : (i + 2) * block_size,
                      ].toarray())
       for i in range(mask.shape[0] // block_size - 1)]
      + [jnp.zeros((block_size, block_size), dtype=mask.dtype)])
  mask_lower_diag_blocks = jnp.stack(
      [jnp.zeros((block_size, block_size), dtype=mask.dtype)]
      + [jnp.array(mask[(i + 1) * block_size : (i + 2) * block_size,
                        i * block_size : (i + 1) * block_size,
                        ].toarray())
         for i in range(mask.shape[0] // block_size - 1)])
  mask = jnp.stack(
      [mask_daig_blocks, mask_upper_diag_blocks, mask_lower_diag_blocks]
  )
  mask = jnp.expand_dims(mask, (0, 3))
  return mask



def _pad_mask(mask, num_padding_nodes: Tuple[int, int]) -> jnp.ndarray:
  q_padding, kv_padding = num_padding_nodes
  mask_padding_rows = sp.sparse.csr_matrix(
      (q_padding, mask.shape[1]), dtype=np.bool_)
  mask = sp.sparse.vstack([mask, mask_padding_rows])
  mask_padding_cols = sp.sparse.csr_matrix(
      (mask.shape[0], kv_padding), dtype=np.bool_)
  mask = sp.sparse.hstack([mask, mask_padding_cols])
  return mask



class WeatherMeshMask(splash_attention.splash_attention_mask.Mask):
  """Lazy local mask, prevent attention to embeddings outside window.

  Attributes:
    mask:
  """

  _shape: Tuple[int, int]
  mask: sp.sparse.spmatrix

  def __init__(
      self,
      mask: Any
  ):
    self._shape = mask.shape
    self.mask = mask

  @property
  def shape(self) -> Tuple[int, int]:
    return self._shape

  def __getitem__(self, idx) -> np.ndarray:
    if len(idx) != 2:
      raise NotImplementedError(f'Unsupported slice: {idx}')
    q_slice, kv_slice = idx
    if not isinstance(q_slice, slice) or not isinstance(kv_slice, slice):
      raise NotImplementedError(f'Unsupported slice: {idx}')

    return self.mask[q_slice, kv_slice].toarray()




############################################# NNX Modules #############################################

class FeedForward(nnx.Module):
  """Feed-forward block."""
  
  def __init__(self, cfg: _ModelConfig, rngs: nnx.Rngs):
    ffw_winit = nnx.initializers.variance_scaling(cfg.ffw_winit_mult / cfg.num_layers, 
                                                  'fan_in', 'truncated_normal')
    ffw_winit_final = nnx.initializers.variance_scaling(cfg.ffw_winit_final_mult / cfg.num_layers, 
                                                  'fan_in', 'truncated_normal')
    self.mlp = nnx.Sequential(
      nnx.Linear(in_features=cfg.d_model, out_features=cfg.ffw_hidden, 
                 kernel_init=ffw_winit, rngs=rngs),
      getattr(nnx, cfg.activation),
      nnx.Linear(in_features=cfg.ffw_hidden, out_features=cfg.d_model,
                 kernel_init=ffw_winit_final, rngs=rngs),
    )
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    return self.mlp(x)


class MultiheadLinear(nnx.Module):
  """Linearly project `x` to have `head_size` dimensions per head."""
  def __init__(self, qkv: str, cfg: _ModelConfig, *, rngs: nnx.Rngs):
    head_size = cfg.value_size if qkv == 'v' else cfg.key_size
    attn_winit = nnx.initializers.variance_scaling(cfg.attn_winit_mult / cfg.num_layers, 
                                                  'fan_in', 'truncated_normal')
    self.linear = nnx.Linear(
        cfg.d_model, # Input features
        cfg.num_heads * head_size,
        kernel_init=attn_winit,
        use_bias=False,
        rngs=rngs,
    )
    self.head_size = head_size
    self.num_heads = cfg.num_heads

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    out = self.linear(x)
    shape = out.shape[:-1] + (self.num_heads, self.head_size)
    return jnp.reshape(out, shape)



class TriblockdiagMHA(nnx.Module):
  """Triblockdiag multihead attention."""
  def __init__(self, cfg: _ModelConfig, *, rngs: nnx.Rngs):
    self.q_proj = MultiheadLinear('q', cfg, rngs=rngs)
    self.k_proj = MultiheadLinear('k', cfg, rngs=rngs)
    self.v_proj = MultiheadLinear('v', cfg, rngs=rngs)

    attn_winit_final = nnx.initializers.variance_scaling(self._cfg.attn_winit_final_mult / self._cfg.num_layers, 
                                                        'fan_in', 'truncated_normal')
    self.final_linear = nnx.Linear(in_features=self._cfg.num_heads * self._cfg.value_size, 
                                   out_features=self._cfg.d_model, 
                                   kernel_init=attn_winit_final, 
                                   rngs=self.rngs)

    self._cfg = cfg

  def __call__(self, q_input: jnp.ndarray, kv_input: jnp.ndarray,
               mask: jnp.ndarray) -> jnp.ndarray:
    # q_inputs, kv_input: (batch, num_blocks, block_size, num_heads, d_model)
    q = self.q_proj(q_input)
    k = self.k_proj(kv_input)
    v = self.v_proj(kv_input)

    k = jnp.pad(k, ((0, 0), (1, 1), (0, 0), (0, 0), (0, 0)))
    v = jnp.pad(v, ((0, 0), (1, 1), (0, 0), (0, 0), (0, 0)))

    def qk_prod(queries, keys):
      return jnp.einsum('bnqhd,bnkhd->bnhqk', queries, keys)

    # q shape is (batch, num_blocks, block_size, num_heads, qk_dim)
    # k shape is (batch, num_blocks + 2, block_size, num_heads, qk_dim)
    logits_d = qk_prod(q, k[:, 1:-1, ...]) * self._cfg.key_size**-0.5
    logits_u = qk_prod(q, k[:, 2:, ...]) * self._cfg.key_size**-0.5
    logits_l = qk_prod(q, k[:, :-2, ...]) * self._cfg.key_size**-0.5

    # apply mask
    logits_d = jnp.where(mask[:, 0, ...], logits_d, -1e30)
    logits_u = jnp.where(mask[:, 1, ...], logits_u, -1e30)
    logits_l = jnp.where(mask[:, 2, ...], logits_l, -1e30)

    logits_d, logits_u, logits_l = utils.wrap_fn_for_upcast_downcast(
        (logits_d, logits_u, logits_l),
        triblockdiag_softmax
        )

    def av_prod(attn_weights, values):
      return jnp.einsum('bnhqk,bnkhd->bnqhd', attn_weights, values)

    out_d = av_prod(logits_d, v[:, 1:-1, ...])
    out_u = av_prod(logits_u, v[:, 2:, ...])
    out_l = av_prod(logits_l, v[:, :-2, ...])
    # x shape is (batch, num_blocks, block_size, num_heads, d_model)
    x = out_d + out_u + out_l

    x = jnp.reshape(x, x.shape[:-2] + (self._cfg.num_heads * self._cfg.value_size,))

    x = self.final_linear(x)
    return x



class MHA(nnx.Module):
  """Multi head attention."""
  def __init__(self, cfg: _ModelConfig, *, rngs: nnx.Rngs):
    self.q_proj = MultiheadLinear('q', cfg, rngs=rngs)
    self.k_proj = MultiheadLinear('k', cfg, rngs=rngs)
    self.v_proj = MultiheadLinear('v', cfg, rngs=rngs)

    attn_winit_final = nnx.initializers.variance_scaling(self._cfg.attn_winit_final_mult / self._cfg.num_layers, 
                                                         'fan_in', 'truncated_normal')
    self.final_linear = nnx.Linear(in_features=self._cfg.num_heads * self._cfg.value_size, 
                                   out_features=self._cfg.d_model, 
                                   kernel_init=attn_winit_final, 
                                   rngs=self.rngs)

    self._cfg = cfg

  def __call__(self, q_input: jnp.ndarray, kv_input: jnp.ndarray,
               mask: jnp.ndarray, normalize_logits: bool = True) -> jnp.ndarray:

    q = self.q_proj(q_input)
    k = self.k_proj(kv_input)
    v = self.v_proj(kv_input)

    logits = jnp.einsum('bthd, bThd->bhtT', q, k)
    if normalize_logits:
      logits *= self._cfg.key_size**-0.5
    if mask is not None:
      def apply_mask(m, l):
        return jnp.where(m, l, -1e30)
      logits = jax.vmap(jax.vmap(
          apply_mask, in_axes=[None, 0]), in_axes=[None, 0])(mask, logits)

    # Wrap softmax weights for upcasting & downcasting in case of BF16 activations
    weights = utils.wrap_fn_for_upcast_downcast(logits, jax.nn.softmax)

    # Note: our mask never has all 0 rows, since nodes always have self edges,
    # so no need to account for that possibility explicitly.

    x = jnp.einsum('bhtT,bThd->bthd', weights, v)
    x = jnp.reshape(x, x.shape[:-2] + (self._cfg.num_heads * self._cfg.value_size,))

    x = self.final_linear(x)
    return x


class SplashMHA(nnx.Module):
  """Splash attention."""
  def __init__(self, cfg: _ModelConfig, *, rngs: nnx.Rngs):
    self.q_proj = MultiheadLinear('q', cfg, rngs=rngs)
    self.k_proj = MultiheadLinear('k', cfg, rngs=rngs)
    self.v_proj = MultiheadLinear('v', cfg, rngs=rngs)

    attn_winit_final = nnx.initializers.variance_scaling(self._cfg.attn_winit_final_mult / self._cfg.num_layers, 
                                                         'fan_in', 'truncated_normal')

    self.final_linear = nnx.Linear(in_features=self._cfg.num_heads * self._cfg.value_size, 
                              out_features=self._cfg.d_model, 
                              kernel_init=attn_winit_final, 
                              rngs=self.rngs)

    self._cfg = cfg

  def __call__(self, q_input: jnp.ndarray, kv_input: jnp.ndarray,
               mask: jnp.ndarray | splash_attention.splash_attention_mask.Mask,
               tanh_soft_cap: Optional[float] = None,
               normalize_q: bool = True) -> jnp.ndarray:

    q = self.q_proj(q_input)
    k = self.k_proj(kv_input)
    v = self.v_proj(kv_input)

    _, _, num_heads, head_dim = q.shape

    assert head_dim % 128 == 0  # splash attention kernel requires this

    attn = _make_splash_mha(
        mask=mask,
        mask_type=self._cfg.mask_type,
        num_heads=num_heads,
        block_q=self._cfg.block_q,
        block_kv=self._cfg.block_kv,
        block_kv_compute=self._cfg.block_kv_compute,
        block_q_dkv=self._cfg.block_q_dkv,
        block_kv_dkv=self._cfg.block_kv_dkv,
        block_kv_dkv_compute=self._cfg.block_kv_dkv_compute,
        tanh_soft_cap=tanh_soft_cap,
    )
    attn = jax.vmap(attn)  # Add batch axis.

    if normalize_q:
      q *= self._cfg.key_size**-0.5

    # (batch, nodes, num_heads, head_dim) -> (batch, num_heads, nodes, head_dim)
    reformat = lambda y: y.transpose(0, 2, 1, 3)
    x = attn(q=reformat(q), k=reformat(k), v=reformat(v))
    x = x.transpose(0, 2, 1, 3)

    x = jnp.reshape(x, x.shape[:-2] + (self._cfg.num_heads * self._cfg.value_size,))

    x = self.final_linear(x)
    return x


class Block(nnx.Module):
  """Transformer block (mha and ffw)."""

  def __init__(self, cfg: _ModelConfig, mask: jnp.ndarray, num_nodes: int,
               num_padding_nodes: Tuple[int, int] | int, *, rngs: nnx.Rngs):
    self._cfg = cfg
    self.mask = mask
    self.num_nodes = num_nodes
    self.num_padding_nodes = num_padding_nodes

    # Instantiate attention and FFW modules in __init__
    if self._cfg.attention_type == 'triblockdiag_mha':
      self.attn_module = TriblockdiagMHA(cfg, rngs=rngs)
    elif self._cfg.attention_type == 'mha':
      self.attn_module = MHA(cfg, rngs=rngs)
    elif self._cfg.attention_type == 'splash_mha':
      self.attn_module = SplashMHA(cfg, rngs=rngs)
    else:
      raise NotImplementedError()

    self.ffw_module = FeedForward(cfg, rngs=rngs)

    # Instantiate LinearNormConditioning and LayerNorm
    self.norm_cond_attn = mlp_builder.LinearNormConditioning(feature_size=self._cfg.d_model, rngs=rngs)
    self.norm_cond_ffw = mlp_builder.LinearNormConditioning(feature_size=self._cfg.d_model, rngs=rngs)
    self.ln1 = nnx.LayerNorm(features=self._cfg.d_model, use_scale=False, use_bias=False, rngs=rngs)
    self.ln2 = nnx.LayerNorm(features=self._cfg.d_model, use_scale=False, use_bias=False, rngs=rngs)


  def __call__(self, x: jnp.ndarray, global_norm_conditioning: jax.Array):
    # x shape is (batch, num_nodes, feature_dim)
    def call_attn(x_in):
      if self._cfg.attention_type == 'triblockdiag_mha':
        # We pad -> reshape -> compute attn -> reshape -> select at each block
        # so as to avoid complications involved in making the norm layers and
        # ffw blocks account for the padding. However, this might be decreasing
        # efficiency.
        x_padded = jnp.pad(x_in, ((0, 0), (0, self.num_padding_nodes), (0, 0)))
        x_reshaped = x_padded.reshape(x_padded.shape[0],
                                      x_padded.shape[1]//self._cfg.mask_block_size,
                                      self._cfg.mask_block_size,
                                      x_padded.shape[-1])
        x_attn = self.attn_module(x_reshaped, x_reshaped, mask=self.mask)
        x_attn_reshaped = x_attn.reshape(x_attn.shape[0],
                                          self.num_nodes + self.num_padding_nodes,
                                          x_attn.shape[-1])
        return x_attn_reshaped[:,:self.num_nodes, :]

      elif self._cfg.attention_type == 'mha':
        return self.attn_module(x_in, x_in, mask=self.mask)

      elif self._cfg.attention_type == 'splash_mha':
        # Add padding so that number of nodes is divisible by block sizes.
        x_padded = jnp.pad(x_in, ((0, 0), (0, self.num_padding_nodes[0]), (0, 0)))
        x_attn = self.attn_module(x_padded, x_padded, mask=self.mask)
        return x_attn[:,:self.num_nodes, :]

      else:
        raise NotImplementedError()

    normed_x_attn = self.ln1(x)
    conditioned_x_attn = self.norm_cond_attn(normed_x_attn, norm_conditioning=jnp.expand_dims(global_norm_conditioning, 1))
    x = x + call_attn(conditioned_x_attn)

    normed_x_ffw = self.ln2(x)
    conditioned_x_ffw = self.norm_cond_ffw(normed_x_ffw, norm_conditioning=jnp.expand_dims(global_norm_conditioning, 1))
    x = x + self.ffw_module(conditioned_x_ffw)
    return x


class Transformer(nnx.Module):
  """Main transformer module that processes embeddings.

  All but the very first and very last layer of a 'classic' Transformer:
  Receives already embedded inputs instead of discrete tokens.
  Outputs an embedding for each 'node'/'position' rather than logits.
  """

  def __init__(self,
               adj_mat: sp.sparse.csr_matrix,
               attention_k_hop: int,
               attention_type: Literal['splash_mha', 'triblockdiag_mha', 'mha'],
               mask_type: Literal['full', 'lazy'],
               num_heads: int = 1,
               *,
               rngs: nnx.Rngs,
               block_q: Optional[int] = None,
               block_kv: Optional[int] = None,
               block_kv_compute: Optional[int] = None,
               block_q_dkv: Optional[int] = None,
               block_kv_dkv: Optional[int] = None,
               block_kv_dkv_compute: Optional[int] = None,
               **kwargs):

    # Construct mask and deduce block size.
    mask = adj_mat ** attention_k_hop
    mask_block_size = get_mask_block_size(mask)
    logging.info('mask_block_size: %s.', mask_block_size)

    if attention_type == 'triblockdiag_mha':
      # we will stack the nodes in blocks of 'block_size' nodes, so we need to
      # pad the input such that (num_nodes + num_padding_nodes) % block_size = 0
      self.num_padding_nodes = int(np.ceil(
          mask.shape[0]/mask_block_size)*mask_block_size
                                   - mask.shape[0])
      self.mask = mask_block_diags(
          mask, self.num_padding_nodes, mask_block_size)
    elif attention_type == 'splash_mha':
      max_q_block_size = np.maximum(block_q, block_q_dkv)
      max_kv_block_size = np.maximum(block_kv, block_kv_dkv)
      q_padding = int(np.ceil(
          mask.shape[0]/max_q_block_size)*max_q_block_size - mask.shape[0])
      kv_padding = int(np.ceil(
          mask.shape[1]/max_kv_block_size)*max_kv_block_size - mask.shape[1])
      self.num_padding_nodes = (q_padding, kv_padding)
      mask = _pad_mask(mask, self.num_padding_nodes)
      if mask_type == 'lazy':
        splash_mask = [
            WeatherMeshMask(mask)
            for _ in range(num_heads)
        ]
        self.mask = splash_attention.splash_attention_mask.MultiHeadMask(
            splash_mask)
      elif mask_type == 'full':
        self.mask = mask.toarray()
    elif attention_type == 'mha':
      self.mask = jnp.array(mask.toarray())
      self.num_padding_nodes = 0
    else:
      raise ValueError(
          'Unsupported attention type: %s' % attention_type
      )

    # Construct config for use within class.
    self._cfg = _ModelConfig(
        mask_block_size=mask_block_size,
        attention_type=attention_type,
        mask_type=mask_type,
        num_heads=num_heads,
        block_q=block_q,
        block_kv=block_kv,
        block_kv_compute=block_kv_compute,
        block_q_dkv=block_q_dkv,
        block_kv_dkv=block_kv_dkv,
        block_kv_dkv_compute=block_kv_dkv_compute,
        **kwargs)

    # Instantiate all Block modules
    self.blocks = []
    for _ in range(self._cfg.num_layers):
      self.blocks.append(
          Block(cfg=self._cfg, 
                mask=self.mask,
                num_nodes=adj_mat.shape[0], # Pass actual num_nodes from adj_mat
                num_padding_nodes=self.num_padding_nodes,
                rngs=rngs # Pass rngs to sub-modules
                )
      )

    # Final LayerNorm and LinearNormConditioning
    self.final_ln = nnx.LayerNorm(features=self._cfg.d_model, use_scale=False, use_bias=False, rngs=rngs)
    self.final_norm_cond = mlp_builder.LinearNormConditioning(
        feature_size=self._cfg.d_model, rngs=rngs)

  def __call__(self, node_features: jnp.ndarray, global_norm_conditioning: jax.Array):
    # node_features expected to have shape (batch, num_nodes, d)
    x = node_features
    for block in self.blocks:
      x = block(x, global_norm_conditioning=global_norm_conditioning)

    x = self.final_norm_cond(
        self.final_ln(x),
        norm_conditioning=jnp.expand_dims(global_norm_conditioning, 1)
    )
    return x