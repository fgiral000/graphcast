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
"""Constructors for MLPs."""
import jax
import jax.numpy as jnp
import jraph
import flax.nnx as nnx

import functools
from typing import Optional


### Flax NNX modules ###
class LinearNormConditioning(nnx.Module):
  """Module for norm conditioning.

  Conditions the normalization of "inputs" by applying a linear layer to the
  "norm_conditioning" which produces the scale and variance which are applied to
  each channel (across the last dim) of "inputs".

  NOTE: This is a reimplementation of the Haiku module using Flax NNX.
  """

  def __init__(self, feature_size: int, rngs: nnx.Rngs):
    self.feature_size = feature_size
    self.conditional_linear_layer = nnx.Linear(
      input_size=self.feature_size,
      output_size=2 * self.feature_size,
      kernel_init=nnx.initializers.truncated_normal(stddev=1e-8),
      rngs=rngs,
    )
  
  def __call__(self, inputs: jax.Array, norm_conditioning: jax.Array):
    conditional_scale_offset = self.conditional_linear_layer(norm_conditioning)
    scale_minus_one, offset = jnp.split(conditional_scale_offset, 2, axis=-1)
    scale = scale_minus_one + 1.
    return inputs * scale + offset
  

    

def build_mlp(mlp_input_size, 
              mlp_hidden_size, 
              mlp_num_hidden_layers,
              mlp_output_size, 
              activation,
              *,
              rngs: nnx.Rngs,): 
  layers = []
  feature_size = mlp_input_size
  for _ in range(mlp_num_hidden_layers):
    layers.append(nnx.Linear(feature_size, mlp_hidden_size, rngs=rngs))
    feature_size = mlp_hidden_size
    layers.append(activation)
  layers.append(nnx.Linear(mlp_hidden_size, mlp_output_size, rngs=rngs))
  return jraph.concatenated_args(nnx.Sequential(*layers))



def build_mlp_with_maybe_layer_norm( mlp_input_size: int,
    mlp_hidden_size: int,
    mlp_num_hidden_layers: int,
    mlp_output_size: int,
    activation,
    *,
    use_layer_norm: bool,
    use_norm_conditioning: bool,
    global_norm_conditioning: Optional[jax.Array] = None,
    rngs: nnx.Rngs,
):
  """Builds an MLP, optionally with LayerNorm and norm-conditioning."""
  network = build_mlp(
      mlp_input_size=mlp_input_size,
      mlp_hidden_size=mlp_hidden_size,
      mlp_num_hidden_layers=mlp_num_hidden_layers,
      mlp_output_size=mlp_output_size,
      activation=activation,
      rngs=rngs,
  )
  # If one if used, the other cannot be used.
  assert not (use_layer_norm and use_norm_conditioning), (
      "Cannot use both `use_layer_norm` and `use_norm_conditioning` at the same"
      " time. Please choose one of them.")
  
  stages = [network]
  if use_norm_conditioning:
    if global_norm_conditioning is None:
      raise ValueError(
          "When using norm conditioning, `global_norm_conditioning` must"
          "be passed to the call method.")
    # If using norm conditioning, it is no longer the responsibility of the
    # LayerNorm module itself to learn its scale and offset. These will be
    # learned for the module by the norm conditioning layer instead.
  else:
    if global_norm_conditioning is not None:
      raise ValueError(
          "`globa_norm_conditioning` was passed, but `norm_conditioning`"
          " is not enabled.")

  if use_layer_norm:
    layer_norm = nnx.LayerNorm(num_features=mlp_output_size,
                               feature_axes=-1,
                               rngs=rngs)
    stages.append(layer_norm)

  if use_norm_conditioning:
    norm_conditioning_layer = LinearNormConditioning(feature_size=mlp_output_size,
                                                     rngs=rngs)
    norm_conditioning_layer = functools.partial(
        norm_conditioning_layer,
        # Broadcast to the node/edge axis.
        norm_conditioning=global_norm_conditioning[None],
    )
    stages.append(norm_conditioning_layer)

  network = nnx.Sequential(*stages)
  return jraph.concatenated_args(network)