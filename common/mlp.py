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
      in_features=self.feature_size,
      out_features=2 * self.feature_size,
      kernel_init=nnx.initializers.truncated_normal(stddev=1e-8),
      rngs=rngs,
    )
  
  def __call__(self, inputs: jax.Array, norm_conditioning: jax.Array):
    conditional_scale_offset = self.conditional_linear_layer(norm_conditioning)
    scale_minus_one, offset = jnp.split(conditional_scale_offset, 2, axis=-1)
    scale = scale_minus_one + 1.
    return inputs * scale + offset
  


class MLP(nnx.Module): 
  """A simple MLP module."""
  def __init__(self,
               mlp_input_size: int,
               mlp_hidden_size: int,
               mlp_num_hidden_layers: int,
               mlp_output_size: int,
               activation,
               *,
               rngs: nnx.Rngs):
    """Initializes the MLP module."""
    layers = []
    feature_size = mlp_input_size
    for _ in range(mlp_num_hidden_layers):
      layers.append(nnx.Linear(feature_size, mlp_hidden_size, rngs=rngs))
      feature_size = mlp_hidden_size
      layers.append(activation)
    layers.append(nnx.Linear(mlp_hidden_size, mlp_output_size, rngs=rngs))
    self.network = nnx.Sequential(*layers)
  
  def __call__(self, inputs: jax.Array):
    return self.network(inputs)


class MLPWithNormConditioning(nnx.Module):
  """An MLP module with norm conditioning."""
  def __init__(self,
               mlp_input_size: int,
               mlp_hidden_size: int,
               mlp_num_hidden_layers: int,
               mlp_output_size: int,
               activation,
               *,
               use_layer_norm: bool,
               use_norm_conditioning: bool,
               rngs: nnx.Rngs):
    """Initializes the MLP with norm conditioning."""
    
    self._use_layer_norm = use_layer_norm
    self._use_norm_conditioning = use_norm_conditioning

    self.network = MLP(
        mlp_input_size=mlp_input_size,
        mlp_hidden_size=mlp_hidden_size,
        mlp_num_hidden_layers=mlp_num_hidden_layers,
        mlp_output_size=mlp_output_size,
        activation=activation,
        rngs=rngs,
    )

    if self._use_norm_conditioning:
      # If using norm conditioning, it is no longer the responsibility of the
      # LayerNorm module itself to learn its scale and offset. These will be
      # learned for the module by the norm conditioning layer instead.
      create_scale = create_offset = False
    else:
      create_scale = create_offset = True

    if self._use_layer_norm:
      self.layer_norm = nnx.LayerNorm(num_features=mlp_output_size,
                                use_scale=create_scale,
                                use_bias=create_offset,
                                feature_axes=-1,
                                rngs=rngs)

    if self._use_norm_conditioning:
      self.norm_conditioning_layer = LinearNormConditioning(feature_size=mlp_output_size,
                                                      rngs=rngs)

  
  def __call__(self, inputs: jax.Array, global_norm_conditioning: Optional[jax.Array] = None):

    if self._use_norm_conditioning:
      if global_norm_conditioning is None:
        raise ValueError(
            "When using norm conditioning, `global_norm_conditioning` must"
            "be passed to the call method.")
    else:
      if global_norm_conditioning is not None:
        raise ValueError(
            "`globa_norm_conditioning` was passed, but `norm_conditioning`"
            " is not enabled.")

    x = self.network(inputs)
    if self._use_layer_norm:
      x = self.layer_norm(x)
      if self._use_norm_conditioning:
        #broadcast the global norm conditioning to match the batch size
        global_norm_conditioning = global_norm_conditioning[None]
        x = self.norm_conditioning_layer(x, global_norm_conditioning)

    return x


    
    




if __name__ == "__main__":
  # Example usage
  rngs = nnx.Rngs(0)

  mlp = MLPWithNormConditioning(
      mlp_input_size=10,
      mlp_hidden_size=20,
      mlp_num_hidden_layers=3,
      mlp_output_size=5,
      activation=nnx.relu,
      use_layer_norm=True,
      use_norm_conditioning=True,
      rngs=rngs
  )
  # Visualize the model architecture
  # nnx.display(mlp)
  # This will print the structure of the MLP.

  # Let's use some dummy data to test the MLP
  dummy_input = jnp.ones((1,10))  # Batch size of 1, input size of 10
  global_norm_conditioning = jnp.ones((5,))  # Example global norm conditioning
  


  output = mlp(dummy_input, global_norm_conditioning=global_norm_conditioning)
  print("Output shape:", output.shape)  # Should be (1, 5) for the output size of 5