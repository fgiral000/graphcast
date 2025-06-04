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
"""A Transformer model for weather predictions.

This model wraps the transformer model and swaps the leading two axes of the
nodes in the input graph prior to evaluating the model to make it compatible
with a [nodes, batch, ...] ordering of the inputs.
"""

from typing import Any, Mapping, Optional, Type

from common import typed_graph
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse

Kwargs = Mapping[str, Any]


def _get_adj_matrix_for_edge_set(
    graph: typed_graph.TypedGraph,
    edge_set_name: str,
    add_self_edges: bool,
):
  """Returns the adjacency matrix for the given graph and edge set."""
  # Get nodes and edges of the graph.
  edge_set_key = graph.edge_key_by_name(edge_set_name)
  sender_node_set, receiver_node_set = edge_set_key.node_sets

  # Compute number of sender and receiver nodes.
  sender_n_node = graph.nodes[sender_node_set].n_node[0]
  receiver_n_node = graph.nodes[receiver_node_set].n_node[0]

  # Build adjacency matrix.
  adj_mat = sparse.csr_matrix((sender_n_node, receiver_n_node), dtype=np.bool_)
  edge_set = graph.edges[edge_set_key]
  s, r = edge_set.indices
  adj_mat[s, r] = True
  if add_self_edges:
    # Should only do this if we are certain the adjacency matrix is square.
    assert sender_node_set == receiver_node_set
    adj_mat[np.arange(sender_n_node), np.arange(receiver_n_node)] = True
  return adj_mat


# Let's define a type alias for the Transformer class.

from typing import TypeVar
Transformer = TypeVar('Transformer', bound=nnx.Module)

class MeshTransformer(nnx.Module):
  """A Transformer for inputs with ordering [nodes, batch, ...]."""

  # Store the transformer_ctor as a type hint, not an instantiated module.
  _transformer_ctor: Type[Transformer]
  _transformer_kwargs: Kwargs

  def __init__(self,
               transformer_ctor: Type[Transformer], # Expects the NNX Transformer class
               transformer_kwargs: Kwargs,
               *, # rngs must be a keyword argument
               rngs: nnx.Rngs,
               ):
    """Initialises the Transformer model.

    Args:
      transformer_ctor: Constructor for transformer (the NNX Transformer class).
      transformer_kwargs: Kwargs to pass to the transformer module.
      rngs: The PRNG key for initializing any submodules.
      name: Optional name for nnx module.
    """
    # We store the constructor and kwargs. The actual transformer module
    # will be initialized lazily in the first call.
    self._transformer_ctor = transformer_ctor
    self._transformer_kwargs = transformer_kwargs
    # The actual transformer instance, initialized to None.
    # This will be an nnx.Module when instantiated.
    self.batch_first_transformer: Optional[Transformer] = None

    # Store the rngs for later use when initializing the sub-module.
    # This is crucial for NNX's state management.
    self.rngs = rngs

  def __call__(
      self, x: typed_graph.TypedGraph,
      global_norm_conditioning: jax.Array
  ) -> typed_graph.TypedGraph:
    """Applies the model to the input graph and returns graph of same shape."""

    if set(x.nodes.keys()) != {'mesh_nodes'}:
      raise ValueError(
          f'Expected x.nodes to have key `mesh_nodes`, got {x.nodes.keys()}.'
      )
    features = x.nodes['mesh_nodes'].features
    if features.ndim != 3:
      raise ValueError(
          'Expected `x.nodes["mesh_nodes"].features` to be 3, got'
          f' {features.ndim}.'
      )

    # Lazy initialization of the transformer.
    # This block will only run on the very first forward pass.
    if self.batch_first_transformer is None:

      self.batch_first_transformer = self._transformer_ctor(
          adj_mat=_get_adj_matrix_for_edge_set(
              graph=x,
              edge_set_name='mesh',
              add_self_edges=True,
          ),
          rngs=self.rngs, # Pass the derived rngs
          **self._transformer_kwargs,
      )

    y = jnp.transpose(features, axes=[1, 0, 2])
    y = self.batch_first_transformer(y, global_norm_conditioning)
    y = jnp.transpose(y, axes=[1, 0, 2])
    x = x._replace(
        nodes={
            'mesh_nodes': x.nodes['mesh_nodes']._replace(
                features=y.astype(features.dtype)
            )
        }
    )
    return x