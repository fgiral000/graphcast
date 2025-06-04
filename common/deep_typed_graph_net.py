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
"""JAX implementation of Graph Networks Simulator.

Generalization to TypedGraphs of the deep Graph Neural Network from:

@inproceedings{pfaff2021learning,
  title={Learning Mesh-Based Simulation with Graph Networks},
  author={Pfaff, Tobias and Fortunato, Meire and Sanchez-Gonzalez, Alvaro and
      Battaglia, Peter},
  booktitle={International Conference on Learning Representations},
  year={2021}
}

@inproceedings{sanchez2020learning,
  title={Learning to simulate complex physics with graph networks},
  author={Sanchez-Gonzalez, Alvaro and Godwin, Jonathan and Pfaff, Tobias and
      Ying, Rex and Leskovec, Jure and Battaglia, Peter},
  booktitle={International conference on machine learning},
  pages={8459--8468},
  year={2020},
  organization={PMLR}
}
"""

from __future__ import annotations

import functools
from typing import Callable, List, Mapping, Optional, Tuple, Type

import chex
from common import mlp as mlp_builder 
from common import typed_graph
from common import typed_graph_net

import flax.nnx as nnx

import jax
import jax.numpy as jnp
import jraph


GraphToGraphNetwork = Callable[[typed_graph.TypedGraph, Optional[chex.Array]], typed_graph.TypedGraph]

############################################### Helper Functions ###############################################
def _get_activation_fn(name):
  """Return activation function corresponding to function_name."""
  if name == "identity":
    return lambda x: x
  if hasattr(nnx, name):
    return getattr(nnx, name)
  if hasattr(jax.nn, name):
    return getattr(jax.nn, name)
  if hasattr(jnp, name):
    return getattr(jnp, name)
  raise ValueError(f"Unknown activation function {name} specified.")


def _get_aggregate_edges_for_nodes_fn(name):
  """Return aggregate_edges_for_nodes_fn corresponding to function_name."""
  if hasattr(jraph, name):
    return getattr(jraph, name)
  raise ValueError(
      f"Unknown aggregate_edges_for_nodes_fn function {name} specified.")


def _build_update_fns_for_node_types(
    *,
    builder_fn: Type[nnx.Module],
    graph_template: typed_graph.TypedGraph,
    mlp_hidden_size: int,
    mlp_num_hidden_layers: int,
    activation: Callable[[jnp.ndarray], jnp.ndarray],
    use_layer_norm: bool,
    use_norm_conditioning: bool,
    rngs: nnx.Rngs,
    input_sizes: Optional[Mapping[str, int]] = None,
    output_sizes: Optional[Mapping[str, int]] = None
) -> Mapping[str, nnx.Module]: # Returns a dict of instantiated NNX modules
  """Builds an update function for all node types or a subset of them."""
  output_fns = {}
  for node_set_name in graph_template.nodes.keys():
    output_size = None
    if output_sizes is not None:
      if node_set_name in output_sizes:
        output_size = output_sizes[node_set_name]
      else:
        continue # Skip if no explicit output size and output_sizes is provided

    current_input_size = None
    if input_sizes is None:
      current_input_size = graph_template.nodes[node_set_name].features.shape[-1]
    else:
      current_input_size = input_sizes[node_set_name]

    output_fns[node_set_name] = builder_fn(
      mlp_input_size=current_input_size,
      mlp_hidden_size=mlp_hidden_size,
      mlp_num_hidden_layers=mlp_num_hidden_layers,
      mlp_output_size=output_size,
      activation=activation,
      use_layer_norm=use_layer_norm,
      use_norm_conditioning=use_norm_conditioning,
      rngs=rngs,
    )
  return output_fns


def _build_update_fns_for_edge_types(
    *,
    builder_fn: Type[nnx.Module],
    graph_template: typed_graph.TypedGraph,
    mlp_hidden_size: int,
    mlp_num_hidden_layers: int,
    activation: Callable[[jnp.ndarray], jnp.ndarray],
    use_layer_norm: bool,
    use_norm_conditioning: bool,
    rngs: nnx.Rngs,
    input_sizes: Optional[Mapping[typed_graph.EdgeSetKey, int]] = None,
    output_sizes: Optional[Mapping[str, int]] = None
) -> Mapping[str, nnx.Module]: # Returns a dict of instantiated NNX modules
  """Builds an edge function for all edge types or a subset of them."""
  output_fns = {}
  for edge_set_key in graph_template.edges.keys():
    edge_set_name = edge_set_key.name
    output_size = None
    if output_sizes is not None:
      if edge_set_name in output_sizes:
        output_size = output_sizes[edge_set_name]
      else:
        continue # Skip if no explicit output size and output_sizes is provided

    current_input_size = None
    if input_sizes is None:
      current_input_size = graph_template.edges[edge_set_key].features.shape[-1]
    else:
      # Note: input_sizes for edges are typically keyed by EdgeSetKey, not just name string
      current_input_size = input_sizes[edge_set_name]

    output_fns[edge_set_name] = builder_fn(
        mlp_input_size=current_input_size,
        mlp_hidden_size=mlp_hidden_size,
        mlp_num_hidden_layers=mlp_num_hidden_layers,
        mlp_output_size=output_size,
        activation=activation,
        use_layer_norm=use_layer_norm,
        use_norm_conditioning=use_norm_conditioning,
        rngs=rngs,
    )
  return output_fns


class DeepTypedGraphNet(nnx.Module):
  """Deep Graph Neural Network.

  It works with TypedGraphs with typed nodes and edges. It runs message
  passing on all of the node sets and all of the edge sets in the graph. For
  each message passing step a `typed_graph_net.InteractionNetwork` is used to
  update the full TypedGraph by using different MLPs for each of the node sets
  and each of the edge sets.

  If embed_{nodes,edges} is specified the node/edge features will be embedded
  into a fixed dimensionality before running the first step of message passing.

  If {node,edge}_output_size the final node/edge features will be embedded into
  the specified output size.

  This class may be used for shared or unshared message passing:
  * num_message_passing_steps = N, num_processor_repetitions = 1, gives
    N layers of message passing with fully unshared weights:
    [W_1, W_2, ... , W_M] (default)
  * num_message_passing_steps = 1, num_processor_repetitions = M, gives
    N layers of message passing with fully shared weights:
    [W_1] * M
  * num_message_passing_steps = N, num_processor_repetitions = M, gives
    M*N layers of message passing with both shared and unshared message passing
    such that the weights used at each iteration are:
    [W_1, W_2, ... , W_N] * M

    NOTE: this is a reimplementation of the original Haiku module using Flax NNX.

  """

  def __init__(self,
               *,
               node_latent_size: Mapping[str, int],
               edge_latent_size: Mapping[str, int],
               mlp_hidden_size: int,
               mlp_num_hidden_layers: int,
               num_message_passing_steps: int,
               num_processor_repetitions: int = 1,
               embed_nodes: bool = True,
               embed_edges: bool = True,
               node_output_size: Optional[Mapping[str, int]] = None,
               edge_output_size: Optional[Mapping[str, int]] = None,
               include_sent_messages_in_node_update: bool = False,
               use_layer_norm: bool = True,
               use_norm_conditioning: bool = False,
               activation: str = "relu",
               f32_aggregation: bool = False,
               aggregate_edges_for_nodes_fn: str = "segment_sum",
               aggregate_normalization: Optional[float] = None,
               name: str = "DeepTypedGraphNet",
               rngs: nnx.Rngs):
    """Inits the model.

    Args:
      node_latent_size: Size of the node latent representations.
      edge_latent_size: Size of the edge latent representations.
      mlp_hidden_size: Hidden layer size for all MLPs.
      mlp_num_hidden_layers: Number of hidden layers in all MLPs.
      num_message_passing_steps: Number of unshared message passing steps
         in the processor steps.
      num_processor_repetitions: Number of times that the same processor is
         applied sequencially.
      embed_nodes: If False, the node embedder will be omitted.
      embed_edges: If False, the edge embedder will be omitted.
      node_output_size: Size of the output node representations for
         each node type. For node types not specified here, the latent node
         representation from the output of the processor will be returned.
      edge_output_size: Size of the output edge representations for
         each edge type. For edge types not specified here, the latent edge
         representation from the output of the processor will be returned.
      include_sent_messages_in_node_update: Whether to include pooled sent
          messages from each node in the node update.
      use_layer_norm: Whether it uses layer norm or not.
      use_norm_conditioning: If True, the latent feaures outputted by the
        activation normalization that follows the MLPs (e.g. LayerNorm), rather
        than being scaled/offset by learned  parameters of the normalization
        module, will be scaled/offset by offsets/biases produced by a linear
        layer (with different weights for each MLP), which takes an extra
        argument "global_norm_conditioning". This argument is used to condition
        the normalization of all nodes and all edges (hence global), and would
        usually only have a batch and feature axis. This is typically used to
        condition diffusion models on the "diffusion time". Will raise an error
        if this is set to True but the "global_norm_conditioning" is not passed
        to the __call__ method, as well as if this is set to False, but
        "global_norm_conditioning" is passed to the call method.
      activation: name of activation function.
      f32_aggregation: Use float32 in the edge aggregation.
      aggregate_edges_for_nodes_fn: function used to aggregate messages to each
        node.
      aggregate_normalization: An optional constant that normalizes the output
        of aggregate_edges_for_nodes_fn. For context, this can be used to
        reduce the shock the model undergoes when switching resolution, which
        increase the number of edges connected to a node. In particular, this is
        useful when using segment_sum, but should not be combined with
        segment_mean.
      name: Name of the model.
    """
    super().__init__(name=name)

    # Store all parameters except graph_template
    self._node_latent_size = node_latent_size
    self._edge_latent_size = edge_latent_size
    self._mlp_hidden_size = mlp_hidden_size
    self._mlp_num_hidden_layers = mlp_num_hidden_layers
    self._num_message_passing_steps = num_message_passing_steps
    self._num_processor_repetitions = num_processor_repetitions
    self._embed_nodes = embed_nodes
    self._embed_edges = embed_edges
    self._node_output_size = node_output_size
    self._edge_output_size = edge_output_size
    self._include_sent_messages_in_node_update = (
        include_sent_messages_in_node_update)
    if use_norm_conditioning and not use_layer_norm:
      raise ValueError(
          "`norm_conditioning` can only be used when "
          "`use_layer_norm` is true."
      )
    self._use_layer_norm = use_layer_norm
    self._use_norm_conditioning = use_norm_conditioning
    self._activation = _get_activation_fn(activation)
    self._f32_aggregation = f32_aggregation
    self._aggregate_edges_for_nodes_fn = _get_aggregate_edges_for_nodes_fn(
        aggregate_edges_for_nodes_fn)
    self._aggregate_normalization = aggregate_normalization
    self._rngs = rngs # Store the RNGs for lazy initialization

    if aggregate_normalization:
      assert aggregate_edges_for_nodes_fn == "segment_sum"

    # Initialize modules to None; they'll be built on the first __call__
    self.embedder_network: Optional[typed_graph_net.GraphMapFeatures] = None
    self.processor_networks: Optional[List[typed_graph_net.InteractionNetwork]] = None
    self.decoder_network: Optional[typed_graph_net.GraphMapFeatures] = None

  def _initialize_networks(self, graph_template: typed_graph.TypedGraph):
    """Initializes all sub-networks using the provided graph_template."""
    # The embedder graph network independently embeds edge and node features.
    embed_edge_fn = None
    if self._embed_edges:
      embed_edge_fn = _build_update_fns_for_edge_types(
          builder_fn=mlp_builder.MLPWithNormConditioning,
          graph_template=graph_template,
          mlp_hidden_size=self._mlp_hidden_size,
          mlp_num_hidden_layers=self._mlp_num_hidden_layers,
          activation=self._activation,
          use_layer_norm=self._use_layer_norm,
          use_norm_conditioning=self._use_norm_conditioning,
          rngs=self._rngs,
          input_sizes=None, # Inferred from graph_template
          output_sizes=self._edge_latent_size)

    embed_node_fn = None
    if self._embed_nodes:
      embed_node_fn = _build_update_fns_for_node_types(
          builder_fn=mlp_builder.MLPWithNormConditioning,
          graph_template=graph_template,
          mlp_hidden_size=self._mlp_hidden_size,
          mlp_num_hidden_layers=self._mlp_num_hidden_layers,
          activation=self._activation,
          use_layer_norm=self._use_layer_norm,
          use_norm_conditioning=self._use_norm_conditioning,
          rngs=self._rngs,
          input_sizes=None, # Inferred from graph_template
          output_sizes=self._node_latent_size)

    self.embedder_network = typed_graph_net.GraphMapFeatures(
        embed_edge_fn=embed_edge_fn,
        embed_node_fn=embed_node_fn,
    )

    if self._f32_aggregation:
      def aggregate_fn(data, *args, **kwargs):
        dtype = data.dtype
        data = data.astype(jnp.float32)
        output = self._aggregate_edges_for_nodes_fn(data, *args, **kwargs)
        if self._aggregate_normalization:
          output = output / self._aggregate_normalization
        output = output.astype(dtype)
        return output
    else:
      def aggregate_fn(data, *args, **kwargs):
        output = self._aggregate_edges_for_nodes_fn(data, *args, **kwargs)
        if self._aggregate_normalization:
          output = output / self._aggregate_normalization
        return output

    # Create `num_message_passing_steps` graph networks with unshared parameters
    # that update the node and edge latent features.
    self.processor_networks = []
    for _ in range(self._num_message_passing_steps):
      self.processor_networks.append(
          typed_graph_net.InteractionNetwork(
              update_edge_fn=_build_update_fns_for_edge_types(
                  builder_fn=mlp_builder.MLPWithNormConditioning,
                  graph_template=graph_template,
                  mlp_hidden_size=self._mlp_hidden_size,
                  mlp_num_hidden_layers=self._mlp_num_hidden_layers,
                  activation=self._activation,
                  use_layer_norm=self._use_layer_norm,
                  use_norm_conditioning=self._use_norm_conditioning,
                  rngs=self._rngs,
                  input_sizes=self._edge_latent_size,
                  output_sizes=self._edge_latent_size),
              update_node_fn=_build_update_fns_for_node_types(
                  builder_fn=mlp_builder.MLPWithNormConditioning,
                  graph_template=graph_template,
                  mlp_hidden_size=self._mlp_hidden_size,
                  mlp_num_hidden_layers=self._mlp_num_hidden_layers,
                  activation=self._activation,
                  use_layer_norm=self._use_layer_norm,
                  use_norm_conditioning=self._use_norm_conditioning,
                  rngs=self._rngs, 
                  input_sizes=self._node_latent_size,
                  output_sizes=self._node_latent_size),
              aggregate_edges_for_nodes_fn=aggregate_fn,
              include_sent_messages_in_node_update=(
                  self._include_sent_messages_in_node_update),
              )
      )

    # The output MLPs converts edge/node latent features into the output sizes.
    output_edge_fn = None
    if self._edge_output_size:
        output_edge_fn = _build_update_fns_for_edge_types(
            builder_fn=mlp_builder.MLPWithNormConditioning,
            graph_template=graph_template,
            mlp_hidden_size=self._mlp_hidden_size,
            mlp_num_hidden_layers=self._mlp_num_hidden_layers,
            activation=self._activation,
            use_layer_norm=False, # Output MLPs usually don't have layer norm
            use_norm_conditioning=False, # Output MLPs usually don't have conditioning
            rngs=self._rngs, 
            input_sizes=self._edge_latent_size,
            output_sizes=self._edge_output_size
        )
    output_node_fn = None
    if self._node_output_size:
        output_node_fn = _build_update_fns_for_node_types(
            builder_fn=mlp_builder.MLPWithNormConditioning,
            graph_template=graph_template,
            mlp_hidden_size=self._mlp_hidden_size,
            mlp_num_hidden_layers=self._mlp_num_hidden_layers,
            activation=self._activation,
            use_layer_norm=False, # Output MLPs usually don't have layer norm
            use_norm_conditioning=False, # Output MLPs usually don't have conditioning
            rngs=self._rngs, # Use derived RNGs
            input_sizes=self._node_latent_size,
            output_sizes=self._node_output_size
        )

    self.decoder_network = typed_graph_net.GraphMapFeatures(
        embed_edge_fn=output_edge_fn,
        embed_node_fn=output_node_fn,
    )


  def __call__(self,
               input_graph: typed_graph.TypedGraph,
               global_norm_conditioning: Optional[chex.Array] = None
               ) -> typed_graph.TypedGraph:
    """Forward pass of the learnable dynamics model."""

    # Lazily initialize networks on the first call
    if self.embedder_network is None: # Check any of the networks
      self._initialize_networks(input_graph)


    # Embed input features (if applicable).
    latent_graph_0 = self._embed(input_graph, self.embedder_network, global_norm_conditioning)

    # Do `m` message passing steps in the latent graphs.
    latent_graph_m = self._process(latent_graph_0, self.processor_networks, global_norm_conditioning)

    # Compute outputs from the last latent graph (if applicable).
    return self._output(latent_graph_m, self.decoder_network)

  def _embed(
      self,
      input_graph: typed_graph.TypedGraph,
      embedder_network: typed_graph_net.GraphMapFeatures, # Type hint for clarity
      global_norm_conditioning: Optional[chex.Array] = None
  ) -> typed_graph.TypedGraph:
    """Embeds the input graph features into a latent graph."""

    # Copy the context to all of the node types, if applicable.
    context_features = input_graph.context.features
    if jax.tree_util.tree_leaves(context_features):
      # This code assumes a single input feature array for the context and for
      # each node type.
      assert len(jax.tree_util.tree_leaves(context_features)) == 1
      new_nodes = {}
      for node_set_name, node_set in input_graph.nodes.items():
        node_features = node_set.features
        broadcasted_context = jnp.repeat(
            context_features, node_set.n_node, axis=0,
            total_repeat_length=node_features.shape[0])
        new_nodes[node_set_name] = node_set._replace(
            features=jnp.concatenate(
                [node_features, broadcasted_context], axis=-1))
      input_graph = input_graph._replace(
          nodes=new_nodes,
          context=input_graph.context._replace(features=()))

    # Embeds the node and edge features.
    latent_graph_0 = embedder_network(input_graph,
                                      global_norm_conditioning)
    return latent_graph_0

  def _process(
      self,
      latent_graph_0: typed_graph.TypedGraph,
      processor_networks: List[typed_graph_net.InteractionNetwork], # Type hint for clarity
      global_norm_conditioning: Optional[chex.Array] = None
  ) -> typed_graph.TypedGraph:
    """Processes the latent graph with several steps of message passing."""

    latent_graph = latent_graph_0
    for unused_repetition_i in range(self._num_processor_repetitions):
      for processor_network in processor_networks:
        latent_graph = self._process_step(processor_network, latent_graph, global_norm_conditioning)

    return latent_graph

  def _process_step(
      self, processor_network_k: typed_graph_net.InteractionNetwork, # Type hint for clarity
      latent_graph_prev_k: typed_graph.TypedGraph,
      global_norm_conditioning: Optional[chex.Array] = None
  ) -> typed_graph.TypedGraph:
    """Single step of message passing with node/edge residual connections."""

    latent_graph_k = processor_network_k(latent_graph_prev_k, global_norm_conditioning)

    nodes_with_residuals = {}
    for k, prev_set in latent_graph_prev_k.nodes.items():
      nodes_with_residuals[k] = prev_set._replace(
          features=prev_set.features + latent_graph_k.nodes[k].features)

    edges_with_residuals = {}
    for k, prev_set in latent_graph_prev_k.edges.items():
      edges_with_residuals[k] = prev_set._replace(
          features=prev_set.features + latent_graph_k.edges[k].features)

    latent_graph_k = latent_graph_k._replace(
        nodes=nodes_with_residuals, edges=edges_with_residuals)
    return latent_graph_k

  def _output(
      self,
      latent_graph: typed_graph.TypedGraph,
      output_network: typed_graph_net.GraphMapFeatures, # Type hint for clarity
  ) -> typed_graph.TypedGraph:
    """Produces the output from the latent graph."""
    return output_network(latent_graph)