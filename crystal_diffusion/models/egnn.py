"""Equivariant Graph Neural Network (EGNN).

This implementation is based on the following link:
https://github.com/vgsatorras/egnn/blob/3c079e7267dad0aa6443813ac1a12425c3717558/models/egnn_clean/egnn_clean.py#L106

It implements EGNN as described in the paper "E(n) Equivariant Graph Neural Networks".

The file is modified from the original download to fit our own linting style and add additional controls.
"""
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from crystal_diffusion.models.egnn_utils import (unsorted_segment_mean,
                                                 unsorted_segment_sum)


class E_GCL(nn.Module):
    """E(n) Equivariant Convolutional Layer."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        message_n_hidden_dimensions: int,
        message_hidden_dimensions_size: int,
        node_n_hidden_dimensions: int,
        node_hidden_dimensions_size: int,
        coordinate_n_hidden_dimensions: int,
        coordinate_hidden_dimensions_size: int,
        act_fn: Callable = nn.SiLU(),
        residual: bool = True,
        attention: bool = False,
        normalize: bool = False,
        coords_agg: str = "mean",
        message_agg: str = "mean",
        tanh: bool = False,
    ):
        """E_GCL layer initialization.

        Args:
            input_size: number of node features in the input
            output_size: number of node features in the output
            message_n_hidden_dimensions: number of hidden layers of the message (edge) MLP
            message_hidden_dimensions_size: size of the hidden layers of the message (edge) MLP
            node_n_hidden_dimensions: number of hidden layers of the node update MLP
            node_hidden_dimensions_size: size of the hidden layers of the node update MLP
            coordinate_n_hidden_dimensions: number of hidden layers of the coordinate update MLP
            coordinate_hidden_dimensions_size: size of the hidden layers of the coordinate update MLP
            act_fn: activation function used in the MLPs. Defaults to nn.SiLU()
            residual: if True, add a skip connection in the nodes update. Defaults to True.
            attention: if True, multiply the message output by a gated value of the output. Defaults to False.
            normalize: if True, use a normalized version of the coordinates update i.e. x_i^l - x_j^l would be a unit
                vector in eq. 4 in https://arxiv.org/pdf/2102.09844. Defaults to False.
            coords_agg: Use a mean or sum aggregation for the coordinates update. Defaults to mean.
            message_agg: Use a mean or sum aggregation for the messages. Defaults to mean.
            tanh: if True, add a tanh non-linearity after the coordinates update. Defaults to False.
        """
        super(E_GCL, self).__init__()
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.tanh = tanh
        self.epsilon = 1e-8


        if coords_agg not in ["mean", "sum"]:
            raise ValueError(f"coords_agg should be mean or sum. Got {coords_agg}")
        self.coords_agg_fn = unsorted_segment_sum if coords_agg == "sum" else unsorted_segment_mean
        self.msg_agg_fn = unsorted_segment_sum if message_agg == "sum" else unsorted_segment_mean

        # message update MLP i.e. message m_{ij} used in the graph neural network.
        # \phi_e is eq. (3) in https://arxiv.org/pdf/2102.09844
        # Input is a concatenation of the two node features and distance
        message_input_size = input_size * 2 + 1
        self.message_mlp = nn.Sequential(
            nn.Linear(message_input_size, message_hidden_dimensions_size),
            act_fn
        )
        for _ in range(message_n_hidden_dimensions):
            self.message_mlp.append(nn.Linear(message_hidden_dimensions_size, message_hidden_dimensions_size))
            self.message_mlp.append(act_fn)

        # node update mlp. Input is the node feature (size input_size) and the aggregated messages from neighbors
        # size (message_hidden_dimension)
        # \phi_h in eq. (6) in https://arxiv.org/pdf/2102.09844
        node_input_size = input_size + message_hidden_dimensions_size
        self.node_mlp = nn.Sequential(
            nn.Linear(node_input_size, node_hidden_dimensions_size),
            act_fn
        )
        for _ in range(node_n_hidden_dimensions):
            self.node_mlp.append(nn.Linear(node_hidden_dimensions_size, node_hidden_dimensions_size))
            self.node_mlp.append(act_fn)
        self.node_mlp.append(nn.Linear(node_hidden_dimensions_size, output_size))

        # coordinate (x) update MLP. Input is the message m_{ij}
        # \phi_x in eq.(4) in https://arxiv.org/pdf/2102.09844

        coordinate_input_size = message_hidden_dimensions_size
        self.coord_mlp = nn.Sequential(nn.Linear(coordinate_input_size, coordinate_hidden_dimensions_size))
        self.coord_mlp.append(act_fn)
        for _ in range(coordinate_n_hidden_dimensions):
            self.coord_mlp.append(nn.Linear(coordinate_hidden_dimensions_size, coordinate_hidden_dimensions_size))
            self.coord_mlp.append(act_fn)
        final_coordinate_layer = nn.Linear(coordinate_hidden_dimensions_size, 1, bias=False)
        # based on the original implementation - multiply the random initialization by 0.001 (default is 1)
        # torch.nn.init.xavier_uniform_(final_coordinate_layer.weight, gain=0.001)
        self.coord_mlp.append(final_coordinate_layer)  # initialized with a different
        if self.tanh:  # optional, add a tanh to saturate the messages
            self.coord_mlp.append(nn.Tanh())

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(message_hidden_dimensions_size, 1), nn.Sigmoid())

    def message_model(self, source: torch.Tensor, target: torch.Tensor, radial: torch.Tensor) -> torch.Tensor:
        r"""Constructs the message m_{ij} from source (j) to target (i).

        .. math::

            m_ij = \phi_e(h_i^l, h_j^l, ||x_i^l - x_j^l||^2, a_{ij)}

        with :math:`a_{ij}` the edge attributes

        Args:
            source: source node features (size: number of edges, input_size)
            target: target node features (size: number of edges, input_size)
            radial: distance squared between nodes i and j (size: number of edges, 1)

        Returns:
            messages :math:`m_{ij}` size: number of edges, message_hidden_dimensions_size
        """
        out = torch.cat([source, target, radial], dim=1)
        out = self.message_mlp(out)
        if self.attention:  # gate the message by itself - optional
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x: torch.Tensor, edge_index: torch.Tensor, messages: torch.Tensor
                   ) -> torch.Tensor:
        r"""Update the node features.

        .. math::

            h_i^{(l+1)} = \phi_h (h_i^l, m_i)

        Args:
            x: node features. Size number of nodes, input_size
            edge_index: source and target indices defining the edges. size: number of edges, 2
            messages: messages between nodes. size: number of edges, message_hidden_dimension_size

        Returns:
            updated node features. size: number of nodes, output_size
        """
        row = edge_index[:, 0]
        agg = self.msg_agg_fn(messages, row, num_segments=x.size(0))  # sum messages m_i = \sum_j m_{ij}
        agg = torch.cat([x, agg], dim=1)  # concat h_i and m_i
        out = self.node_mlp(agg)
        if self.residual:  # optional skip connection
            out = x + out
        return out

    def coord_model(self, coord: torch.Tensor, edge_index: torch.Tensor, coord_diff: torch.Tensor,
                    messages: torch.Tensor) -> torch.Tensor:
        r"""Update the coordinates.

        .. math::

            x_i^{(l+1)} = x_i^l + C \sum_{i \neq j} (x_i - x_j) \phi_x(m_{ij})

        Args:
            coord: coordinates. size: number of nodes, spatial dimension
            edge_index: edge indices. size: number of edges, 2
            coord_diff: difference between coordinates, :math:`x_i - x_j`. size: number of edges, spatial dimension
            messages: messages between nodes i and j.  size: number of edges, message_hidden_dimensions_size

        Returns:
            updates coordinates. size: number of nodes, spatial dimension
        """
        row = edge_index[:, 0]
        trans = coord_diff * self.coord_mlp(messages)  # (x_i  - x_j) *  \phi_m(m_{ij})
        agg = self.coords_agg_fn(trans, row, num_segments=coord.size(0))  # sum over j
        coord += agg
        return coord

    def coord2radial(self, edge_index: torch.Tensor, coord: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute distances between linked nodes.

        Args:
            edge_index: source and destination indices defining the edges. Size: number of edges, 2
            coord: coordinates. size: number of nodes, spatial dimension

        Returns:
            distance squared between nodes. size: number of edges
            distance vector between nodes. size: number of edges, spatial dimension
        """
        row, col = edge_index[:, 0], edge_index[:, 1]
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:  # normalize distance vector to be unit vector
            # norm is detached from gradient in the original implementation - not clear why
            # norm = torch.sqrt(radial).detach() + self.epsilon
            norm = torch.sqrt(radial) + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, coord: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute node embeddings and coordinates.

        Args:
            h: node features. size: number of nodes, input_size
            edge_index: indices defining the edges. size: number of edges, 2
            coord: node positions. size: number of nodes, spatial dimension

        Returns:
            updated node features. size: number of nodes, output_size
            updated coordinates. size: number of nodes, spatial dimension
        """
        row, col = edge_index[:, 0], edge_index[:, 1]
        # compute distances between nodes (atoms)
        radial, coord_diff = self.coord2radial(edge_index, coord)

        messages = self.message_model(h[row], h[col], radial)  # compute m_{ij}
        coord = self.coord_model(coord, edge_index, coord_diff, messages)  # update x_i
        h = self.node_model(h, edge_index, messages)  # update h_i

        return h, coord


class EGNN(nn.Module):
    """EGNN model."""
    def __init__(
        self,
        input_size: int,
        message_n_hidden_dimensions: int,
        message_hidden_dimensions_size: int,
        node_n_hidden_dimensions: int,
        node_hidden_dimensions_size: int,
        coordinate_n_hidden_dimensions: int,
        coordinate_hidden_dimensions_size: int,
        act_fn: Callable = nn.SiLU(),
        residual: bool = True,
        attention: bool = False,
        normalize: bool = False,
        tanh: bool = False,
        coords_agg: str = "mean",
        message_agg: str = "mean",
        n_layers: int = 4,
    ):
        """EGNN model stacking multiple E_GCL layers.

        Args:
            input_size: number of node features in the input
            message_n_hidden_dimensions: number of hidden layers of the message (edge) MLP
            message_hidden_dimensions_size: size of the hidden layers of the message (edge) MLP
            node_n_hidden_dimensions: number of hidden layers of the node update MLP
            node_hidden_dimensions_size: size of the hidden layers of the node update MLP
            coordinate_n_hidden_dimensions: number of hidden layers of the coordinate update MLP
            coordinate_hidden_dimensions_size: size of the hidden layers of the coordinate update MLP
            act_fn: activation function used in the MLPs. Defaults to nn.SiLU()
            residual: if True, add a skip connection in the nodes update. Defaults to True.
            attention: if True, multiply the message output by a gated value of the output. Defaults to False.
            normalize: if True, use a normalized version of the coordinates update i.e. x_i^l - x_j^l would be a unit
                vector in eq. 4 in https://arxiv.org/pdf/2102.09844. Defaults to False.
            coords_agg: Use a mean or sum aggregation for the coordinates update. Defaults to mean.
            message_agg: Use a mean or sum aggregation for the messages. Defaults to mean.
            tanh: if True, add a tanh non-linearity after the coordinates update. Defaults to False.
            n_layers: number of E_GCL layers. Defaults to 4.
        """
        super(EGNN, self).__init__()
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(input_size, node_hidden_dimensions_size)
        self.graph_layers = nn.ModuleList([])
        for _ in range(0, n_layers):
            self.graph_layers.append(
                E_GCL(
                    input_size=node_hidden_dimensions_size,
                    output_size=node_hidden_dimensions_size,
                    message_n_hidden_dimensions=message_n_hidden_dimensions,
                    message_hidden_dimensions_size=message_hidden_dimensions_size,
                    node_n_hidden_dimensions=node_n_hidden_dimensions,
                    node_hidden_dimensions_size=node_hidden_dimensions_size,
                    coordinate_n_hidden_dimensions=coordinate_n_hidden_dimensions,
                    coordinate_hidden_dimensions_size=coordinate_hidden_dimensions_size,
                    act_fn=act_fn,
                    residual=residual,
                    attention=attention,
                    normalize=normalize,
                    coords_agg=coords_agg,
                    message_agg=message_agg,
                    tanh=tanh,
                )
            )

    def forward(self, h: torch.Tensor, edges: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward instructions for the model.

        Args:
            h: node features. size is number of nodes (atoms), input size
            edges: source and destination node indices defining the graph edges. size is number of edges, 2
            x: node coordinates. size is number of nodes, spatial dimension

        Returns:
            estimated score. size is number of nodes, spatial dimension
        """
        h = self.embedding_in(h)
        for graph_layer in self.graph_layers:
            h, x = graph_layer(h, edges, x)
        return x
