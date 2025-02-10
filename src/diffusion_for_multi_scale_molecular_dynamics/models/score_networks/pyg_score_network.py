from dataclasses import dataclass
from typing import AnyStr, Dict

import torch
from torch import nn
from torch_geometric.nn.models import GCN, GIN, GAT

from diffusion_for_multi_scale_molecular_dynamics.models.egnn_utils import get_edges_with_radial_cutoff
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network import (
    ScoreNetwork, ScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, NOISE, NOISY_AXL_COMPOSITION, TIME)
from diffusion_for_multi_scale_molecular_dynamics.utils.d3pm_utils import \
    class_index_to_onehot


@dataclass(kw_only=True)
class PygScoreNetworkParameters(ScoreNetworkParameters):
    """Specific Hyper-parameters for MLP score networks."""

    architecture: str = "pyg_gcn"  # or pyg_gin, pyg_gat
    number_of_atoms: int  # the number of atoms in a configuration.
    hidden_channels: int  # size of each hidden same.
    num_layers: int  # Number of message passing layers.
    dropout: float = 0.0  # dropout probability
    act: str = "relu"  # The non-linear activation function to use.
    v2: bool = False  # for GAT only
    # the dimension of the embedding of the noise parameter.
    noise_embedding_dimensions_size: int

    # the dimension of the embedding of the relative coordinates
    relative_coordinates_embedding_dimensions_size: int

    # the dimension of the embedding of the time parameter.
    time_embedding_dimensions_size: int

    # the dimension of the embedding of the atom types
    atom_type_embedding_dimensions_size: int

    # dimension of the conditional variable embedding
    condition_embedding_size: int = 64

    radial_cutoff: float = 5.0


class PygScoreNetwork(ScoreNetwork):
    """Simple Model Class.

    Inherits from the given framework's model class. This is a simple MLP model.
    """

    def __init__(self, hyper_params: PygScoreNetworkParameters):
        """__init__.

        Args:
            hyper_params : hyper parameters from the config file.
        """
        super(PygScoreNetwork, self).__init__(hyper_params)

        self._natoms = hyper_params.number_of_atoms
        self.num_atom_types = hyper_params.num_atom_types
        self.num_classes = self.num_atom_types + 1  # add 1 for the MASK class
        self.radial_cutoff = hyper_params.radial_cutoff

        # Each relative coordinate will be embedded on the unit circle with (cos, sin), leading to
        # a doubling of the number of dimensions.
        flat_relative_coordinates_input_dimension = (
            2 * self.spatial_dimension
        )

        coordinate_output_dimension = self.spatial_dimension

        input_dimension = (
            hyper_params.relative_coordinates_embedding_dimensions_size
            + hyper_params.noise_embedding_dimensions_size
            + hyper_params.time_embedding_dimensions_size
            + hyper_params.atom_type_embedding_dimensions_size
        )

        self.relative_coordinates_embedding_layer = nn.Linear(
            flat_relative_coordinates_input_dimension,
            hyper_params.relative_coordinates_embedding_dimensions_size,
        )

        self.noise_embedding_layer = nn.Linear(
            1, hyper_params.noise_embedding_dimensions_size
        )

        self.time_embedding_layer = nn.Linear(
            1, hyper_params.time_embedding_dimensions_size
        )

        self.atom_type_embedding_layer = nn.Linear(
            self.num_classes, hyper_params.atom_type_embedding_dimensions_size
        )

        self.condition_embedding_layer = nn.Linear(
            coordinate_output_dimension, hyper_params.condition_embedding_size
        )

        match hyper_params.architecture:
            case "gcn" | "gin":
                gnn_model = GCN if hyper_params.architecture == "pyg_gcn" else GIN
                self.gnn_model = gnn_model(
                    in_channels=input_dimension,
                    hidden_channels=hyper_params.hidden_channels,
                    num_layers=hyper_params.num_layers,
                    out_channels=self.spatial_dimension
                )
            case "gat":
                self.gnn_model = GAT(
                    in_channels=input_dimension,
                    hidden_channels=hyper_params.hidden_channels,
                    num_layers=hyper_params.num_layers,
                    out_channels=self.spatial_dimension,
                    v2=hyper_params.v2
                )

            case _:
                raise ValueError(
                    f"Pytorch geometric model should be gcn, gin or gat. Got {hyper_params.architecture}."
                )

        # Create a self nn object to be discoverable to be placed on the correct device
        self.output_A_layer = nn.Linear(
            self.spatial_dimension, self.num_classes,
        )  # convert the X score to something matching A diffusion. This is experimental and meant for 1 atom type only.

        self.output_X_layer = nn.Identity()  # take the output of GCN directly
        self.output_L_layer = nn.Identity()
        self.output_layers = AXL(
            A=self.output_A_layer, X=self.output_X_layer, L=self.output_L_layer
        )

    def _check_batch(self, batch: Dict[AnyStr, torch.Tensor]):
        super(PygScoreNetwork, self)._check_batch(batch)
        number_of_atoms = batch[NOISY_AXL_COMPOSITION].X.shape[1]
        assert (
            number_of_atoms == self._natoms
        ), "The dimension corresponding to the number of atoms is not consistent with the configuration."

    def _forward_unchecked(
        self, batch: Dict[AnyStr, torch.Tensor], conditional: bool = False
    ) -> AXL:
        """Forward unchecked.

        Args:
            batch : dictionary containing the data to be processed by the model.
            conditional (optional): if True, do a forward as though the model was conditional on the forces.
                Defaults to False.

        Returns:
            computed_scores : the scores computed by the model in an AXL namedtuple.
        """
        # shape [batch_size, number_of_atoms, spatial_dimension]
        relative_coordinates = batch[NOISY_AXL_COMPOSITION].X

        # Embed each coordinate on the unit circle to ensure periodicity
        angles = 2.0 * torch.pi * relative_coordinates

        # flatten the dimensions associated with sin/cos, natoms and spatial dimension.
        relative_coordinates_input = torch.concat([angles.cos(), angles.sin()], dim=-1)

        relative_coordinates_embedding = self.relative_coordinates_embedding_layer(
            relative_coordinates_input
        )

        sigmas = batch[NOISE].to(relative_coordinates.device)  # shape [batch_size, 1]
        noise_embedding = self.noise_embedding_layer(
            sigmas
        )  # shape [batch_size, noise_embedding_dimension]

        times = batch[TIME].to(relative_coordinates.device)  # shape [batch_size, 1]
        time_embedding = self.time_embedding_layer(
            times
        )  # shape [batch_size, time_embedding_dimension]

        atom_types = batch[NOISY_AXL_COMPOSITION].A
        atom_types_one_hot = class_index_to_onehot(
            atom_types, num_classes=self.num_classes
        )
        atom_type_embedding = self.atom_type_embedding_layer(
            atom_types_one_hot
        )  # shape [batch_size, atom_type_embedding_dimension]

        input = torch.cat(
            [
                relative_coordinates_embedding,
                noise_embedding.unsqueeze(1).repeat(1, self._natoms, 1),
                time_embedding.unsqueeze(1).repeat(1, self._natoms, 1),
                atom_type_embedding,
            ],
            dim=-1,
        ).flatten(start_dim=0, end_dim=1)  # (batch * natoms, nfeatures)

        # get the edges
        edges = get_edges_with_radial_cutoff(
            relative_coordinates,
            batch["unit_cell"],
            radial_cutoff=self.radial_cutoff,
            spatial_dimension=relative_coordinates.shape[-1]
        )
        edges_val = torch.ones(edges.shape[0]).to(input)
        edges = torch.sparse_coo_tensor(
            edges.transpose(0, 1),
            edges_val,
            (input.shape[0], input.shape[0])
        )

        output = self.gnn_model(input, edge_index=edges)

        coordinates_output = output.reshape(
            relative_coordinates.shape
        )

        atom_types_output = self.output_layers.A(output).reshape(
            atom_types_one_hot.shape
        )
        lattice_output = torch.zeros_like(coordinates_output)  # TODO placeholder

        axl_output = AXL(A=atom_types_output, X=coordinates_output, L=lattice_output)
        return axl_output
