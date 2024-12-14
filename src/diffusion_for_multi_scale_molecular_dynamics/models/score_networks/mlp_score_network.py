from dataclasses import dataclass
from typing import AnyStr, Dict, List

import einops
import torch
from torch import nn

from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network import (
    ScoreNetwork, ScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, CARTESIAN_FORCES, NOISE, NOISY_AXL_COMPOSITION, TIME)
from diffusion_for_multi_scale_molecular_dynamics.utils.d3pm_utils import \
    class_index_to_onehot


@dataclass(kw_only=True)
class MLPScoreNetworkParameters(ScoreNetworkParameters):
    """Specific Hyper-parameters for MLP score networks."""

    architecture: str = "mlp"
    number_of_atoms: int  # the number of atoms in a configuration.
    n_hidden_dimensions: int  # the number of hidden layers.
    hidden_dimensions_size: int  # the dimensions of the hidden layers.
    noise_embedding_dimensions_size: (
        int  # the dimension of the embedding of the noise parameter.
    )
    time_embedding_dimensions_size: (
        int  # the dimension of the embedding of the time parameter.
    )
    atom_type_embedding_dimensions_size: (
        int  # the dimension of the embedding of the atom types
    )
    condition_embedding_size: int = (
        64  # dimension of the conditional variable embedding
    )

    # Should the output normalized score be mutliplied by a learned prefactor that is
    # independent of coordinates?
    use_prefactor_in_score: bool = False


class MLPScoreNetwork(ScoreNetwork):
    """Simple Model Class.

    Inherits from the given framework's model class. This is a simple MLP model.
    """

    def __init__(self, hyper_params: MLPScoreNetworkParameters):
        """__init__.

        Args:
            hyper_params : hyper parameters from the config file.
        """
        super(MLPScoreNetwork, self).__init__(hyper_params)

        self.use_prefactor_in_score = hyper_params.use_prefactor_in_score

        self.natoms = hyper_params.number_of_atoms
        self.num_atom_types = hyper_params.num_atom_types
        self.num_classes = self.num_atom_types + 1  # add 1 for the MASK class

        self.coordinate_output_dimension = self.spatial_dimension * self.natoms
        self.coordinate_embedding_dimension = 2 * self.spatial_dimension * self.natoms
        self.atom_type_output_dimension = self.natoms * self.num_classes
        self.noise_embedding_dimensions_size = hyper_params.noise_embedding_dimensions_size
        self.time_embedding_dimensions_size = hyper_params.time_embedding_dimensions_size
        self.atom_type_embedding_dimensions_size = hyper_params.atom_type_embedding_dimensions_size

        self.flatten = nn.Flatten()

        input_dimension = self._get_main_input_dimension()

        hidden_dimensions = [hyper_params.hidden_dimensions_size] * (
            hyper_params.n_hidden_dimensions
        )

        self.noise_embedding_layer = nn.Linear(
            1, self.noise_embedding_dimensions_size
        )

        self.time_embedding_layer = nn.Linear(
            1, self.time_embedding_dimensions_size
        )

        self.atom_type_embedding_layer = nn.Linear(
            self.num_classes, hyper_params.atom_type_embedding_dimensions_size
        )

        self.condition_embedding_layer = nn.Linear(
            self.coordinate_output_dimension, hyper_params.condition_embedding_size
        )

        main_layer_dimensions = [input_dimension] + hidden_dimensions
        self.conditioning_mlp = ConditioningMLP(main_layer_dimensions=main_layer_dimensions,
                                                condition_embedding_dimension=hyper_params.condition_embedding_size,
                                                non_linearity=nn.ReLU())

        if self.use_prefactor_in_score:
            prefactor_input_dimension = self._get_prefactor_input_dimension()
            prefactor_layer_dimensions = [prefactor_input_dimension] + hidden_dimensions + [1]
            self.prefactor_mlp = SimplMLP(layer_dimensions=prefactor_layer_dimensions,
                                          non_linearity=nn.ReLU())

        # Create a self nn object to be discoverable to be placed on the correct device
        self.output_A_layer = nn.Linear(hyper_params.hidden_dimensions_size, self.atom_type_output_dimension)
        self.output_X_layer = nn.Linear(hyper_params.hidden_dimensions_size, self.coordinate_output_dimension)
        self.output_L_layer = nn.Identity()
        self.output_layers = AXL(A=self.output_A_layer,
                                 X=self.output_X_layer,
                                 L=self.output_L_layer)  # TODO placeholder

    def _check_batch(self, batch: Dict[AnyStr, torch.Tensor]):
        super(MLPScoreNetwork, self)._check_batch(batch)
        number_of_atoms = batch[NOISY_AXL_COMPOSITION].X.shape[1]
        assert (
            number_of_atoms == self.natoms
        ), "The dimension corresponding to the number of atoms is not consistent with the configuration."

    def _forward_unchecked(
        self, batch: Dict[AnyStr, torch.Tensor], conditional: bool = False
    ) -> AXL:
        """Forward unchecked.

        This method assumes that the input data has already been checked with respect to expectations
        and computes the scores assuming that the data is in the correct format.

        Args:
            batch : dictionary containing the data to be processed by the model.
            conditional (optional): if True, do a forward as though the model was conditional on the forces.
                Defaults to False.

        Returns:
            computed_scores : the scores computed by the model in an AXL namedtuple.
        """
        main_input = self._get_main_input(batch)
        conditioning_input = self._get_conditioning_input(batch)

        latent_representation = self.conditioning_mlp(main_input=main_input,
                                                      conditioning_input=conditioning_input,
                                                      conditional=conditional)

        coordinates_output = self.output_layers.X(latent_representation).reshape(-1,
                                                                                 self.natoms,
                                                                                 self.spatial_dimension)

        if self.use_prefactor_in_score:
            prefactor_input = self._get_prefactor_input(batch)
            prefactor_output = self.prefactor_mlp.forward(prefactor_input).reshape(-1, 1, 1)
            coordinates_output *= prefactor_output

        atom_types_output = self.output_layers.A(latent_representation).reshape(-1,
                                                                                self.natoms,
                                                                                self.num_classes)

        lattice_output = torch.zeros_like(atom_types_output)  # TODO placeholder

        axl_output = AXL(A=atom_types_output, X=coordinates_output, L=lattice_output)
        return axl_output

    def _get_main_input_dimension(self):
        main_input_dimension = (
            self.coordinate_embedding_dimension
            + self.natoms * self.atom_type_embedding_dimensions_size
            + self.noise_embedding_dimensions_size
            + self.time_embedding_dimensions_size
        )
        return main_input_dimension

    def _get_main_input(self, batch):
        (relative_coordinates_embedding, atom_type_embedding,
         noise_embedding, time_embedding) = self._get_input_embeddings(batch)

        main_input = torch.cat(
            [relative_coordinates_embedding, atom_type_embedding, noise_embedding, time_embedding], dim=1)
        return main_input

    def _get_prefactor_input_dimension(self):
        prefactor_input_dimension = self.noise_embedding_dimensions_size + self.time_embedding_dimensions_size
        return prefactor_input_dimension

    def _get_prefactor_input(self, batch):
        (flat_relative_coordinates, atom_type_embedding,
         noise_embedding, time_embedding) = self._get_input_embeddings(batch)

        prefactor_input = torch.cat([noise_embedding, time_embedding], dim=1)
        return prefactor_input

    def _get_conditioning_input(self, batch):
        conditioning_input = self.condition_embedding_layer(
            self.flatten(batch[CARTESIAN_FORCES])
        )
        return conditioning_input

    @staticmethod
    def _get_relative_coordinates_embedding(
        relative_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """Get relative coordinates embedding.

        Get the positions that take points on the torus into a higher dimensional
        (ie, 2 x spatial_dimension) Euclidean space.

        Args:
            relative_coordinates: relative coordinates, of dimensions [batch , natoms, spatial_dimension]

        Returns:
            embeddings: uplifted relative coordinates to a higher dimensional Euclidean space.
        """
        # Uplift the relative coordinates to the embedding Euclidean space
        angles = 2.0 * torch.pi * relative_coordinates
        cosines = angles.cos()
        sines = angles.sin()
        embeddings = einops.rearrange(
            torch.stack([cosines, sines]),
            "two batch natoms space -> batch (natoms space two)",
        )
        return embeddings

    def _get_input_embeddings(self, batch):
        """Extract relevant information from the batch."""
        # shape [batch_size, number_of_atoms, spatial_dimension]
        relative_coordinates = batch[NOISY_AXL_COMPOSITION].X

        device = relative_coordinates.device

        # shape [batch_size, 2 * number_of_atoms * spatial_dimension]
        relative_coordinates_embedding = self._get_relative_coordinates_embedding(relative_coordinates)

        # shape [batch_size, 1]
        sigmas = batch[NOISE].to(device)

        # shape [batch_size, noise_embedding_dimension]
        noise_embedding = self.noise_embedding_layer(sigmas)

        # shape [batch_size, 1]
        times = batch[TIME].to(device)

        # shape [batch_size, time_embedding_dimension]
        time_embedding = self.time_embedding_layer(times)

        atom_types = batch[NOISY_AXL_COMPOSITION].A
        atom_types_one_hot = class_index_to_onehot(
            atom_types, num_classes=self.num_classes
        )

        # shape [batch_size, atom_type_embedding_dimension]
        atom_type_embedding = self.flatten(self.atom_type_embedding_layer(atom_types_one_hot))

        return relative_coordinates_embedding, atom_type_embedding, noise_embedding, time_embedding


def create_linear_layers(input_dimensions: List[int], output_dimensions: List[int]) -> nn.Module:
    """Create linear layers.

    Creates a list of linear layers.

    Args:
        input_dimensions: list of input dimensions.
        output_dimensions: list of output dimensions.

    Returns:
        layers: list of torch module linear layers
    """
    layers = nn.ModuleList()
    assert len(input_dimensions) == len(output_dimensions), "inconsistent dimensions array"

    for in_d, out_d in zip(input_dimensions, output_dimensions):
        linear_layer = nn.Linear(in_d, out_d)
        layers.append(linear_layer)

    return layers


class ConditioningMLP(torch.nn.Module):
    """A mlp that combines main and conditioning information."""

    def __init__(self, main_layer_dimensions: List[int],
                 condition_embedding_dimension: int,
                 non_linearity: nn.Module):
        """Initialization method.

        Args:
            main_layer_dimensions: list of dimensions. It is assumed to be of the form
                [main_input_dimension, hidden_dimensions1, hidden_dimension2,... , output_dimension].
            condition_embedding_dimension: input dimension of the conditioning embedding.

            non_linearity: non-linearity to apply between linear layers.
        """
        super(ConditioningMLP, self).__init__()

        input_dimensions = main_layer_dimensions[:-1]
        output_dimensions = main_layer_dimensions[1:]

        conditioning_input_dimensions = len(input_dimensions) * [condition_embedding_dimension]

        self.main_linear_layers = create_linear_layers(input_dimensions, output_dimensions)
        self.conditioning_linear_layers = create_linear_layers(conditioning_input_dimensions, output_dimensions)

        self.non_linearity = non_linearity

    def forward(self, main_input: torch.Tensor, conditioning_input: torch.Tensor, conditional: bool) -> torch.Tensor:
        """Forward method.

        Args:
            main_input: a torch tensor of shape [batch_size, main_input_dimension].
            conditioning_input: a torch tensor of shape [batch_size, conditioning_input_dimension].
            conditional: should the conditioning input be used.

        Returns:
            output: a torch tensor of shape [batch_size, output_dimension].
        """
        first_main_layer = self.main_linear_layers[0]
        first_cond_layer = self.conditioning_linear_layers[0]

        output = first_main_layer(main_input)
        if conditional:
            output += first_cond_layer(conditioning_input)

        for main_layer, cond_layer in zip(self.main_linear_layers[1:], self.conditioning_linear_layers[1:]):
            output = self.non_linearity(output)
            output = main_layer(output)
            if conditional:
                output += cond_layer(conditioning_input)

        return output


class SimplMLP(torch.nn.Module):
    """A simple MLP block."""

    def __init__(self, layer_dimensions: List[int], non_linearity: nn.Module):
        """Initialization method.

        Args:
            layer_dimensions: list of dimensions. It is assumed to be of the form
                [main_input_dimension, hidden_dimensions1, hidden_dimension2,... , output_dimension].

            non_linearity: non-linearity to apply between linear layers.
        """
        super(SimplMLP, self).__init__()

        input_dimensions = layer_dimensions[:-1]
        output_dimensions = layer_dimensions[1:]

        self.linear_layers = create_linear_layers(input_dimensions, output_dimensions)
        self.non_linearity = non_linearity

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward method.

        Args:
            input: a torch tensor of shape [batch_size, input_dimension].

        Returns:
            output: a torch tensor of shape [batch_size, output_dimension].
        """
        first_layer = self.linear_layers[0]
        output = first_layer(input)

        for layer in self.linear_layers[1:]:
            output = layer(self.non_linearity(output))

        return output
