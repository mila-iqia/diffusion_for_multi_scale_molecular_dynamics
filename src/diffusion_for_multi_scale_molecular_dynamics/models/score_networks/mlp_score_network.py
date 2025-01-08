from dataclasses import dataclass
from typing import Any, AnyStr, Dict

import torch
from torch import nn

from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network import (
    ScoreNetwork, ScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, CARTESIAN_FORCES, NOISE, NOISY_AXL_COMPOSITION, TIME)
from diffusion_for_multi_scale_molecular_dynamics.utils.d3pm_utils import \
    class_index_to_onehot
from diffusion_for_multi_scale_molecular_dynamics.utils.symmetry_utils import \
    get_all_permutation_indices


@dataclass(kw_only=True)
class MLPScoreNetworkParameters(ScoreNetworkParameters):
    """Specific Hyper-parameters for MLP score networks."""

    architecture: str = "mlp"
    number_of_atoms: int  # the number of atoms in a configuration.
    n_hidden_dimensions: int  # the number of hidden layers.
    hidden_dimensions_size: int  # the dimensions of the hidden layers.

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

    # should the analytical score consider every coordinate permutations.
    # Careful! The number of permutations will scale as number_of_atoms!. This will not
    # scale to large number of atoms.
    use_permutation_invariance: bool = False


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
        hidden_dimensions = [hyper_params.hidden_dimensions_size] * (
            hyper_params.n_hidden_dimensions
        )
        self._natoms = hyper_params.number_of_atoms
        self.num_atom_types = hyper_params.num_atom_types
        self.num_classes = self.num_atom_types + 1  # add 1 for the MASK class

        self.use_permutation_invariance = hyper_params.use_permutation_invariance
        if self.use_permutation_invariance:
            # Shape : [number of permutations, number of atoms]
            self.perm_indices, self.inverse_perm_indices = get_all_permutation_indices(self._natoms)

        # Each relative coordinate will be embedded on the unit circle with (cos, sin), leading to
        # a doubling of the number of dimensions.
        flat_relative_coordinates_input_dimension = 2 * self.spatial_dimension * self._natoms

        coordinate_output_dimension = self.spatial_dimension * self._natoms
        atom_type_output_dimension = self._natoms * self.num_classes

        input_dimension = (
            hyper_params.relative_coordinates_embedding_dimensions_size
            + hyper_params.noise_embedding_dimensions_size
            + hyper_params.time_embedding_dimensions_size
            + self._natoms * hyper_params.atom_type_embedding_dimensions_size
        )

        self.relative_coordinates_embedding_layer = nn.Linear(
            flat_relative_coordinates_input_dimension,
            hyper_params.relative_coordinates_embedding_dimensions_size)

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

        self.flatten = nn.Flatten()
        self.mlp_layers = nn.ModuleList()
        self.conditional_layers = nn.ModuleList()
        input_dimensions = [input_dimension] + hidden_dimensions[:-1]
        output_dimensions = hidden_dimensions

        for input_dimension, output_dimension in zip(
            input_dimensions, output_dimensions
        ):
            self.mlp_layers.append(nn.Linear(input_dimension, output_dimension))
            self.conditional_layers.append(
                nn.Linear(hyper_params.condition_embedding_size, output_dimension)
            )
        self.non_linearity = nn.SiLU()

        # Create a self nn object to be discoverable to be placed on the correct device
        self.output_A_layer = nn.Linear(hyper_params.hidden_dimensions_size, atom_type_output_dimension)
        self.output_X_layer = nn.Linear(hyper_params.hidden_dimensions_size, coordinate_output_dimension)
        self.output_L_layer = nn.Identity()
        self.output_layers = AXL(A=self.output_A_layer,
                                 X=self.output_X_layer,
                                 L=self.output_L_layer)  # TODO placeholder

    def _check_batch(self, batch: Dict[AnyStr, torch.Tensor]):
        super(MLPScoreNetwork, self)._check_batch(batch)
        number_of_atoms = batch[NOISY_AXL_COMPOSITION].X.shape[1]
        assert (
            number_of_atoms == self._natoms
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
        if self.use_permutation_invariance:
            # An equivariant vectorial score network takes the form
            #
            #       s_{sym}(x) = 1/|G| \sum_{g \in G} g^{-1}.s(g.x)
            #
            # The atom type predictions need only be invariant since they are scalars.
            #
            list_model_outputs = []
            for permutation, inverse_permutation in zip(self.perm_indices, self.inverse_perm_indices):
                permuted_batch = self.get_permuted_batch(batch, permutation)
                model_output = self._forward_unchecked_single_permutation(permuted_batch, conditional)
                permuted_model_output = AXL(A=model_output.A,
                                            X=model_output.X[:, inverse_permutation],
                                            L=model_output.L)

                list_model_outputs.append(permuted_model_output)

            output = AXL(A=torch.stack([output.A for output in list_model_outputs]).mean(dim=0),
                         X=torch.stack([output.X for output in list_model_outputs]).mean(dim=0),
                         L=torch.stack([output.L for output in list_model_outputs]).mean(dim=0)
                         )

        else:
            output = self._forward_unchecked_single_permutation(batch, conditional)

        return output

    def get_permuted_batch(self, batch, permutation) -> Dict[AnyStr, Any]:
        """Get permuted batch."""
        composition = batch[NOISY_AXL_COMPOSITION]
        new_composition = AXL(A=composition.A[:, permutation],
                              X=composition.X[:, permutation],
                              L=composition.L)

        permuted_batch = dict()

        for key in batch.keys():
            if key == NOISY_AXL_COMPOSITION:
                permuted_batch[key] = new_composition
            else:
                permuted_batch[key] = batch[key]

        return permuted_batch

    def _forward_unchecked_single_permutation(
        self, batch: Dict[AnyStr, torch.Tensor], conditional: bool = False
    ) -> AXL:
        """Forward unchecked single permutation.

        compute the model for a given configuration.

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
        relative_coordinates_input = self.flatten(
            torch.stack([angles.cos(), angles.sin()], dim=1))

        relative_coordinates_embedding = self.relative_coordinates_embedding_layer(relative_coordinates_input)

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
                noise_embedding,
                time_embedding,
                self.flatten(atom_type_embedding),
            ],
            dim=1,
        )

        forces_input = self.condition_embedding_layer(
            self.flatten(batch[CARTESIAN_FORCES])
        )

        output = input
        for i, (layer, condition_layer) in enumerate(
            zip(self.mlp_layers, self.conditional_layers)
        ):
            if i != 0:
                output = self.non_linearity(output)
            output = layer(output)
            if conditional:
                output += condition_layer(forces_input)

        coordinates_output = self.output_layers.X(output).reshape(
            relative_coordinates.shape
        )
        atom_types_output = self.output_layers.A(output).reshape(
            atom_types_one_hot.shape
        )
        lattice_output = torch.zeros_like(atom_types_output)  # TODO placeholder

        axl_output = AXL(A=atom_types_output, X=coordinates_output, L=lattice_output)
        return axl_output
