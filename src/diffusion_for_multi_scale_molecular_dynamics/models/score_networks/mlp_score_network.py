from dataclasses import dataclass
from typing import AnyStr, Dict

import torch
from torch import nn

from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network import (
    ScoreNetwork, ScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    CARTESIAN_FORCES, NOISE, NOISY_RELATIVE_COORDINATES)


@dataclass(kw_only=True)
class MLPScoreNetworkParameters(ScoreNetworkParameters):
    """Specific Hyper-parameters for MLP score networks."""

    architecture: str = "mlp"
    number_of_atoms: int  # the number of atoms in a configuration.
    n_hidden_dimensions: int  # the number of hidden layers.
    hidden_dimensions_size: int  # the dimensions of the hidden layers.
    embedding_dimensions_size: (
        int  # the dimension of the embedding of the noise parameter.
    )
    condition_embedding_size: int = (
        64  # dimension of the conditional variable embedding
    )


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
        hidden_dimensions = [
            hyper_params.hidden_dimensions_size
        ] * hyper_params.n_hidden_dimensions
        self._natoms = hyper_params.number_of_atoms

        output_dimension = self.spatial_dimension * self._natoms
        input_dimension = output_dimension + hyper_params.embedding_dimensions_size

        self.noise_embedding_layer = nn.Linear(
            1, hyper_params.embedding_dimensions_size
        )

        self.condition_embedding_layer = nn.Linear(
            output_dimension, hyper_params.condition_embedding_size
        )

        self.flatten = nn.Flatten()
        self.mlp_layers = nn.ModuleList()
        self.conditional_layers = nn.ModuleList()
        input_dimensions = [input_dimension] + hidden_dimensions
        output_dimensions = hidden_dimensions + [output_dimension]

        for input_dimension, output_dimension in zip(
            input_dimensions, output_dimensions
        ):
            self.mlp_layers.append(nn.Linear(input_dimension, output_dimension))
            self.conditional_layers.append(
                nn.Linear(hyper_params.condition_embedding_size, output_dimension)
            )
        self.non_linearity = nn.ReLU()

    def _check_batch(self, batch: Dict[AnyStr, torch.Tensor]):
        super(MLPScoreNetwork, self)._check_batch(batch)
        number_of_atoms = batch[NOISY_RELATIVE_COORDINATES].shape[1]
        assert (
            number_of_atoms == self._natoms
        ), "The dimension corresponding to the number of atoms is not consistent with the configuration."

    def _forward_unchecked(
        self, batch: Dict[AnyStr, torch.Tensor], conditional: bool = False
    ) -> torch.Tensor:
        """Forward unchecked.

        This method assumes that the input data has already been checked with respect to expectations
        and computes the scores assuming that the data is in the correct format.

        Args:
            batch : dictionary containing the data to be processed by the model.
            conditional (optional): if True, do a forward as though the model was conditional on the forces.
                Defaults to False.

        Returns:
            computed_scores : the scores computed by the model.
        """
        relative_coordinates = batch[NOISY_RELATIVE_COORDINATES]
        # shape [batch_size, number_of_atoms, spatial_dimension]

        sigmas = batch[NOISE].to(relative_coordinates.device)  # shape [batch_size, 1]
        noise_embedding = self.noise_embedding_layer(
            sigmas
        )  # shape [batch_size, embedding_dimension]

        input = torch.cat([self.flatten(relative_coordinates), noise_embedding], dim=1)

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

        output = output.reshape(relative_coordinates.shape)
        return output
