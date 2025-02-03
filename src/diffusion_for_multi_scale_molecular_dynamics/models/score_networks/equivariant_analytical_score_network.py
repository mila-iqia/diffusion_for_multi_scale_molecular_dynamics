from dataclasses import dataclass
from typing import Any, AnyStr, Dict, List

import einops
import torch

from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network import (
    ScoreNetwork, ScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, NOISE, NOISY_AXL_COMPOSITION)
from diffusion_for_multi_scale_molecular_dynamics.score.wrapped_gaussian_score import \
    get_sigma_normalized_score
from diffusion_for_multi_scale_molecular_dynamics.transport.transporter import \
    Transporter
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from diffusion_for_multi_scale_molecular_dynamics.utils.geometric_utils import \
    get_cubic_point_group_symmetries


@dataclass(kw_only=True)
class EquivariantAnalyticalScoreNetworkParameters(ScoreNetworkParameters):
    """Specific Hyper-parameters for analytical score networks."""

    architecture: str = "equivariant_analytical"

    # the number of atoms in a configuration.
    number_of_atoms: int

    # the maximum lattice translation along any dimension. Translations will be [-kmax,..,kmax].
    kmax: int

    equilibrium_relative_coordinates: List[List[float]]

    use_point_group_symmetries: bool = False

    # the data distribution variance.
    sigma_d: float

    def __post_init__(self):
        """Post init."""
        assert self.sigma_d > 0.0, "the sigma_d parameter should be positive."

        assert (
            len(self.equilibrium_relative_coordinates) == self.number_of_atoms
        ), "There should be exactly one list of equilibrium coordinates per atom."

        for x in self.equilibrium_relative_coordinates:
            assert (
                len(x) == self.spatial_dimension
            ), "The equilibrium coordinates should be consistent with the spatial dimension."


class EquivariantAnalyticalScoreNetwork(ScoreNetwork):
    """Score network based on analytical integration of Gaussian distributions.

    This 'score network' is for exploring and debugging.
    """

    def __init__(self, hyper_params: EquivariantAnalyticalScoreNetworkParameters):
        """__init__.

        Args:
            hyper_params : hyper parameters from the config file.
            device: device to use.
        """
        super(EquivariantAnalyticalScoreNetwork, self).__init__(hyper_params)

        self.number_of_atomic_classes = (
            hyper_params.num_atom_types + 1
        )  # account for the MASK class.
        self.natoms = hyper_params.number_of_atoms
        self.spatial_dimension = hyper_params.spatial_dimension
        self.nd = self.natoms * self.spatial_dimension
        self.kmax = hyper_params.kmax

        self.sigma_d_square = hyper_params.sigma_d**2

        # shape: [number_of_translations]
        translations_k = self._get_all_translations(self.kmax)
        self.translations_k = torch.nn.Parameter(translations_k, requires_grad=False)

        if hyper_params.use_point_group_symmetries:
            self.point_group_operations = torch.nn.Parameter(
                get_cubic_point_group_symmetries(self.spatial_dimension),
                requires_grad=False,
            )
        else:
            self.point_group_operations = torch.nn.Parameter(
                torch.diag(torch.ones(self.spatial_dimension)).unsqueeze(0),
                requires_grad=False,
            )

        self.number_of_translations = len(self.translations_k)

        self.equilibrium_relative_coordinates = torch.tensor(
            hyper_params.equilibrium_relative_coordinates
        )

        self.transporter = Transporter(
            point_group_operations=self.point_group_operations,
            maximum_number_of_steps=10,
        )

    @staticmethod
    def _get_all_translations(kmax: int) -> torch.Tensor:
        return torch.arange(-kmax, kmax + 1)

    def get_normalized_scores(
        self, xt: torch.tensor, sigmas_t: torch.Tensor
    ) -> torch.Tensor:
        """Get normalized scores centered on closest equilibrium positions.

        Args:
            xt: input relative coordinates: should be between 0 and 1, with
                dimensions [batch, number_of_atoms, spatial_dimension]
            sigmas_t : the values of sigma(t). Should have the same dimension as relative coordinates.

        Returns:
            sigma_normalized_scores : sigma normalized scores, of
                dimensions [batch, natoms, spatial_dimension]
        """
        assert xt.shape == sigmas_t.shape, "xt and sigmas_t have different shapes."
        assert len(xt.shape) == 3, "relative_coordinates should have 3 dimensions."

        effective_sigmas = torch.sqrt(self.sigma_d_square + sigmas_t**2)

        batch_size = xt.shape[0]

        x = map_relative_coordinates_to_unit_cell(xt)

        y = einops.repeat(
            self.equilibrium_relative_coordinates,
            "natoms d -> batch natoms d",
            batch=batch_size,
        )

        nearest_equilibrium_coordinates = []
        for batch_idx in range(batch_size):
            transported_y, _ = self.transporter.get_optimal_transport(
                x[batch_idx], y[batch_idx]
            )
            nearest_equilibrium_coordinates.append(transported_y)

        nearest_equilibrium_coordinates = torch.stack(
            nearest_equilibrium_coordinates, dim=0
        )

        # We leverage the fact that the probability is a wrapped Gaussian to extract the
        # score.
        u = map_relative_coordinates_to_unit_cell(x - nearest_equilibrium_coordinates)
        effective_sigma_normalized_scores = get_sigma_normalized_score(
            u, effective_sigmas, self.kmax
        )
        sigma_normalized_scores = (
            sigmas_t * effective_sigma_normalized_scores / effective_sigmas
        )

        return sigma_normalized_scores

    def _forward_unchecked(
        self, batch: Dict[AnyStr, Any], conditional: bool = False
    ) -> AXL:
        """Forward unchecked.

        This method assumes that the input data has already been checked with respect to expectations
        and computes the scores assuming that the data is in the correct format.

        Args:
            batch : dictionary containing the data to be processed by the model.
            conditional (optional): CURRENTLY DOES NOTHING.

        Returns:
            output : an AXL namedtuple with
                    - the coordinates scores computed by the model as a [batch_size, n_atom, spatial_dimension] tensor.
                    - perfect atom type predictions, assuming a single atom type possibility.
                    - a tensor of zeros for the lattice parameters.
        """
        sigmas = batch[NOISE]  # dimension: [batch_size, 1]
        xt = batch[NOISY_AXL_COMPOSITION].X
        batch_size = xt.shape[0]

        broadcast_sigmas = einops.repeat(
            sigmas, "batch 1 -> batch n d", n=self.natoms, d=self.spatial_dimension
        )
        sigma_normalized_scores = self.get_normalized_scores(
            xt=xt, sigmas_t=broadcast_sigmas
        )

        # Mimic perfect predictions of single possible atomic type.
        atomic_logits = torch.zeros(
            batch_size, self.natoms, self.number_of_atomic_classes
        )
        atomic_logits[..., -1] = -torch.inf

        axl_scores = AXL(
            A=atomic_logits,
            X=sigma_normalized_scores,
            L=torch.zeros_like(sigma_normalized_scores),
        )

        return axl_scores
