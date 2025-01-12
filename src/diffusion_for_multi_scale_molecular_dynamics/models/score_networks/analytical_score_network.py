"""Analytical Score Network.

This module implements an "exact" score network that is obtained under the approximation that the
atomic positions are just small Gaussian-distributed displacements around some equilibrium positions.
Furthermore, it is assumed that the covariance matrix is proportional to the identity;
this makes the lattice sums manageable.

Optionally, the score network is made permutation invariant by summing on all atomic permutations.

The goal of this module is to investigate and understand the properties of diffusion. It is not
meant to generate 'production' results.
"""

from dataclasses import dataclass
from typing import Any, AnyStr, Dict, List, Tuple

import einops
import torch

from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network import (
    ScoreNetwork, ScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, NOISE, NOISY_AXL_COMPOSITION)
from diffusion_for_multi_scale_molecular_dynamics.score.wrapped_gaussian_score import (
    get_log_wrapped_gaussians, get_coordinates_sigma_normalized_score)
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from diffusion_for_multi_scale_molecular_dynamics.utils.symmetry_utils import \
    get_all_permutation_indices


@dataclass(kw_only=True)
class AnalyticalScoreNetworkParameters(ScoreNetworkParameters):
    """Specific Hyper-parameters for analytical score networks."""

    architecture: str = "analytical"

    # the number of atoms in a configuration.
    number_of_atoms: int

    # the maximum lattice translation along any dimension. Translations will be [-kmax,..,kmax].
    kmax: int

    equilibrium_relative_coordinates: List[List[float]]

    # the data distribution variance.
    sigma_d: float

    # should the analytical score consider every coordinate permutations.
    # Careful! The number of permutations will scale as number_of_atoms!. This will not
    # scale to large number of atoms.
    use_permutation_invariance: bool = False

    def __post_init__(self):
        """Post init."""
        assert self.sigma_d > 0.0, "the sigma_d parameter should be positive."

        assert len(self.equilibrium_relative_coordinates) == self.number_of_atoms, \
            "There should be exactly one list of equilibrium coordinates per atom."

        for x in self.equilibrium_relative_coordinates:
            assert len(x) == self.spatial_dimension, \
                "The equilibrium coordinates should be consistent with the spatial dimension."


class AnalyticalScoreNetwork(ScoreNetwork):
    """Score network based on analytical integration of Gaussian distributions.

    This 'score network' is for exploring and debugging.
    """

    def __init__(self, hyper_params: AnalyticalScoreNetworkParameters):
        """__init__.

        Args:
            hyper_params : hyper parameters from the config file.
            device: device to use.
        """
        super(AnalyticalScoreNetwork, self).__init__(hyper_params)

        self.number_of_atomic_classes = (
            hyper_params.num_atom_types + 1
        )  # account for the MASK class.
        self.natoms = hyper_params.number_of_atoms
        self.spatial_dimension = hyper_params.spatial_dimension
        self.nd = self.natoms * self.spatial_dimension
        self.kmax = hyper_params.kmax

        self.sigma_d_square = hyper_params.sigma_d**2

        self.use_permutation_invariance = hyper_params.use_permutation_invariance

        # shape: [number_of_translations]
        translations_k = self._get_all_translations(self.kmax)
        self.translations_k = torch.nn.Parameter(translations_k, requires_grad=False)

        self.number_of_translations = len(self.translations_k)

        self.equilibrium_relative_coordinates = torch.tensor(hyper_params.equilibrium_relative_coordinates,
                                                             dtype=torch.float)

        if self.use_permutation_invariance:
            # Shape : [natom!, natoms, spatial dimension]
            all_x0 = self._get_all_equilibrium_permutations(
                self.equilibrium_relative_coordinates
            )
        else:
            # Shape : [1, natoms, spatial dimension]
            all_x0 = einops.rearrange(
                self.equilibrium_relative_coordinates, "natom d -> 1 natom d"
            )

        self.all_x0 = torch.nn.Parameter(all_x0, requires_grad=False)

    @staticmethod
    def _get_all_translations(kmax: int) -> torch.Tensor:
        return torch.arange(-kmax, kmax + 1)

    @staticmethod
    def _get_all_equilibrium_permutations(
        relative_coordinates: torch.Tensor,
    ) -> torch.Tensor:

        number_of_atoms = relative_coordinates.shape[0]

        # Shape : [number of permutations, number of atoms]
        perm_indices, _ = get_all_permutation_indices(number_of_atoms)

        equilibrium_permutations = relative_coordinates[perm_indices]
        return equilibrium_permutations

    def get_log_wrapped_gaussians_and_normalized_scores_centered_on_equilibrium_positions(
        self, relative_coordinates: torch.tensor, sigmas_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all probabilities and normalized scores centered on equilibrium positions.

        Args:
            relative_coordinates : input relative coordinates: should be between 0 and 1.
            relative_coordinates has dimensions [batch, number_of_atoms, spatial_dimension]
            sigmas_t : the values of sigma(t). Should have the same dimension as relative coordinates.

        Returns:
            list_log_wrapped_gaussians: list of log wrapped gaussians, of
                dimensions [number_of_equilibrium_positions, batch]
            list_sigma_normalized_scores : list of sigma normalized scores, of
                dimensions [number_of_equilibrium_positions, batch, natoms, spatial_dimension]
        """
        assert (
            relative_coordinates.shape == sigmas_t.shape
        ), "relative_coordinates and sigmas_t have different shapes."
        assert (
            len(relative_coordinates.shape) == 3
        ), "relative_coordinates should have 3 dimensions."

        effective_sigmas = torch.sqrt(self.sigma_d_square + sigmas_t**2)

        number_of_equilibrium_positions = self.all_x0.shape[0]
        batch_size = relative_coordinates.shape[0]

        x = einops.repeat(
            relative_coordinates,
            "batch natoms  space -> (batch n) natoms space",
            n=number_of_equilibrium_positions,
        )

        x0 = einops.repeat(
            self.all_x0, "n natoms space -> (batch n) natoms space", batch=batch_size
        )

        u = map_relative_coordinates_to_unit_cell(x - x0)

        repeated_sigmas_t = einops.repeat(
            sigmas_t,
            "batch natoms  space -> (batch n) natoms space",
            n=number_of_equilibrium_positions,
        )

        repeated_effective_sigmas = einops.repeat(
            effective_sigmas,
            "batch natoms  space -> (batch n) natoms space",
            n=number_of_equilibrium_positions,
        )

        list_log_wrapped_gaussians = get_log_wrapped_gaussians(
            u, repeated_effective_sigmas, self.kmax
        )

        # We leverage the fact that the probability is a wrapped Gaussian to extract the
        # score. However, the normalized score thus obtained is improperly normalized.
        # the empirical scores are normalized with the time-dependent sigmas; the "data" sigma
        # are unknown (or even ill-defined) in general!
        list_effective_sigma_normalized_scores = get_coordinates_sigma_normalized_score(
            u, repeated_effective_sigmas, self.kmax
        )
        list_scores = list_effective_sigma_normalized_scores / repeated_effective_sigmas
        list_sigma_normalized_scores = repeated_sigmas_t * list_scores

        wrapped_gaussians = einops.rearrange(
            list_log_wrapped_gaussians,
            "(batch n) -> n batch",
            n=number_of_equilibrium_positions,
            batch=batch_size,
        )

        sigma_normalized_scores = einops.rearrange(
            list_sigma_normalized_scores,
            "(batch n) natoms space -> n batch natoms space",
            n=number_of_equilibrium_positions,
            batch=batch_size,
        )

        return wrapped_gaussians, sigma_normalized_scores

    def get_probabilities_and_normalized_scores(
        self, relative_coordinates: torch.tensor, sigmas_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get probabilities and normalized scores.

        Args:
            relative_coordinates: relative coordinates, of dimensions [batch_size, natoms, spatial_dimension]
            sigmas_t : the values of sigma(t). Should have the same dimension as relative coordinates.

        Returns:
            probabilities : probability P(x, t) at the input relative coordinates. Dimension [batch_size].
            normalized_scores : normalized scores sigma S(x, t) at the input relative coordinates.
                Dimension [batch_size, natoms, space_dimension].
        """
        batch_size, natoms, space_dimension = relative_coordinates.shape
        # list_log_w has dimensions [number_of_equilibrium_positions, batch_size]
        # list_s has dimensions [number_of_equilibrium_positions, batch_size, natoms, spatial_dimensions]
        list_log_w, list_s = (
            self.get_log_wrapped_gaussians_and_normalized_scores_centered_on_equilibrium_positions(
                relative_coordinates, sigmas_t
            )
        )

        number_of_equilibrium_positions = list_log_w.shape[0]

        probabilities = (
            torch.exp(list_log_w).sum(dim=0) / number_of_equilibrium_positions
        )

        list_weights = einops.repeat(
            torch.softmax(list_log_w, dim=0),
            "n batch -> n batch natoms space",
            natoms=natoms,
            space=space_dimension,
        )

        normalized_scores = (list_weights * list_s).sum(dim=0)

        return probabilities, normalized_scores

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
        _, sigma_normalized_scores = self.get_probabilities_and_normalized_scores(
            relative_coordinates=xt, sigmas_t=broadcast_sigmas
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

    def _compute_unnormalized_log_probability(
        self, sigmas: torch.Tensor, xt: torch.Tensor, x_eq: torch.Tensor
    ) -> torch.Tensor:

        batch_size = sigmas.shape[0]

        # Recast various spatial arrays to the correct dimensions to combine them,
        # in dimensions [batch, nd, number_of_translations]
        effective_variance = einops.repeat(
            sigmas**2 + self.sigma_d_square,
            "batch 1 -> batch nd t",
            t=self.number_of_translations,
            nd=self.nd,
        )

        sampling_coordinates = einops.repeat(
            xt,
            "batch natom d -> batch (natom d) t",
            batch=batch_size,
            t=self.number_of_translations,
        )

        equilibrium_coordinates = einops.repeat(
            x_eq,
            "natom d -> batch (natom d) t",
            batch=batch_size,
            t=self.number_of_translations,
        )

        translations = einops.repeat(
            self.translations_k, "t -> batch nd t", batch=batch_size, nd=self.nd
        )

        exponent = (
            -0.5
            * (sampling_coordinates - equilibrium_coordinates - translations) ** 2
            / effective_variance
        )
        # logsumexp on lattice translation vectors, then sum on spatial indices
        unnormalized_log_prob = torch.logsumexp(exponent, dim=2, keepdim=False).sum(
            dim=1
        )

        return unnormalized_log_prob


class TargetScoreBasedAnalyticalScoreNetwork(AnalyticalScoreNetwork):
    """Target Score-Based Analytical Score Network.

    An analytical score network that leverages the computation of the target for the score network.
    This can only work if the permutation equivariance is turned off. This should produce exactly the same results
    as the AnalyticalScoreNetwork, but does not require gradient calculation.
    """

    def __init__(self, hyper_params: AnalyticalScoreNetworkParameters):
        """__init__.

        Args:
            hyper_params : hyper parameters from the config file.
        """
        super(TargetScoreBasedAnalyticalScoreNetwork, self).__init__(hyper_params)
        assert (
            not hyper_params.use_permutation_invariance
        ), "This implementation is only valid in the absence of permutation equivariance."
        self.x0 = self.all_x0[0]

    def _forward_unchecked(
        self, batch: Dict[AnyStr, Any], conditional: bool = False
    ) -> torch.Tensor:
        """Forward unchecked.

        This method assumes that the input data has already been checked with respect to expectations
        and computes the scores assuming that the data is in the correct format.

        Args:
            batch : dictionary containing the data to be processed by the model.
            conditional (optional): CURRENTLY DOES NOTHING.

        Returns:
            output : the scores computed by the model as a [batch_size, n_atom, spatial_dimension] tensor.
        """
        sigmas = batch[NOISE]  # dimension: [batch_size, 1]
        xt = batch[NOISY_AXL_COMPOSITION].X
        batch_size = xt.shape[0]

        broadcast_sigmas = einops.repeat(
            sigmas,
            "batch 1 -> batch natoms spatial_dimension",
            natoms=self.natoms,
            spatial_dimension=self.spatial_dimension,
        )

        broadcast_effective_sigmas = (broadcast_sigmas**2 + self.sigma_d_square).sqrt()

        delta_relative_coordinates = map_relative_coordinates_to_unit_cell(xt - self.x0)
        misnormalized_scores = get_coordinates_sigma_normalized_score(
            delta_relative_coordinates, broadcast_effective_sigmas, kmax=self.kmax
        )

        sigma_normalized_scores = (
            broadcast_sigmas / broadcast_effective_sigmas * misnormalized_scores
        )

        # Mimic perfect predictions of single possible atomic type.
        atomic_logits = torch.zeros(batch_size, self.natoms, self.number_of_atomic_classes)
        atomic_logits[..., -1] = -torch.inf

        axl_scores = AXL(
            A=atomic_logits,
            X=sigma_normalized_scores,
            L=torch.zeros_like(sigma_normalized_scores),
        )

        return axl_scores
