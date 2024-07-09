"""Analytical Score Network.

This module implements an "exact" score network that is obtained under the approximation that the
atomic positions are just small displacements around some equilibrium positions and that the
energy is purely harmonic (ie, quadratic in the displacements).  Furthermore, it is assumed that
the covariance matrix is proportional to the identity; this makes the lattice sums manageable.

Optionally, the score network is made permutation invariant by summing on all atomic permutations.

This goal of this module is to investigate and understand the properties of diffusion. It is not
meant to generate 'production' results.
"""
import itertools
from dataclasses import dataclass
from typing import Any, AnyStr, Dict

import einops
import torch

from crystal_diffusion.models.score_networks.score_network import (
    ScoreNetwork, ScoreNetworkParameters)
from crystal_diffusion.namespace import NOISE, NOISY_RELATIVE_COORDINATES


@dataclass(kw_only=True)
class AnalyticalScoreNetworkParameters(ScoreNetworkParameters):
    """Specific Hyper-parameters for analytical score networks."""
    architecture: str = 'analytical'
    number_of_atoms: int   # the number of atoms in a configuration.
    kmax: int  # the maximum lattice translation along any dimension. Translations will be [-kmax,..,kmax].
    equilibrium_relative_coordinates: torch.Tensor  # Should have shape [number_of_atoms, spatial_dimensions]
    # Harmonic energy is defined as U = 1/2 u^T . Phi . u for "u"
    # the relative coordinate displacements. The 'inverse covariance' is beta Phi
    # and should be unitless. This is assumed to be proportional to the identity, such that
    # (beta Phi)^{-1} = sigma_d^2 Id, where the shape of Id is
    #   [number_of_atoms, spatial_dimensions,number_of_atoms, spatial_dimensions]
    variance_parameter: float  # sigma_d^2
    use_permutation_invariance: bool = False  # should the analytical score consider every coordinate permutations.


class AnalyticalScoreNetwork(ScoreNetwork):
    """Score network based on analytical integration of Gaussian distributions.

    This 'score network' is for exploring and debugging.
    """

    def __init__(self, hyper_params: AnalyticalScoreNetworkParameters):
        """__init__.

        Args:
            hyper_params : hyper parameters from the config file.
        """
        super(AnalyticalScoreNetwork, self).__init__(hyper_params)

        self.natoms = hyper_params.number_of_atoms
        self.spatial_dimension = hyper_params.spatial_dimension
        self.nd = self.natoms * self.spatial_dimension
        self.kmax = hyper_params.kmax

        assert hyper_params.variance_parameter > 0., "the variance parameter should be positive."
        self.sigma_d_square = hyper_params.variance_parameter

        assert hyper_params.equilibrium_relative_coordinates.shape == (self.natoms, self.spatial_dimension), \
            "equilibrium relative coordinates have the wrong shape"

        self.use_permutation_invariance = hyper_params.use_permutation_invariance
        self.device = hyper_params.equilibrium_relative_coordinates.device

        # shape: [number_of_translations]
        self.translations_k = self._get_all_translations(self.kmax).to(self.device)
        self.number_of_translations = len(self.translations_k)

        self.equilibrium_relative_coordinates = hyper_params.equilibrium_relative_coordinates

        if self.use_permutation_invariance:
            # Shape : [natom!, natoms, spatial dimension]
            self.all_x0 = self._get_all_equilibrium_permutations(self.equilibrium_relative_coordinates)
        else:
            # Shape : [1, natoms, spatial dimension]
            self.all_x0 = einops.rearrange(hyper_params.equilibrium_relative_coordinates, 'natom d -> 1 natom d')

    @staticmethod
    def _get_all_translations(kmax: int) -> torch.Tensor:
        return torch.arange(-kmax, kmax + 1)

    @staticmethod
    def _get_all_equilibrium_permutations(relative_coordinates: torch.Tensor) -> torch.Tensor:

        number_of_atoms = relative_coordinates.shape[0]

        # Shape : [number of permutations, number of atoms]
        perm_indices = torch.stack([torch.tensor(perm) for perm in itertools.permutations(range(number_of_atoms))])
        equilibrium_permutations = relative_coordinates[perm_indices]
        return equilibrium_permutations

    def _forward_unchecked(self, batch: Dict[AnyStr, Any], conditional: bool = False) -> torch.Tensor:
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
        xt = batch[NOISY_RELATIVE_COORDINATES]
        xt.requires_grad_(True)

        list_unnormalized_log_prob = []
        for x0 in self.all_x0:
            unnormalized_log_prob = self._compute_unnormalized_log_probability(sigmas, xt, x0)
            list_unnormalized_log_prob.append(unnormalized_log_prob)

        list_unnormalized_log_prob = torch.stack(list_unnormalized_log_prob)
        log_probs = torch.logsumexp(list_unnormalized_log_prob, dim=0, keepdim=False)

        grad_outputs = [torch.ones_like(log_probs)]

        scores = torch.autograd.grad(outputs=[log_probs],
                                     inputs=[xt],
                                     grad_outputs=grad_outputs)[0]

        # We actually want sigma x score.
        broadcast_sigmas = einops.repeat(sigmas, 'batch 1 -> batch n d', n=self.natoms, d=self.spatial_dimension)
        sigma_normalized_scores = broadcast_sigmas * scores

        return sigma_normalized_scores

    def _compute_unnormalized_log_probability(self, sigmas: torch.Tensor,
                                              xt: torch.Tensor,
                                              x_eq: torch.Tensor) -> torch.Tensor:

        batch_size = sigmas.shape[0]

        # Recast various spatial arrays to the correct dimensions to combine them,
        # in dimensions [batch, nd, number_of_translations]
        effective_variance = einops.repeat(sigmas ** 2 + self.sigma_d_square, 'batch 1 -> batch nd t',
                                           t=self.number_of_translations, nd=self.nd)

        sampling_coordinates = einops.repeat(xt, 'batch natom d -> batch (natom d) t',
                                             batch=batch_size, t=self.number_of_translations)

        equilibrium_coordinates = einops.repeat(x_eq, 'natom d -> batch (natom d) t',
                                                batch=batch_size, t=self.number_of_translations)

        translations = einops.repeat(self.translations_k, 't -> batch nd t', batch=batch_size, nd=self.nd)

        exponent = -0.5 * (sampling_coordinates - equilibrium_coordinates - translations) ** 2 / effective_variance
        # logsumexp on lattice translation vectors, then sum on spatial indices
        unnormalized_log_prob = torch.logsumexp(exponent, dim=2, keepdim=False).sum(dim=1)

        return unnormalized_log_prob
