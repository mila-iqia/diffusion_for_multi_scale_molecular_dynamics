"""Analytical Score Network.

This module implements an "exact" score network that is obtained under the approximation that the
atomic positions are just small displacements around some equilibrium positions and that the
energy is purely harmonic (ie, quadratic in the displacements). The score network is made
permutation invariant by summing on all atomic permutations.

This goal of this module is to investigate and understand the properties of diffusion. It is not
meant to generate 'production' results.
"""
import itertools
from dataclasses import dataclass
from typing import AnyStr, Dict

import einops
import torch

from crystal_diffusion.models.score_network import (ScoreNetwork,
                                                    ScoreNetworkParameters)
from crystal_diffusion.namespace import NOISE, NOISY_RELATIVE_COORDINATES


@dataclass(kw_only=True)
class AnalyticalScoreNetworkParameters(ScoreNetworkParameters):
    """Specific Hyper-parameters for MLP score networks."""
    architecture: str = 'analytical'
    number_of_atoms: int   # the number of atoms in a configuration.
    kmax: int  # the maximum lattice translation along any dimension. Translations will be [-kmax,..,kmax].
    equilibrium_relative_coordinates: torch.Tensor  # Should have shape [number_of_atoms, spatial_dimensions]
    # Harmonic energy is defined as U = 1/2 u^T . Phi . u for "u"
    # the relative coordinate displacements. The 'inverse covariance' is beta Phi
    # and should be unitless. The shape should be
    #   [number_of_atoms, spatial_dimensions,number_of_atoms, spatial_dimensions]
    inverse_covariance: torch.Tensor


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
        self.kmax = hyper_params.kmax

        self.flat_dim = self.natoms * self.spatial_dimension

        assert hyper_params.equilibrium_relative_coordinates.shape == (self.natoms, self.spatial_dimension), \
            "equilibrium relative coordinates have the wrong shape"

        expected_shape = (self.natoms, self.spatial_dimension, self.natoms, self.spatial_dimension)
        assert hyper_params.inverse_covariance.shape == expected_shape, "inverse covariance has the wrong shape"

        # Shape : [natom!, natoms, spatial dimension]
        permuted_equilibrium_relative_coordinates = (
            self._get_all_equilibrium_permutations(hyper_params.equilibrium_relative_coordinates))

        # Shape : [natom!, flat_dim]
        self.permutations_x0 = einops.rearrange(permuted_equilibrium_relative_coordinates, 'p a d -> p (a d)')

        # shape: [ (2 kmax + 1)^flat_dim,  flat_dim]
        self.translations_k = self._get_all_translations(self.kmax, self.flat_dim)

        # shape [ (2 kmax + 1)^flat_dim x natom!, flat_dim]
        self.all_offsets = self._get_all_flat_offsets(self.permutations_x0, self.translations_k)

        self.beta_phi_matrix = einops.rearrange(hyper_params.inverse_covariance,
                                                "n1 d1 n2 d2 -> (n1 d1) (n2 d2)")

        eigen = torch.linalg.eigh(self.beta_phi_matrix)
        self.spring_constants = eigen.eigenvalues
        self.eigenvectors_as_columns = eigen.eigenvectors

        assert torch.all(self.spring_constants > 0.), "the inverse covariance has non-positive eigenvalues."

    @staticmethod
    def _get_all_translations(kmax: int, flat_dim: int) -> torch.Tensor:
        shifts = range(-kmax, kmax + 1)
        all_translations = 1.0 * torch.tensor(list(itertools.product(shifts, repeat=flat_dim)))
        return all_translations

    @staticmethod
    def _get_all_equilibrium_permutations(relative_coordinates: torch.Tensor) -> torch.Tensor:

        number_of_atoms = relative_coordinates.shape[0]

        # Shape : [number of permutations, number of atoms]
        perm_indices = torch.stack([torch.tensor(perm) for perm in itertools.permutations(range(number_of_atoms))])
        equilibrium_permutations = relative_coordinates[perm_indices]
        return equilibrium_permutations

    @staticmethod
    def _get_all_flat_offsets(permutations_x0: torch.Tensor, translations_k: torch.Tensor) -> torch.Tensor:

        assert len(permutations_x0.shape) == 2
        assert len(translations_k.shape) == 2

        assert permutations_x0.shape[1] == translations_k.shape[1]

        all_offsets = permutations_x0.unsqueeze(0) + translations_k.unsqueeze(1)
        all_offsets = einops.rearrange(all_offsets, 'c1 c2 f -> (c1 c2) f')
        return all_offsets

    def _get_effective_inverse_covariance_matrices(self, sigmas: torch.Tensor) -> torch.Tensor:

        # Shape [batch_size, natom x spatial dimension]
        renormalized_spring_constants = self.spring_constants / (sigmas**2 * self.spring_constants + 1.0)
        lambda_matrix = torch.diag_embed(renormalized_spring_constants)

        batch_size = sigmas.shape[0]
        u_matrix = einops.repeat(self.eigenvectors_as_columns,
                                 'c1 c2 -> b c1 c2', b=batch_size, c1=self.flat_dim, c2=self.flat_dim)

        lambda_u_transpose = torch.bmm(lambda_matrix, u_matrix.transpose(2, 1))

        effective_inverse_covariance_matrices = torch.bmm(u_matrix, lambda_u_transpose)

        return effective_inverse_covariance_matrices

    def _forward_unchecked(self, batch: Dict[AnyStr, torch.Tensor]) -> torch.Tensor:
        """Forward unchecked.

        This method assumes that the input data has already been checked with respect to expectations
        and computes the scores assuming that the data is in the correct format.

        Args:
            batch : dictionary containing the data to be processed by the model.

        Returns:
            output : the scores computed by the model as a [batch_size, n_atom, spatial_dimension] tensor.
        """
        sigmas = batch[NOISE]  # dimension: [batch_size, 1]

        # Dimension: [batch_size, flat_dim, flat_dim]
        effective_inverse_covariance_matrices = self._get_effective_inverse_covariance_matrices(sigmas)

        xt = batch[NOISY_RELATIVE_COORDINATES]
        xt.requires_grad_(True)

        flat_xt = einops.rearrange(xt, 'b n d -> b (n d)')

        # shape [batch size, number of offsets, flat_dim]
        all_displacements = flat_xt.unsqueeze(1) - self.all_offsets.unsqueeze(0)

        m_u = einops.einsum(effective_inverse_covariance_matrices, all_displacements,
                            "batch f1 f2, batch o f2 -> batch f1 o")

        u_m_u = einops.einsum(all_displacements, m_u, "batch o f, batch f o -> batch o")

        exponent = -0.5 * u_m_u

        unnormalized_log_prob = torch.logsumexp(exponent, dim=1, keepdim=True)

        grad_outputs = [torch.ones_like(unnormalized_log_prob)]

        flat_scores = torch.autograd.grad(outputs=[unnormalized_log_prob],
                                          inputs=[flat_xt],
                                          grad_outputs=grad_outputs)[0]

        # We actually want sigma x score.
        flat_sigma_normalized_score = sigmas * flat_scores

        sigma_normalized_scores = einops.rearrange(flat_sigma_normalized_score, 'batch (n d) -> batch n d',
                                                   n=self.natoms, d=self.spatial_dimension)

        return sigma_normalized_scores
