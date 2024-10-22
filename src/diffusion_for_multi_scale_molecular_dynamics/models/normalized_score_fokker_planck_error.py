from typing import Callable

import einops
import torch
from crystal_diffusion.models.score_networks import ScoreNetwork
from crystal_diffusion.namespace import (CARTESIAN_FORCES, NOISE,
                                         NOISY_RELATIVE_COORDINATES, TIME,
                                         UNIT_CELL)
from crystal_diffusion.samplers.exploding_variance import ExplodingVariance
from src.crystal_diffusion.samplers.variance_sampler import NoiseParameters
from torch.func import jacrev


class NormalizedScoreFokkerPlanckError(torch.nn.Module):
    """Class to calculate the Normalized Score Fokker Planck Error.

    This concept is defined in the paper:
        "FP-Diffusion: Improving Score-based Diffusion Models by Enforcing  the Underlying Score Fokker-Planck Equation"

    The Fokker-Planck equation, which is applicable to the time-dependent probability distribution, is generalized
    to an ODE that the score should satisfy. The departure from satisfying this equation thus defines the FP error.

    The score Fokker-Planck equation is defined as:

        d S(x, t) / dt = 1/2 g(t)^2 nabla [ nabla.S(x,t) + |S(x,t)|^2]

    where S(x, t) is the score. Define the Normalized Score as N(x, t) == sigma(t) S(x, t), the equation above
    becomes

        d N(x, t) / dt = sigma_dot(t) / sigma(t) N(x, t) + sigma_dot(t) nabla [ sigma(t) nabla. N(x,t) + |N(x,t)|^2]

    where is it assumed that g(t)^2 == 2 sigma(t) sigma_dot(t).

    The great advantage of this approach is that it only requires knowledge of the normalized score
    (and its derivative), which is the quantity we seek to learn.
    """

    def __init__(
        self,
        sigma_normalized_score_network: ScoreNetwork,
        noise_parameters: NoiseParameters,
    ):
        """Init method."""
        super().__init__()

        self.exploding_variance = ExplodingVariance(noise_parameters)
        self.sigma_normalized_score_network = sigma_normalized_score_network

    def _normalized_scores_function(
        self,
        relative_coordinates: torch.Tensor,
        times: torch.Tensor,
        unit_cells: torch.Tensor,
    ) -> torch.Tensor:
        """Normalized Scores Function.

        This method computes the normalized score, as defined by the sigma_normalized_score_network.

        Args:
            relative_coordinates : relative coordinates. Dimensions : [batch_size, number_of_atoms, spatial_dimension].
            times : diffusion times. Dimensions : [batch_size, 1].
            unit_cells : unit cells. Dimensions : [batch_size, spatial_dimension, spatial_dimension].

        Returns:
            normalized scores: the scores for given input.
                Dimensions : [batch_size, number_of_atoms, spatial_dimension].
        """
        forces = torch.zeros_like(relative_coordinates)
        sigmas = self.exploding_variance.get_sigma(times)

        augmented_batch = {
            NOISY_RELATIVE_COORDINATES: relative_coordinates,
            TIME: times,
            NOISE: sigmas,
            UNIT_CELL: unit_cells,
            CARTESIAN_FORCES: forces,
        }

        sigma_normalized_scores = self.sigma_normalized_score_network(
            augmented_batch, conditional=False
        )

        return sigma_normalized_scores

    def _normalized_scores_square_norm_function(
        self,
        relative_coordinates: torch.Tensor,
        times: torch.Tensor,
        unit_cells: torch.Tensor,
    ) -> torch.Tensor:
        """Normalized Scores Square Norm Function.

        This method computes the square norm of the normalized score, as defined
        by the sigma_normalized_score_network.

        Args:
            relative_coordinates : relative coordinates. Dimensions : [batch_size, number_of_atoms, spatial_dimension].
            times : diffusion times. Dimensions : [batch_size, 1].
            unit_cells : unit cells. Dimensions : [batch_size, spatial_dimension, spatial_dimension].

        Returns:
            normalized_scores_square_norm: |normalized scores|^2. Dimension: [batch_size].
        """
        normalized_scores = self._normalized_scores_function(
            relative_coordinates, times, unit_cells
        )

        flat_scores = einops.rearrange(
            normalized_scores,
            "batch natoms spatial_dimension -> batch (natoms spatial_dimension)",
        )
        square_norms = (flat_scores**2).sum(dim=1)
        return square_norms

    def _get_dn_dt(
        self,
        relative_coordinates: torch.Tensor,
        times: torch.Tensor,
        unit_cells: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the time derivative of the normalized score."""
        # "_normalized_scores_function" is a Callable, with time as its second argument (index = 1)
        time_jacobian_function = jacrev(self._normalized_scores_function, argnums=1)

        # Computing the Jacobian returns an array of dimension [batch_size, natoms, space, batch_size, 1]
        time_jacobian = time_jacobian_function(relative_coordinates, times, unit_cells)

        # Only the "diagonal" along the batch dimensions is meaningful.
        # Also, squeeze out the needless last 'time' dimension.
        batch_diagonal = torch.diagonal(time_jacobian.squeeze(-1), dim1=0, dim2=3)

        # torch.diagonal puts the diagonal dimension (here, the batch index) at the end. Bring it back to the front.
        dn_dt = einops.rearrange(
            batch_diagonal, "natoms space batch -> batch natoms space"
        )

        return dn_dt

    def _get_gradient(
        self,
        scalar_function: Callable,
        relative_coordinates: torch.Tensor,
        times: torch.Tensor,
        unit_cells: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the gradient of the provided scalar function."""
        # We cannot use the "grad" function because our "scalar" function actually returns one value per batch entry.
        grad_function = jacrev(scalar_function, argnums=0)

        # Gradients have dimension [batch_size, batch_size, natoms, spatial_dimension]
        overbatched_gradients = grad_function(relative_coordinates, times, unit_cells)

        batch_diagonal = torch.diagonal(overbatched_gradients, dim1=0, dim2=1)

        # torch.diagonal puts the diagonal dimension (here, the batch index) at the end. Bring it back to the front.
        gradients = einops.rearrange(
            batch_diagonal, "natoms space batch -> batch natoms space"
        )
        return gradients

    def _divergence_function(
        self,
        relative_coordinates: torch.Tensor,
        times: torch.Tensor,
        unit_cells: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the divergence of the normalized score."""
        # "_normalized_scores_function" is a Callable, with space as its zeroth argument
        space_jacobian_function = jacrev(self._normalized_scores_function, argnums=0)

        # Computing the Jacobian returns an array of dimension [batch_size, natoms, space, batch_size, natoms, space]
        space_jacobian = space_jacobian_function(
            relative_coordinates, times, unit_cells
        )

        # Take only the diagonal batch term. "torch.diagonal" puts the batch index at the end...
        batch_diagonal = torch.diagonal(space_jacobian, dim1=0, dim2=3)

        flat_jacobian = einops.rearrange(
            batch_diagonal,
            "natoms1 space1 natoms2 space2 batch "
            "-> batch (natoms1 space1) (natoms2 space2)",
        )

        # take the trace of the Jacobian to get the divergence.
        divergence = torch.vmap(torch.trace)(flat_jacobian)
        return divergence

    def get_normalized_score_fokker_planck_error(
        self,
        relative_coordinates: torch.Tensor,
        times: torch.Tensor,
        unit_cells: torch.Tensor,
    ) -> torch.Tensor:
        """Get Normalized Score Fokker-Planck Error.

        Args:
            relative_coordinates : relative coordinates. Dimensions : [batch_size, number_of_atoms, spatial_dimension].
            times : diffusion times. Dimensions : [batch_size, 1].
            unit_cells : unit cells. Dimensions : [batch_size, spatial_dimension, spatial_dimension].

        Returns:
            FP_error: how much the normalized score Fokker-Planck equation is violated.
                Dimensions : [batch_size, spatial_dimension, spatial_dimension].
        """
        batch_size, natoms, spatial_dimension = relative_coordinates.shape

        sigmas = einops.repeat(
            self.exploding_variance.get_sigma(times),
            "batch 1 -> batch natoms space",
            natoms=natoms,
            space=spatial_dimension,
        )

        dot_sigmas = einops.repeat(
            self.exploding_variance.get_sigma_time_derivative(times),
            "batch 1 -> batch natoms space",
            natoms=natoms,
            space=spatial_dimension,
        )

        n = self._normalized_scores_function(relative_coordinates, times, unit_cells)

        dn_dt = self._get_dn_dt(relative_coordinates, times, unit_cells)

        grad_n2 = self._get_gradient(
            self._normalized_scores_square_norm_function,
            relative_coordinates,
            times,
            unit_cells,
        )

        grad_div_n = self._get_gradient(
            self._divergence_function, relative_coordinates, times, unit_cells
        )

        fp_errors = (
            dn_dt
            - dot_sigmas / sigmas * n
            - sigmas * dot_sigmas * grad_div_n
            - dot_sigmas * grad_n2
        )

        return fp_errors

    def get_normalized_score_fokker_planck_error_by_iterating_over_batch(
        self,
        relative_coordinates: torch.Tensor,
        times: torch.Tensor,
        unit_cells: torch.Tensor,
    ) -> torch.Tensor:
        """Get the error by iterating over the elements of the batch."""
        list_errors = []
        for x, t, c in zip(relative_coordinates, times, unit_cells):
            # Iterate over the elements of the batch. In effect, compute over "batch_size = 1" tensors.
            errors = self.get_normalized_score_fokker_planck_error(x.unsqueeze(0),
                                                                   t.unsqueeze(0),
                                                                   c.unsqueeze(0)).squeeze(0)
            list_errors.append(errors)

        return torch.stack(list_errors)
