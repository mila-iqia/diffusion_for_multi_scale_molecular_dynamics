from dataclasses import dataclass

import einops
import torch

from crystal_diffusion.models.score_networks import ScoreNetwork
from crystal_diffusion.namespace import (CARTESIAN_FORCES, NOISE,
                                         NOISY_RELATIVE_COORDINATES, TIME,
                                         UNIT_CELL)
from crystal_diffusion.samplers.exploding_variance import ExplodingVariance
from crystal_diffusion.samplers.variance_sampler import NoiseParameters


@dataclass(kw_only=True)
class FokkerPlankRegularizerParameters:
    """Specific Hyper-parameters for the Fokker Planck Regularization."""
    weight: float


class FokkerPlanckLossCalculator:
    """Fokker Planck Loss Calculator."""
    def __init__(self,
                 sigma_normalized_score_network: ScoreNetwork,
                 noise_parameters: NoiseParameters,
                 regularizer_parameters: FokkerPlankRegularizerParameters):
        """Init method."""
        self._weight = regularizer_parameters.weight
        self.fokker_planck_error_calculator = ScoreFokkerPlanckError(sigma_normalized_score_network,
                                                                     noise_parameters)

    def compute_fokker_planck_loss_term(self, augmented_batch):
        """Compute Fokker-Planck loss term."""
        fokker_planck_errors = self.fokker_planck_error_calculator.get_score_fokker_planck_error(
            augmented_batch[NOISY_RELATIVE_COORDINATES],
            augmented_batch[TIME],
            augmented_batch[UNIT_CELL],
        )
        fokker_planck_rmse = (fokker_planck_errors ** 2).mean().sqrt()
        return self._weight * fokker_planck_rmse


class ScoreFokkerPlanckError(torch.nn.Module):
    """Class to calculate the Score Fokker Planck Error.

    This concept is defined in the paper:
        "FP-Diffusion: Improving Score-based Diffusion Models by Enforcing  the Underlying Score Fokker-Planck Equation"

    The Fokker-Planck equation, which is applicable to the time-dependent probability distribution, is generalized
    to an ODE that the score should satisfy. The departure from satisfying this equation thus defines the FP error.

    The score Fokker-Planck equation is defined as:

        d S(x, t) / dt = 1/2 g(t)^2 nabla [ nabla.S(x,t) + |S(x,t)|^2]

    where S(x, t) is the score.

    The great advantage of this approach is that it only requires knowledge of the score (and its derivative), which
    is the quantity we seek to learn.
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

    def _get_scores(
        self,
        relative_coordinates: torch.Tensor,
        times: torch.Tensor,
        unit_cells: torch.Tensor,
    ) -> torch.Tensor:
        """Get Scores.

        This method computes the un-normalized score, as defined by the sigma_normalized_score_network.

        Args:
            relative_coordinates : relative coordinates. Dimensions : [batch_size, number_of_atoms, spatial_dimension].
            times : diffusion times. Dimensions : [batch_size, 1].
            unit_cells : unit cells. Dimensions : [batch_size, spatial_dimension, spatial_dimension].

        Returns:
            scores: the scores for given input. Dimensions : [batch_size, number_of_atoms, spatial_dimension].
        """
        forces = torch.zeros_like(relative_coordinates)
        sigmas = self.exploding_variance.get_sigma(times)

        batch_size, natoms, spatial_dimension = relative_coordinates.shape

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

        broadcast_sigmas = einops.repeat(
            sigmas,
            "batch 1 -> batch natoms space",
            natoms=natoms,
            space=spatial_dimension,
        )
        scores = sigma_normalized_scores / broadcast_sigmas
        return scores

    def _get_scores_square_norm(
        self,
        relative_coordinates: torch.Tensor,
        times: torch.Tensor,
        unit_cells: torch.Tensor,
    ) -> torch.Tensor:
        """Get Scores square norm.

        This method computes the square norm of the un-normalized score, as defined
        by the sigma_normalized_score_network.

        Args:
            relative_coordinates : relative coordinates. Dimensions : [batch_size, number_of_atoms, spatial_dimension].
            times : diffusion times. Dimensions : [batch_size, 1].
            unit_cells : unit cells. Dimensions : [batch_size, spatial_dimension, spatial_dimension].

        Returns:
            scores_square_norm: |scores|^2. Dimension: [batch_size].
        """
        scores = self._get_scores(relative_coordinates, times, unit_cells)

        flat_scores = einops.rearrange(
            scores, "batch natoms spatial_dimension -> batch (natoms spatial_dimension)"
        )

        square_norms = (flat_scores**2).sum(dim=1)
        return square_norms

    def _get_score_time_derivative(
        self,
        relative_coordinates: torch.Tensor,
        times: torch.Tensor,
        unit_cells: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the time derivative of the score using autograd."""
        assert times.requires_grad, "The input times must require grads."

        def scores_function(t_: torch.Tensor) -> torch.Tensor:
            return self._get_scores(relative_coordinates, t_, unit_cells)

        # The input variable, time, has dimension [batch_size, 1]
        # The output of the function, the score, has dimension [batch_size, natoms, spatial_dimension]
        # The jacobian will thus have dimensions [batch_size, natoms, spatial_dimension, batch_size, 1], that is,
        # every output differentiated with respect to every input.
        jacobian = torch.autograd.functional.jacobian(scores_function, times, create_graph=True)

        # Clearly, only "same batch element" is meaningful. We can squeeze out the needless last dimension in the time
        batch_size = relative_coordinates.shape[0]
        batch_idx = torch.arange(batch_size)
        score_time_derivative = jacobian.squeeze(-1)[batch_idx, :, :, batch_idx]
        return score_time_derivative

    def _get_score_divergence(
        self,
        relative_coordinates: torch.Tensor,
        times: torch.Tensor,
        unit_cells: torch.Tensor,
    ) -> torch.Tensor:
        """Compute nabla . Score."""
        assert (
            relative_coordinates.requires_grad
        ), "The input relative_coordinates must require grads."

        def scores_function(x_):
            return self._get_scores(x_, times, unit_cells)

        # The input variable, x, has dimension [batch_size, natoms, spatial_dimension]
        # The output of the function, the score, has dimension [batch_size, natoms, spatial_dimension]
        # The jacobian will thus have dimensions
        #   [batch_size, natoms, spatial_dimension, batch_size, natoms, spatial_dimension]
        # every output differentiated with respect to every input.
        jacobian = torch.autograd.functional.jacobian(
            scores_function, relative_coordinates, create_graph=True,
        )

        flat_jacobian = einops.rearrange(
            jacobian,
            "batch1 natoms1 space1 batch2 natoms2 space2 -> batch1 batch2 (natoms1 space1) (natoms2 space2)",
        )

        # Clearly, only "same batch element" is meaningful. We can squeeze out the needless last dimension in the time
        batch_size = relative_coordinates.shape[0]
        batch_idx = torch.arange(batch_size)
        batch_flat_jacobian = flat_jacobian[batch_idx, batch_idx]

        divergence = einops.einsum(batch_flat_jacobian, "batch f f -> batch")

        return divergence

    def _get_gradient_term(
        self,
        relative_coordinates: torch.Tensor,
        times: torch.Tensor,
        unit_cells: torch.Tensor,
    ) -> torch.Tensor:
        """Compute nabla [ nabla.Score + |Score|^2]."""
        assert (
            relative_coordinates.requires_grad
        ), "The input relative_coordinates must require grads."

        def term_function(x_):
            # The "term" is nabla . s + |s|^2
            return (self._get_score_divergence(x_, times, unit_cells)
                    + self._get_scores_square_norm(x_, times, unit_cells))

        # The input variable, x, has dimension [batch_size, natoms, spatial_dimension]
        # The output of the function, the term_function, has dimension [batch_size]
        # The jacobian will thus have dimensions
        #   [batch_size, batch_size, natoms, spatial_dimension]
        # every output differentiated with respect to every input.
        jacobian = torch.autograd.functional.jacobian(
            term_function, relative_coordinates, create_graph=True,
        )

        # Clearly, only "same batch element" is meaningful. We can squeeze out the needless last dimension in the time
        batch_size = relative_coordinates.shape[0]
        batch_idx = torch.arange(batch_size)
        gradient_term = jacobian[batch_idx, batch_idx]
        return gradient_term

    def get_score_fokker_planck_error(
        self,
        relative_coordinates: torch.Tensor,
        times: torch.Tensor,
        unit_cells: torch.Tensor,
    ) -> torch.Tensor:
        """Get Score Fokker-Planck Error.

        Args:
            relative_coordinates : relative coordinates. Dimensions : [batch_size, number_of_atoms, spatial_dimension].
            times : diffusion times. Dimensions : [batch_size, 1].
            unit_cells : unit cells. Dimensions : [batch_size, spatial_dimension, spatial_dimension].

        Returns:
            FP_error: how much the score Fokker-Planck equation is violated. Dimensions : [batch_size].
        """
        batch_size, natoms, spatial_dimension = relative_coordinates.shape
        t = times.clone().detach().requires_grad_(True)
        d_score_dt = self._get_score_time_derivative(
            relative_coordinates, t, unit_cells
        )

        x = relative_coordinates.clone().detach().requires_grad_(True)
        gradient_term = self._get_gradient_term(x, times, unit_cells)

        time_prefactor = einops.repeat(
            0.5 * self.exploding_variance.get_g_squared(times),
            "batch 1 -> batch natoms spatial_dimension",
            natoms=natoms,
            spatial_dimension=spatial_dimension,
        )

        fp_errors = d_score_dt - time_prefactor * gradient_term
        return fp_errors
