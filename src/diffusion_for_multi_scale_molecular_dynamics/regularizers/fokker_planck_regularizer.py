from dataclasses import dataclass
from typing import Callable, Dict

import einops
import torch

from diffusion_for_multi_scale_molecular_dynamics.models.score_networks import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, CARTESIAN_FORCES, NOISE, NOISY_AXL_COMPOSITION, TIME, UNIT_CELL)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.exploding_variance import \
    SigmaCalculator


@dataclass(kw_only=True)
class RegularizerParameters:
    """Base Hyper-parameters for score networks."""

    # weighting prefactor for the regularizer loss.
    regularizer_lambda_weight: float = 1.0

    # how many terms should contribute to the regularization batch.
    # Should be no larger than the main batch size.
    batch_size: int

    # how many terms should be used in the score Laplacian approximation
    number_of_hte_terms: int

    # Define the noise schedule. Should be consistent with the score parameters.
    sigma_min: float
    sigma_max: float


class FokkerPlanckRegularizer:
    """Fokker-Planck Regularizer.

    This class implements a modified version of the 'score Fokker-Planck' residual for the purpose
    of regularizing a score network.

    Autodiff is used for the time derivative and the first order space derivative. The
    Hutchinson trace estimator is used for the second order space derivative.
    """

    def __init__(self, regularizer_parameters: RegularizerParameters):
        """Init method."""
        self.sigma_calculator = SigmaCalculator(
            sigma_min=regularizer_parameters.sigma_min,
            sigma_max=regularizer_parameters.sigma_max,
        )
        self.number_of_hte_terms = regularizer_parameters.number_of_hte_terms
        self.lbda_weight = regularizer_parameters.regularizer_lambda_weight
        self.regularizer_batch_size = regularizer_parameters.batch_size

    def _create_batch(
        self,
        relative_coordinates: torch.Tensor,
        times: torch.Tensor,
        atom_types: torch.Tensor,
        unit_cells: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        sigmas_t = self.sigma_calculator(times)

        forces = torch.zeros_like(relative_coordinates)

        composition = AXL(A=atom_types, X=relative_coordinates, L=unit_cells)

        batch = {
            NOISY_AXL_COMPOSITION: composition,
            NOISE: sigmas_t,
            TIME: times,
            UNIT_CELL: unit_cells,
            CARTESIAN_FORCES: forces,
        }
        return batch

    def _create_score_function(
        self,
        score_network: ScoreNetwork,
        atom_types: torch.Tensor,
        unit_cells: torch.Tensor,
    ):
        """Create score function.

        Args:
            score_network: a Score Network for which we seek the Fokker-Planck residual.
            atom_types: atom types, held fixed.
            unit_cells: unit cells, held fixed.

        Returns:
            score_function: a callable with input (relative_coordinates, time) which computes the scores, for
                atom_types and unit_cells held fixed.
        """

        def score_function(relative_coordinates, times):
            batch_size, natoms, spatial_dimension = relative_coordinates.shape
            batch = self._create_batch(
                relative_coordinates, times, atom_types, unit_cells
            )

            sigmas_t = einops.repeat(
                self.sigma_calculator(times),
                "batch 1 -> batch natoms d",
                natoms=natoms,
                d=spatial_dimension,
            )

            sigma_normalized_scores = score_network(batch, conditional=False).X
            scores = sigma_normalized_scores / sigmas_t
            return scores

        return score_function

    def _create_rademacher_random_variables(
        self, batch_size, num_atoms, spatial_dimension
    ):
        # A rademacher random variable can only take values in {-1, +1}.
        rademacher = (
            2
            * torch.randint(
                2, (self.number_of_hte_terms, batch_size, num_atoms, spatial_dimension)
            )
            - 1.0
        )
        return rademacher

    @staticmethod
    def get_hte_laplacian_term(
        score_function_x: Callable,
        relative_coordinates: torch.Tensor,
        rademacher_z: torch.Tensor,
    ):
        """Get HTE Laplacian term.

        Compute a term in the Hutchinson trace estimator (HTE) of the laplacian.

        Args:
            score_function_x: the score function, restricted to a single input, the relative_coordinates.
            relative_coordinates: spatial input for the score function at which we seek the Laplacian.
            rademacher_z: a rademacher random variable, assumed to have a shape compatible with `relative_coordinates`.

        Returns:
            laplacian_term: one term in the HTE estimate of the Laplacian
        """

        def jvp_with_z_func(x):
            return torch.func.jvp(score_function_x, (x,), (rademacher_z,))[1]

        approximate_score_laplacian = torch.func.jvp(
            jvp_with_z_func, (relative_coordinates,), (rademacher_z,)
        )[1]
        return approximate_score_laplacian

    def compute_residual_components(
        self,
        score_network: ScoreNetwork,
        batch: Dict[str, torch.Tensor],
        rademacher_z: torch.Tensor,
    ):
        """Compute residual components.

        This method computes all the terms that are needed to compute the Fokker-Planck residual.

        Args:
            score_network: the score network.
            batch: the batch for which we compute the residual components.
            rademacher_z: rademacher random variables, of
                dimension [number_of_hte_terms, (relative_coordinates_dimension)].

        Returns:
            scores: the scores.
            scores_time_derivative: the scores time derivative.
            scores_divergence_scores: the term score cdot nabla score.
            approximate_scores_laplacian: an HTE appraximation to nabla^2 score.
        """
        relative_coordinates = batch[NOISY_AXL_COMPOSITION].X
        times = batch[TIME]
        atom_types = batch[NOISY_AXL_COMPOSITION].A
        unit_cells = batch[UNIT_CELL]

        score_function = self._create_score_function(
            score_network=score_network, atom_types=atom_types, unit_cells=unit_cells
        )
        scores = score_function(relative_coordinates, times)

        def _batched_summed_score_function(relative_coordinates, times):
            # A convenience function that sums over the batch dimension. Since all the
            # batch elements are independent, differentiating with respect to this function is
            # equivalent to differentiating the original score_function, but minus the extra batch dimensions
            # in the Jacobian tensors.
            return score_function(relative_coordinates, times).sum(dim=0)

        # The _batch_summed_score_function has
        #   - the time input has dimension [batch_size, 1]
        #   - the output has dimension [number_of_atoms, spatial_dimension] : the batch is summed over!
        # The Jacobian tensor below will thus have dimensions [number_of_atoms, spatial_dimension, batch_size, 1]
        time_jacobian = torch.func.jacrev(_batched_summed_score_function, argnums=1)(
            relative_coordinates, times
        )

        # Rearrange the dimensions so that we output a tensor of
        #   dimension [batch_size, number_of_atims, spatial_dimension]
        scores_time_derivative = time_jacobian.squeeze(-1).permute(2, 0, 1)

        def score_function_x(relative_coordinates):
            # A convenience function that only explicitly depends on the relative coordinates
            return score_function(relative_coordinates, times)

        # This is the (S \dot nabla) S term
        scores_divergence_scores = torch.func.jvp(
            score_function_x, (relative_coordinates,), (scores,)
        )[1]

        list_z = rademacher_z.to(relative_coordinates.device)

        list_laplacian_terms = [
            self.get_hte_laplacian_term(score_function_x, relative_coordinates, z)
            for z in list_z
        ]

        approximate_scores_laplacian = torch.stack(list_laplacian_terms).mean(dim=0)

        return (
            scores,
            scores_time_derivative,
            scores_divergence_scores,
            approximate_scores_laplacian,
        )

    def compute_score_fokker_planck_residuals(
        self, score_network: ScoreNetwork, batch: Dict[str, torch.Tensor]
    ):
        """Compute score Fokker-Planck residual.

        Args:
            score_network: the score network.
            batch: a batch containing all that is required to call the score network.

        Returns:
            residuals: the score Fokker-Planck residual.
        """
        batch_size, natoms, spatial_dimension = batch[NOISY_AXL_COMPOSITION].X.shape

        times = batch[TIME]

        sigma = self.sigma_calculator.get_sigma(times)
        sigma_dot = self.sigma_calculator.get_sigma_time_derivative(times)

        sigma_term = einops.repeat(
            sigma * sigma_dot,
            "batch 1 -> batch natoms space",
            natoms=natoms,
            space=spatial_dimension,
        )

        rademacher_z = self._create_rademacher_random_variables(
            batch_size, natoms, spatial_dimension
        )
        (
            scores,
            scores_time_derivative,
            scores_divergence_scores,
            approximate_scores_laplacian,
        ) = self.compute_residual_components(score_network, batch, rademacher_z)

        residuals = scores_time_derivative - sigma_term * (
            2.0 * scores_divergence_scores + approximate_scores_laplacian
        )

        return residuals

    def compute_weighted_regularizer_loss(
        self,
        score_network: ScoreNetwork,
        external_times: torch.Tensor,
        external_atom_types: torch.Tensor,
        external_unit_cells: torch.Tensor,
    ):
        """Compute weighted regularizer loss.

        The loss calculation will rely on externally generated tensors for times, unit cells and atom types,
        presumably coming from Noiser classes. If there is a mismatch between the batch size of these tensors
        and the internal batch size, the smallest of the two will be used.

        Random relative coordinates will be generated internally, forcing the regularizer to see the whole space
        at all times.

        Args:
            score_network: the score network.
            external_times: times, of dimensions [external_batch_size, 1]
            external_atom_types:  atom types, of dimensions [external_batch_size, natoms]
            external_unit_cells:  unit cells, of dimensions [external_batch_size, spatial_dimension, spatial_dimension]

        Returns:
            weighted_loss: the weighted regularizer loss.
        """
        external_batch_size, natoms = external_atom_types.shape
        _, spatial_dimension, _ = external_unit_cells.shape

        if self.regularizer_batch_size <= external_batch_size:
            batch_size = self.regularizer_batch_size
        else:
            batch_size = external_batch_size

        times = external_times[:batch_size]
        atom_types = external_atom_types[:batch_size]
        unit_cells = external_unit_cells[:batch_size]

        # sample random relative_coordinates
        relative_coordinates = torch.rand(batch_size, natoms, spatial_dimension)

        batch = self._create_batch(relative_coordinates, times, atom_types, unit_cells)

        residuals = self.compute_score_fokker_planck_residuals(score_network, batch)

        weighted_loss = self.lbda_weight * torch.mean(residuals**2)
        return weighted_loss
