from typing import Optional

import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.langevin_generator import \
    LangevinGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.generators.trajectory_initializer import \
    TrajectoryInitializer
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters


class AdaptiveCorrectorGenerator(LangevinGenerator):
    """Langevin Dynamics Generator using only a corrector step with adaptive step size for relative coordinates.

    This class implements the Langevin Corrector generation of position samples, following
    Song et. al. 2021, namely:
        "SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS"
    """

    def __init__(
        self,
        noise_parameters: NoiseParameters,
        sampling_parameters: PredictorCorrectorSamplingParameters,
        axl_network: ScoreNetwork,
        trajectory_initializer: Optional[TrajectoryInitializer] = None,
    ):
        """Init method."""
        super().__init__(
            noise_parameters=noise_parameters,
            sampling_parameters=sampling_parameters,
            axl_network=axl_network,
            trajectory_initializer=trajectory_initializer,
        )
        self.corrector_r = noise_parameters.corrector_r

    def _relative_coordinates_update_predictor_step(
        self,
        relative_coordinates: torch.Tensor,
        sigma_normalized_scores: torch.Tensor,
        sigma_i: torch.Tensor,
        score_weight: torch.Tensor,
        gaussian_noise_weight: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Do not update the relative coordinates in the predictor."""
        return relative_coordinates

    def _lattice_parameters_update_predictor_step(
        self,
        lattice_parameters: torch.Tensor,
        sigma_normalized_scores: torch.Tensor,
        sigma_i: torch.Tensor,
        score_weight: torch.Tensor,
        gaussian_noise_weight: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Do not update the lattice parameters in the predictor."""
        return lattice_parameters

    def _get_coordinates_corrector_step_size(
        self,
        index_i: int,
        sigma_i: torch.Tensor,
        model_predictions_i: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the size of the corrector step for the relative coordinates update.

        The update is similar to the one for the lattice parameters, so this method is a wrapper for the more generic
        one used for relative coordinates and lattice parameters.
        """
        return self._generic_corrector_step_size(
            index_i, sigma_i, model_predictions_i, z, n_dim=3
        )

    def _get_lattice_parameters_corrector_step_size(
        self,
        index_i: int,
        sigma_n_i: torch.Tensor,
        model_predictions_i: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the size of the corrector step for the lattice parameters update.

        The update is similar to the one for the relatice coordinates, so this method is a wrapper for the more generic
        one used for relative coordinates and lattice parameters.
        """
        return self._generic_corrector_step_size(
            index_i, sigma_n_i, model_predictions_i, z, n_dim=2
        )

    def _generic_corrector_step_size(
        self,
        index_i: int,
        sigma_n_i: torch.Tensor,
        model_predictions_i: torch.Tensor,
        z: torch.Tensor,
        n_dim: int,
    ) -> torch.Tensor:
        r"""Compute the size of the corrector step for the relative coordinates (n_dim=3) or lattice parameters update
        (n_dim = 2).

        The prefactors determining the changes in the relative coordinates (lattice parameters) are determined using
        the sigma normalized score at that corrector step.
        The relative coordinates (lattice parameters) update is given by:

        .. math::

            x_i \leftarrow x_i + \epsilon_i * s(x_i, t_i) + \sqrt(2 \epsilon_i) z

        where :math:`s(x_i, t_i)` is the score, :math:`z` is a random variable drawn from a normal distribution and
        :math:`\epsilon_i` is given by:

        .. math::

            \epsilon_i = 2 \left(r \frac{||z||_2}{||s(x_i, t_i)||_2}\right)^2

        where :math:`r` is an hyper-parameter (0.17 by default) and :math:`||\cdot||_2` is the L2 norm.
        """
        # to compute epsilon_i, we need the norm of the score for each atom or lattice parameter.
        # For relative coordinates, taking the norm over the last 2 dimensions means summing the squared components
        # over the spatial dimension and the atoms, then taking the square-root.
        # For lattice parameters, we can take the norm over the last dimension only.
        norm_dims = [-2, -1] if n_dim == 3 else -1
        view_dims = (1, 1, 1) if n_dim == 3 else (1, 1)
        sigma_score_norm = (
            torch.linalg.norm(model_predictions_i, dim=norm_dims).mean()
        ).view(*view_dims)
        # note that sigma_score is \sigma * s(x, t), so we need to divide the norm by sigma to get the correct step size
        sigma_score_norm /= sigma_n_i
        # compute the norm of the z random noise similarly
        z_norm = torch.linalg.norm(z, dim=-1).mean().view(*view_dims)

        eps_i = (
            2
            * (
                self.corrector_r
                * z_norm
                / (sigma_score_norm.clip(min=self.small_epsilon))
            )
            ** 2
        )

        return eps_i
