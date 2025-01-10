from dataclasses import dataclass
from typing import Any, AnyStr, Dict, Tuple, Union

import einops
import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.langevin_generator import \
    LangevinGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.analytical_score_network import (
    AnalyticalScoreNetwork, AnalyticalScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, CARTESIAN_FORCES, NOISE, NOISY_AXL_COMPOSITION, TIME)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_scheduler import \
    Noise
from diffusion_for_multi_scale_molecular_dynamics.regularizers.regularizer import (
    Regularizer, RegularizerParameters)
from diffusion_for_multi_scale_molecular_dynamics.score.wrapped_gaussian_score import \
    get_coordinates_sigma_normalized_score
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell


@dataclass(kw_only=True)
class ConsistencyRegularizerParameters(RegularizerParameters):
    """Parameters for Consistency regularization."""

    type: str = "consistency"

    maximum_number_of_steps: int
    # convergence parameter for the Ewald-like sum of the perturbation kernel for coordinates.
    kmax_target_score: int = 4
    noise_parameters: NoiseParameters
    sampling_parameters: PredictorCorrectorSamplingParameters

    # As a sanity checking tool, an analytical score network can be provided for
    # the generation of samples from a trajectory
    analytical_score_network_parameters: Union[AnalyticalScoreNetworkParameters, None] = None


class ConsistencyRegularizer(Regularizer):
    """Consistency Regularizer.

    This class implements a regularizer based on on the consistency property.

    This is inspired by the paper

        Daras, Giannis, et al. "Consistent diffusion models: Mitigating sampling drift by learning
        to be consistent." Advances in Neural Information Processing Systems 36 (2024).

    but modified to be applicable to score networks in a periodic space. In particular, we use
    a target normalized score instead of the score itself to provide a more reliable signal.
    """

    def __init__(self, regularizer_parameters: ConsistencyRegularizerParameters):
        """Init method."""
        super().__init__(regularizer_parameters)
        self.noise_parameters = regularizer_parameters.noise_parameters
        self.sampling_parameters = regularizer_parameters.sampling_parameters

        self.maximum_number_of_steps = regularizer_parameters.maximum_number_of_steps
        self.kmax_target_score = regularizer_parameters.kmax_target_score

        self.analytical_score_network = None
        if regularizer_parameters.analytical_score_network_parameters:
            self.analytical_score_network = (
                AnalyticalScoreNetwork(regularizer_parameters.analytical_score_network_parameters))

    def get_augmented_batch_for_fixed_time(
        self, composition: AXL, time: float, sigma: float
    ):
        """Get augmented batch for fixed time.

        Args:
            composition: a batch composition
            time: the time for all elements in the composition
            sigma:  the corresponding value of sigma

        Returns:
            augmented_batch: input for a score netowrk.
        """
        batch_size = composition.X.shape[0]
        device = composition.X.device

        sigmas = sigma * torch.ones(batch_size, 1, device=device)
        times = time * torch.ones(batch_size, 1, device=device)

        forces = torch.zeros_like(composition.X)

        batch = {
            NOISY_AXL_COMPOSITION: composition,
            NOISE: sigmas,
            TIME: times,
            CARTESIAN_FORCES: forces,
        }
        return batch

    def get_partial_trajectory_start_and_end(
        self, start_time: float, noise: Noise
    ) -> Tuple[int, int, float, float]:
        """Get partial trajectory start and end indices.

        This method identifies the trajectory sampling time indices for a trajectory that begins at start_time
        and ends at end_time = start_time - [maximum_number_of_steps].

        Args:
            start_time: the start time of the trajectory
            noise: the noise instance that relates times and time steps.

        Returns:
            start_time_step_index: the index of the starting time.
            end_time_step_index: the index of the end of the partial trajectory.
            end_time: the time of the final step in the trajectory
            end_sigma: the sigma_t of the final step in the trajectory
        """
        # For a full trajectory, a generator takes the form
        #       for i = N-1, ..., 0
        #           x_i = Predictor(x_{i+1}, i + 1)
        #       return x_0
        #   where x_{N} is the "fully noised" sample and x_0 is the corresponding "fully denoised" sample.
        #
        #  Correspondingly, for a partial trajectory,
        #       for i = I_START-1, ..., I_END
        #           x_i = Predictor(x_{i+1})
        #       return x_{I_END}
        #  which will create the trajectory [x_{I_START-1}, x_{I_start-2}, ..., x_{I_END+1}, x_{I_END}], with
        #  I_START - I_END intermediate samples.
        #
        #  --> The STARTING POINT should be x_{I_START} and the END POINT should be x_{END}.
        #
        # The relationship between time and time index is:
        #        Predictor(x_{i}, i ) < ---- >  time_i = noise_scheduler.time[i - 1]
        # The offset in the index comes from the fact that the Predictor will always see i >= 1, and array indexing
        # in python starts at zero.
        #
        # This implies:
        #   Predictor(x_{I_START}, I_START)  < ---- >  start_time ~  noise_scheduler.time[I_START - 1]
        start_index = (noise.time - start_time).abs().argmin() + 1
        end_index = max(start_index - self.maximum_number_of_steps, 0)

        if end_index == 0:
            end_time = 0.0
            end_sigma = 0.0
        else:
            end_time = noise.time[end_index - 1]
            end_sigma = noise.sigma[end_index - 1]

        return start_index, end_index, end_time, end_sigma

    def generate_starting_composition(
        self, original_composition: AXL, batch_index: int
    ) -> AXL:
        """Generate starting composition.

        Args:
            original_composition: batched AXL composition
            batch_index: index of composition to be used as starting point.

        Returns:
            starting_composition: a batch composition similar to original_composition[batch_index], but with
                random relative coordinates.
        """
        relative_coordinates_shape = original_composition.X.shape
        batch_size = relative_coordinates_shape[0]

        # sample random relative_coordinates
        modified_relative_coordinates = torch.rand(*relative_coordinates_shape)

        modified_atom_types = einops.repeat(
            original_composition.A[batch_index], "... -> batch ...", batch=batch_size
        )

        modified_lattice_parameters = einops.repeat(
            original_composition.L[batch_index], "... -> batch ...", batch=batch_size
        )

        # Create the corresponding modified batch composition
        start_composition = AXL(
            A=modified_atom_types,
            X=modified_relative_coordinates,
            L=modified_lattice_parameters,
        )
        return start_composition

    def get_score_target(
        self,
        start_composition: AXL,
        end_composition: AXL,
        start_sigma: float,
        end_sigma: float,
    ) -> torch.Tensor:
        """Get score target.

        Args:
            start_composition: a batch composition, assumed to all be at the same start time
            end_composition: a batch composition, assumed to all be at the same end time
            start_sigma: the value of sigma for the start time.
            end_sigma: the value of sigma for the end time.

        Returns:
            target_normalized_score: the normalized score target, start_sigma nabla log K.
        """
        device = start_composition.X.device

        delta_relative_coordinates = map_relative_coordinates_to_unit_cell(
            start_composition.X - end_composition.X
        )

        effective_sigma = torch.tensor(start_sigma**2 - end_sigma**2).sqrt()
        effective_sigmas = effective_sigma * torch.ones_like(start_composition.X).to(
            device
        )

        # The output will be sigma_{eff} nabla log K
        wrongly_normalized_target_scores = get_coordinates_sigma_normalized_score(
            delta_relative_coordinates, effective_sigmas, kmax=self.kmax_target_score
        )

        # The desired target is start_sigma nabla log K
        target_normalized_scores = (
            start_sigma / effective_sigmas
        ) * wrongly_normalized_target_scores
        return target_normalized_scores

    def compute_regularizer_loss(
        self, score_network: ScoreNetwork, augmented_batch: Dict[AnyStr, Any]
    ) -> torch.Tensor:
        """Compute Regularizer Loss.

        Args:
            score_network: the score network to be regularized.
            augmented_batch: the augmented batch, which should contain all that is needed to call the score network.

        Returns:
            regularizer_loss: the regularizer loss.
        """
        if self.analytical_score_network:
            trajectory_network = self.analytical_score_network
        else:
            trajectory_network = score_network

        generator = LangevinGenerator(
            noise_parameters=self.noise_parameters,
            sampling_parameters=self.sampling_parameters,
            axl_network=trajectory_network,
        )
        noise: Noise = generator.noise

        original_noisy_composition = augmented_batch[NOISY_AXL_COMPOSITION]

        # Select a random element from the batch, and create a new composition from that element.
        batch_times = augmented_batch[TIME][:, 0]
        batch_size = len(batch_times)

        # Identify times that are large enough so that the we can make the required number of
        # steps without reaching the data manifold.
        valid_time_mask = batch_times > noise.time[self.maximum_number_of_steps]
        if not torch.any(valid_time_mask):
            # There are no valid times in this batch. No regularization.
            return torch.tensor(0.0)

        # Pick a random batch index that leads to a start time that is large enough.
        idx = torch.randint(valid_time_mask.sum(), ())
        random_batch_index = torch.arange(batch_size)[valid_time_mask][idx]

        start_time = augmented_batch[TIME][random_batch_index, 0]
        start_sigma = augmented_batch[NOISE][random_batch_index, 0]

        start_step_index, end_step_index, end_time, end_sigma = (
            self.get_partial_trajectory_start_and_end(start_time, noise)
        )

        start_composition = self.generate_starting_composition(
            original_noisy_composition, random_batch_index
        )

        # Generate samples: no need to keep the grads.
        with torch.no_grad():
            end_composition = generator.sample_from_noisy_composition(
                starting_noisy_composition=start_composition,
                starting_step_index=start_step_index,
                ending_step_index=end_step_index,
            )

        start_batch = self.get_augmented_batch_for_fixed_time(
            start_composition, start_time, start_sigma
        )
        start_normalized_score = score_network(start_batch).X

        normalized_score_target = self.get_score_target(
            start_composition, end_composition, start_sigma, end_sigma
        )

        # |s|^2 - 2 s . s_{target}
        loss = (
            torch.sum(
                start_normalized_score
                * (start_normalized_score - 2.0 * normalized_score_target)
            )
            / batch_size
        )

        return loss
