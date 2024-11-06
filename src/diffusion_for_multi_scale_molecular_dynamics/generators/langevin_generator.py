import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import (
    PredictorCorrectorAXLGenerator, PredictorCorrectorSamplingParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, CARTESIAN_FORCES, NOISE, NOISY_AXL_COMPOSITION, TIME, UNIT_CELL)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.variance_sampler import \
    NoiseScheduler
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from diffusion_for_multi_scale_molecular_dynamics.utils.d3pm_utils import (
    class_index_to_onehot, get_probability_at_previous_time_step)
from diffusion_for_multi_scale_molecular_dynamics.utils.sample_trajectory import (
    NoOpPredictorCorrectorSampleTrajectory, PredictorCorrectorSampleTrajectory)


class LangevinGenerator(PredictorCorrectorAXLGenerator):
    """Annealed Langevin Dynamics Generator.

    This class implements the annealed Langevin Dynamics generation of position samples, following
    Song & Ermon 2019, namely:
        "Generative Modeling by Estimating Gradients of the Data Distribution"
    """

    def __init__(
        self,
        noise_parameters: NoiseParameters,
        sampling_parameters: PredictorCorrectorSamplingParameters,
        axl_network: ScoreNetwork,
    ):
        """Init method."""
        super().__init__(
            number_of_discretization_steps=noise_parameters.total_time_steps,
            number_of_corrector_steps=sampling_parameters.number_of_corrector_steps,
            spatial_dimension=sampling_parameters.spatial_dimension,
            num_atom_types=sampling_parameters.num_atom_types,
        )

        self.noise_parameters = noise_parameters
        sampler = NoiseScheduler(
            noise_parameters, num_classes=sampling_parameters.num_atom_types + 1
        )
        self.noise, self.langevin_dynamics = sampler.get_all_sampling_parameters()
        self.number_of_atoms = sampling_parameters.number_of_atoms
        self.axl_network = axl_network
        self.small_epsilon = sampling_parameters.small_epsilon

        if sampling_parameters.record_samples:
            self.sample_trajectory_recorder = PredictorCorrectorSampleTrajectory()
        else:
            self.sample_trajectory_recorder = NoOpPredictorCorrectorSampleTrajectory()

    def initialize(
        self, number_of_samples: int, device: torch.device = torch.device("cpu")
    ):
        """This method must initialize the samples from the fully noised distribution."""
        # all atoms are initialized as masked
        atom_types = torch.ones(number_of_samples, self.number_of_atoms).long().to(
            device
        ) * (self.num_classes - 1)
        # relative coordinates are sampled from the uniform distribution
        relative_coordinates = torch.rand(
            number_of_samples, self.number_of_atoms, self.spatial_dimension
        ).to(device)
        lattice_vectors = torch.zeros_like(relative_coordinates).to(
            device
        )  # TODO placeholder
        init_composition = AXL(A=atom_types, X=relative_coordinates, L=lattice_vectors)
        return init_composition

    def _draw_gaussian_sample(self, number_of_samples):
        return torch.randn(
            number_of_samples, self.number_of_atoms, self.spatial_dimension
        )

    def _draw_gumbel_sample(self, number_of_samples):
        return -torch.log(
            -torch.log(
                torch.rand(
                    number_of_samples, self.number_of_atoms, self.num_classes
                ).clip(min=self.small_epsilon)
            )
        )

    def _get_model_predictions(
        self,
        composition: AXL,
        time: float,
        sigma_noise: float,
        unit_cell: torch.Tensor,  # TODO replace with AXL-L
        cartesian_forces: torch.Tensor,
    ) -> torch.Tensor:
        """Get sigma normalized scores.

        Args:
            composition : AXL composition with:
                atom types, of shape [number of samples, number_of_atoms]
                relative coordinates, of shape [number_of_samples, number_of_atoms, spatial_dimension]
                lattice vectors, of shape [number_of_samples, spatial_dimension * (spatial_dimension - 1)]  # TODO check
            time : time at which to evaluate the score
            sigma_noise: the diffusion sigma parameter corresponding to the time at which to evaluate the score
            unit_cell: unit cell definition in Angstrom of shape [number_of_samples, spatial_dimension,
                spatial_dimension]
            cartesian_forces: forces to condition the sampling from. Shape [number_of_samples, number_of_atoms,
                spatial_dimension]

        Returns:
            sigma normalized score: sigma x Score(x, t).
        """
        number_of_samples = composition.X.shape[0]

        time_tensor = time * torch.ones(number_of_samples, 1).to(composition.X)
        sigma_noise_tensor = sigma_noise * torch.ones(number_of_samples, 1).to(
            composition.X
        )
        augmented_batch = {
            NOISY_AXL_COMPOSITION: composition,
            TIME: time_tensor,
            NOISE: sigma_noise_tensor,
            UNIT_CELL: unit_cell,  # TODO replace with AXL-L
            CARTESIAN_FORCES: cartesian_forces,
        }

        # TODO do not hard-code conditional to False - need to be able to condition sampling
        model_predictions = self.axl_network(augmented_batch, conditional=False)
        return model_predictions

    def relative_coordinates_update(
        self,
        relative_coordinates: torch.Tensor,
        sigma_normalized_scores: torch.Tensor,
        sigma_i: torch.Tensor,
        score_weight: torch.Tensor,
        gaussian_noise_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Generic update for the relative coordinates.

        This is useful for both the predictor and the corrector step. The score weight and gaussian weight noise differs
        in these two settings.

        Args:
            relative_coordinates: starting coordinates. Dimension: [number_of_samples, number_of_atoms,
                spatial_dimension]
            sigma_normalized_scores: output of the model - an estimate of the normalized score sigma \nabla log p(x).
                Dimension: [number_of_samples, number_of_atoms, spatial_dimension]
            sigma_i: noise parameter for variance exploding noise scheduler. Dimension: [number_of_samples]
            score_weight: prefactor in front of the normalized score update. Should be g2_i in the predictor step and
                eps_i in the corrector step. Dimension: [number_of_samples]
            gaussian_noise_weight: prefactor in front of the random noise update. Should be g_i in the predictor step
                and sqrt_2eps_i in the corrector step. Dimension: [number_of_samples]

        Returns:
            updated_coordinates: relative coordinates after the update. Dimension: [number_of_samples, number_of_atoms,
                spatial_dimension].
        """
        number_of_samples = relative_coordinates.shape[0]
        z = self._draw_gaussian_sample(number_of_samples).to(relative_coordinates)
        updated_coordinates = (
            relative_coordinates
            + score_weight * sigma_normalized_scores / sigma_i
            + gaussian_noise_weight * z
        )
        # map back to the range [0, 1)
        updated_coordinates = map_relative_coordinates_to_unit_cell(updated_coordinates)
        return updated_coordinates

    def atom_types_update(
        self,
        predicted_logits: torch.Tensor,
        atom_types_i: torch.LongTensor,
        q_matrices_i: torch.Tensor,
        q_bar_matrices_i: torch.Tensor,
        q_bar_tm1_matrices_i: torch.Tensor,
    ) -> torch.LongTensor:
        """Generic update of the atom types.

        This should be used in the predictor step only.

        Args:
            predicted_logits: output of the model - an estimate of p(a_0 | a_t). Dimension:
                [number_of_samples, number_of_atoms, num_classes].
            atom_types_i: indices of the atom types at timestep i. Dimension:
                [number_of_samples, number_of_atoms]
            q_matrices_i: one-step transition matrix. Dimension: [number_of_samples, number_of_atoms, num_classes,
                num_classes].
            q_bar_matrices_i: cumulative transition matrix at time step i. Dimension: [number_of_samples,
                number_of_atoms, num_classes, num_classes].
            q_bar_tm1_matrices_i:  cumulative transition matrix at time step 'i - 1'. Dimension: [number_of_samples,
                number_of_atoms, num_classes, num_classes].

        Returns:
            a_im1: updated atom type indices. Dimension: [number_of_samples, number_of_atoms]
        """
        number_of_samples = predicted_logits.shape[0]
        u = self._draw_gumbel_sample(number_of_samples).to(predicted_logits.device)
        one_hot_atom_types_i = class_index_to_onehot(
            atom_types_i, num_classes=self.num_classes
        )
        one_step_transition_probs = get_probability_at_previous_time_step(
            probability_at_zeroth_timestep=predicted_logits,
            one_hot_probability_at_current_timestep=one_hot_atom_types_i,
            q_matrices=q_matrices_i,
            q_bar_matrices=q_bar_matrices_i,
            q_bar_tm1_matrices=q_bar_tm1_matrices_i,
            small_epsilon=self.small_epsilon,
            probability_at_zeroth_timestep_are_onehot=False,
        )  # p(a_{t-1} | a_t) as a [num_samples, num_atoms, num_classes] tensor
        # sample new atom types from p(a_{t-1} | a_t) using the gumbel trick
        a_im1 = torch.argmax(
            torch.log(one_step_transition_probs + self.small_epsilon) + u, dim=-1
        )
        # a_im1 has shape: number_of_samples, number_of_atoms and is a LongTensor
        return a_im1

    def predictor_step(
        self,
        composition_i: AXL,
        index_i: int,
        unit_cell: torch.Tensor,  # TODO replace with AXL-L
        cartesian_forces: torch.Tensor,
    ) -> AXL:
        """Predictor step.

        Args:
            composition_i : sampled composition (atom types, relative coordinates, lattice vectors), at time step i.
            index_i : index of the time step.
            unit_cell: sampled unit cell at time step i.
            cartesian_forces: forces conditioning the sampling process

        Returns:
            composition_im1 : sampled composition, at time step i - 1.
        """
        assert (
            1 <= index_i <= self.number_of_discretization_steps
        ), "The predictor step can only be invoked for index_i between 1 and the total number of discretization steps."

        idx = index_i - 1  # python starts indices at zero
        t_i = self.noise.time[idx].to(composition_i.X)
        g_i = self.noise.g[idx].to(composition_i.X)
        g2_i = self.noise.g_squared[idx].to(composition_i.X)
        sigma_i = self.noise.sigma[idx].to(composition_i.X)
        q_matrices_i = self.noise.q_matrix[idx].to(composition_i.X)
        q_bar_matrices_i = self.noise.q_bar_matrix[idx].to(composition_i.X)
        q_bar_tm1_matrices_i = self.noise.q_bar_tm1_matrix[idx].to(composition_i.X)

        model_predictions_i = self._get_model_predictions(
            composition_i, t_i, sigma_i, unit_cell, cartesian_forces
        )

        # atom types update
        a_im1 = self.atom_types_update(
            model_predictions_i.A,
            composition_i.A,
            q_matrices_i,
            q_bar_matrices_i,
            q_bar_tm1_matrices_i,
        )

        x_im1 = self.relative_coordinates_update(
            composition_i.X, model_predictions_i.X, sigma_i, g2_i, g_i
        )

        composition_im1 = AXL(A=a_im1, X=x_im1, L=composition_i.L)  # TODO placeholder

        self.sample_trajectory_recorder.record_unit_cell(
            unit_cell=unit_cell
        )  # TODO replace with AXL-L
        self.sample_trajectory_recorder.record_predictor_step(
            i_index=index_i,
            time=t_i,
            sigma=sigma_i,
            composition_i=composition_i,
            composition_im1=composition_im1,
            model_predictions_i=model_predictions_i,
        )

        return composition_im1

    def corrector_step(
        self,
        composition_i: AXL,
        index_i: int,
        unit_cell: torch.Tensor,  # TODO replace with AXL-L
        cartesian_forces: torch.Tensor,
    ) -> AXL:
        """Corrector Step.

        Note this is not affecting the atom types. Only the reduced coordinates and lattice vectors.

        Args:
            composition_i : sampled composition (atom types, relative coordinates, lattice vectors), at time step i.
            index_i : index of the time step.
            unit_cell: sampled unit cell at time step i.  # TODO replace with AXL-L
            cartesian_forces: forces conditioning the sampling

        Returns:
            corrected_composition_i : sampled composition, after corrector step.
        """
        assert 0 <= index_i <= self.number_of_discretization_steps - 1, (
            "The corrector step can only be invoked for index_i between 0 and "
            "the total number of discretization steps minus 1."
        )
        # The Langevin dynamics array are indexed with [0,..., N-1]
        eps_i = self.langevin_dynamics.epsilon[index_i].to(composition_i.X)
        sqrt_2eps_i = self.langevin_dynamics.sqrt_2_epsilon[index_i].to(composition_i.X)

        if index_i == 0:
            # TODO: we are extrapolating here; the score network will never have seen this time step...
            sigma_i = (
                self.noise_parameters.sigma_min
            )  # no need to change device, this is a float
            t_i = 0.0  # same for device - this is a float
        else:
            idx = index_i - 1  # python starts indices at zero
            sigma_i = self.noise.sigma[idx].to(composition_i.X)
            t_i = self.noise.time[idx].to(composition_i.X)

        model_predictions_i = self._get_model_predictions(
            composition_i, t_i, sigma_i, unit_cell, cartesian_forces
        )

        corrected_x_i = self.relative_coordinates_update(
            composition_i.X, model_predictions_i.X, sigma_i, eps_i, sqrt_2eps_i
        )

        corrected_composition_i = AXL(
            A=composition_i.A,
            X=corrected_x_i,
            L=composition_i.L,
        )

        self.sample_trajectory_recorder.record_corrector_step(
            i_index=index_i,
            time=t_i,
            sigma=sigma_i,
            composition_i=composition_i,
            corrected_composition_i=corrected_composition_i,
            model_predictions_i=model_predictions_i,
        )

        return corrected_composition_i
