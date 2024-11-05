import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import (
    PredictorCorrectorAXLGenerator,
    PredictorCorrectorSamplingParameters,
)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network import (
    ScoreNetwork,
)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL,
    CARTESIAN_FORCES,
    NOISE,
    NOISY_AXL_COMPOSITION,
    TIME,
    UNIT_CELL,
)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import (
    NoiseParameters,
)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.variance_sampler import (
    NoiseScheduler,
)
from diffusion_for_multi_scale_molecular_dynamics.utils.d3pm_utils import compute_p_atm1_given_at
from diffusion_for_multi_scale_molecular_dynamics.utils.sample_trajectory import (
    NoOpPredictorCorrectorSampleTrajectory,
    PredictorCorrectorSampleTrajectory,
)


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

        if sampling_parameters.record_samples:
            self.sample_trajectory_recorder = PredictorCorrectorSampleTrajectory()
        else:
            self.sample_trajectory_recorder = NoOpPredictorCorrectorSampleTrajectory()

    def initialize(self, number_of_samples: int):
        """This method must initialize the samples from the fully noised distribution."""
        # all atoms are initialized as masked
        atom_types = torch.ones(number_of_samples, self.number_of_atoms).long() * (self.num_classes - 1)
        # relative coordinates are sampled from the uniform distribution
        relative_coordinates = torch.rand(
            number_of_samples, self.number_of_atoms, self.spatial_dimension
        )
        lattice_vectors = torch.zeros_like(relative_coordinates)  # TODO placeholder
        init_composition = AXL(
            A=atom_types,
            X=relative_coordinates,
            L=lattice_vectors
        )
        return init_composition

    def _draw_gaussian_sample(self, number_of_samples):
        return torch.randn(
            number_of_samples, self.number_of_atoms, self.spatial_dimension
        )

    def _draw_gumbel_sample(self, number_of_samples):
        return -torch.log(-torch.log(torch.rand(
            number_of_samples, self.number_of_atoms, self.num_classes
        )))

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
        sigma_noise_tensor = sigma_noise * torch.ones(number_of_samples, 1).to(composition.X)
        augmented_batch = {
            NOISY_AXL_COMPOSITION: composition,  # TODO
            TIME: time_tensor,
            NOISE: sigma_noise_tensor,
            UNIT_CELL: unit_cell,  # TODO replace with AXL-L
            CARTESIAN_FORCES: cartesian_forces,
        }

        # TODO do not hard-code conditional to False - need to be able to condition sampling
        model_predictions = self.axl_network(
            augmented_batch, conditional=False
        )
        return model_predictions

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

        number_of_samples = composition_i.X.shape[0]
        # gaussian sample noise
        z = self._draw_gaussian_sample(number_of_samples).to(composition_i.X)
        # uniform noise with gumbel sampling trick
        u = self._draw_gumbel_sample(number_of_samples).to(composition_i.X)

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
        one_step_transition_probs = compute_p_atm1_given_at(
            model_predictions_i.A,
            q_matrices_i,
            q_bar_matrices_i,
            q_bar_tm1_matrices_i
        )  # p(a_{t-1} | a_t) as a [num_samples, num_atoms, num_classes] tensor
        # sample new atom types from p(a_{t-1} | a_t) using the gumbel trick
        a_im1 = torch.argmax(torch.log(one_step_transition_probs + 1e-8) + u, dim=-1)
        # a_im1 has shape: number_of_samples, number_of_atoms and is a LongTensor

        x_i = composition_i.X  # reduced coordinates
        sigma_score_i = model_predictions_i.X  # sigma normalized score predicted by the model
        x_im1 = x_i + g2_i / sigma_i * sigma_score_i + g_i * z  # Langevin predictor step

        composition_im1 = AXL(
            A=a_im1,
            X=x_im1,
            L=composition_i.L  # TODO placeholder
        )

        self.sample_trajectory_recorder.record_unit_cell(unit_cell=unit_cell)  # TODO replace with AXL-L
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

        x_i = composition_i.X

        number_of_samples = x_i.shape[0]
        z = self._draw_gaussian_sample(number_of_samples).to(x_i)

        # The Langevin dynamics array are indexed with [0,..., N-1]
        eps_i = self.langevin_dynamics.epsilon[index_i].to(x_i)
        sqrt_2eps_i = self.langevin_dynamics.sqrt_2_epsilon[index_i].to(x_i)

        if index_i == 0:
            # TODO: we are extrapolating here; the score network will never have seen this time step...
            sigma_i = (
                self.noise_parameters.sigma_min
            )  # no need to change device, this is a float
            t_i = 0.0  # same for device - this is a float
        else:
            idx = index_i - 1  # python starts indices at zero
            sigma_i = self.noise.sigma[idx].to(x_i)
            t_i = self.noise.time[idx].to(x_i)

        model_predictions_i = self._get_model_predictions(
            composition_i, t_i, sigma_i, unit_cell, cartesian_forces
        )
        sigma_score_i = model_predictions_i.X

        corrected_x_i = x_i + eps_i / sigma_i * sigma_score_i + sqrt_2eps_i * z

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
