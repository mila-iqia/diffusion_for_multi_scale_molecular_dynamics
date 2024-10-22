import torch
from crystal_diffusion.generators.predictor_corrector_position_generator import (
    PredictorCorrectorPositionGenerator, PredictorCorrectorSamplingParameters)
from crystal_diffusion.models.score_networks.score_network import ScoreNetwork
from crystal_diffusion.namespace import (CARTESIAN_FORCES, NOISE,
                                         NOISY_RELATIVE_COORDINATES, TIME,
                                         UNIT_CELL)
from crystal_diffusion.utils.sample_trajectory import (
    NoOpPredictorCorrectorSampleTrajectory, PredictorCorrectorSampleTrajectory)
from src.crystal_diffusion.samplers.variance_sampler import (
    ExplodingVarianceSampler, NoiseParameters)


class LangevinGenerator(PredictorCorrectorPositionGenerator):
    """Annealed Langevin Dynamics Generator.

    This class implements the annealed Langevin Dynamics generation of position samples, following
    Song & Ermon 2019, namely:
        "Generative Modeling by Estimating Gradients of the Data Distribution"
    """

    def __init__(self,
                 noise_parameters: NoiseParameters,
                 sampling_parameters: PredictorCorrectorSamplingParameters,
                 sigma_normalized_score_network: ScoreNetwork,
                 ):
        """Init method."""
        super().__init__(number_of_discretization_steps=noise_parameters.total_time_steps,
                         number_of_corrector_steps=sampling_parameters.number_of_corrector_steps,
                         spatial_dimension=sampling_parameters.spatial_dimension)

        self.noise_parameters = noise_parameters
        sampler = ExplodingVarianceSampler(noise_parameters)
        self.noise, self.langevin_dynamics = sampler.get_all_sampling_parameters()
        self.number_of_atoms = sampling_parameters.number_of_atoms
        self.sigma_normalized_score_network = sigma_normalized_score_network

        if sampling_parameters.record_samples:
            self.sample_trajectory_recorder = PredictorCorrectorSampleTrajectory()
        else:
            self.sample_trajectory_recorder = NoOpPredictorCorrectorSampleTrajectory()

    def initialize(self, number_of_samples: int):
        """This method must initialize the samples from the fully noised distribution."""
        relative_coordinates = torch.rand(number_of_samples, self.number_of_atoms, self.spatial_dimension)
        return relative_coordinates

    def _draw_gaussian_sample(self, number_of_samples):
        return torch.randn(number_of_samples, self.number_of_atoms, self.spatial_dimension)

    def _get_sigma_normalized_scores(self, x: torch.Tensor, time: float,
                                     noise: float, unit_cell: torch.Tensor, cartesian_forces: torch.Tensor
                                     ) -> torch.Tensor:
        """Get sigma normalized scores.

        Args:
            x : relative coordinates, of shape [number_of_samples, number_of_atoms, spatial_dimension]
            time : time at which to evaluate the score
            noise: the diffusion sigma parameter corresponding to the time at which to evaluate the score
            unit_cell: unit cell definition in Angstrom of shape [number_of_samples, spatial_dimension,
                spatial_dimension]
            cartesian_forces: forces to condition the sampling from. Shape [number_of_samples, number_of_atoms,
                spatial_dimension]

        Returns:
            sigma normalized score: sigma x Score(x, t).
        """
        number_of_samples = x.shape[0]

        time_tensor = time * torch.ones(number_of_samples, 1).to(x)
        noise_tensor = noise * torch.ones(number_of_samples, 1).to(x)
        augmented_batch = {NOISY_RELATIVE_COORDINATES: x, TIME: time_tensor, NOISE: noise_tensor, UNIT_CELL: unit_cell,
                           CARTESIAN_FORCES: cartesian_forces}

        # TODO do not hard-code conditional to False - need to be able to condition sampling
        predicted_normalized_scores = self.sigma_normalized_score_network(augmented_batch, conditional=False)
        return predicted_normalized_scores

    def predictor_step(self, x_i: torch.Tensor, index_i: int, unit_cell: torch.Tensor, cartesian_forces: torch.Tensor
                       ) -> torch.Tensor:
        """Predictor step.

        Args:
            x_i : sampled relative coordinates, at time step i.
            index_i : index of the time step.
            unit_cell: sampled unit cell at time step i.
            cartesian_forces: forces conditioning the sampling process

        Returns:
            x_im1 : sampled relative coordinates, at time step i - 1.
        """
        assert 1 <= index_i <= self.number_of_discretization_steps, \
            "The predictor step can only be invoked for index_i between 1 and the total number of discretization steps."

        number_of_samples = x_i.shape[0]
        z = self._draw_gaussian_sample(number_of_samples).to(x_i)

        idx = index_i - 1  # python starts indices at zero
        t_i = self.noise.time[idx].to(x_i)
        g_i = self.noise.g[idx].to(x_i)
        g2_i = self.noise.g_squared[idx].to(x_i)
        sigma_i = self.noise.sigma[idx].to(x_i)
        sigma_score_i = self._get_sigma_normalized_scores(x_i, t_i, sigma_i, unit_cell, cartesian_forces)
        x_im1 = x_i + g2_i / sigma_i * sigma_score_i + g_i * z

        self.sample_trajectory_recorder.record_unit_cell(unit_cell=unit_cell)
        self.sample_trajectory_recorder.record_predictor_step(i_index=index_i, time=t_i, sigma=sigma_i,
                                                              x_i=x_i, x_im1=x_im1, scores=sigma_score_i)

        return x_im1

    def corrector_step(self, x_i: torch.Tensor, index_i: int, unit_cell: torch.Tensor, cartesian_forces: torch.Tensor
                       ) -> torch.Tensor:
        """Corrector Step.

        Args:
            x_i : sampled relative coordinates, at time step i.
            index_i : index of the time step.
            unit_cell: sampled unit cell at time step i.
            cartesian_forces: forces conditioning the sampling

        Returns:
            corrected x_i : sampled relative coordinates, after corrector step.
        """
        assert 0 <= index_i <= self.number_of_discretization_steps - 1, \
            ("The corrector step can only be invoked for index_i between 0 and "
             "the total number of discretization steps minus 1.")

        number_of_samples = x_i.shape[0]
        z = self._draw_gaussian_sample(number_of_samples).to(x_i)

        # The Langevin dynamics array are indexed with [0,..., N-1]
        eps_i = self.langevin_dynamics.epsilon[index_i].to(x_i)
        sqrt_2eps_i = self.langevin_dynamics.sqrt_2_epsilon[index_i].to(x_i)

        if index_i == 0:
            # TODO: we are extrapolating here; the score network will never have seen this time step...
            sigma_i = self.noise_parameters.sigma_min  # no need to change device, this is a float
            t_i = 0.  # same for device - this is a float
        else:
            idx = index_i - 1  # python starts indices at zero
            sigma_i = self.noise.sigma[idx].to(x_i)
            t_i = self.noise.time[idx].to(x_i)

        sigma_score_i = self._get_sigma_normalized_scores(x_i, t_i, sigma_i, unit_cell, cartesian_forces)

        corrected_x_i = x_i + eps_i / sigma_i * sigma_score_i + sqrt_2eps_i * z

        self.sample_trajectory_recorder.record_corrector_step(i_index=index_i, time=t_i,
                                                              sigma=sigma_i, x_i=x_i, corrected_x_i=corrected_x_i,
                                                              scores=sigma_score_i)

        return corrected_x_i
