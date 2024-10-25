from collections import namedtuple
from dataclasses import dataclass
from typing import Tuple

import torch

Noise = namedtuple("Noise", ["time", "sigma", "sigma_squared", "g", "g_squared", "beta",
                             "alpha_bar", "q_matrix", "q_bar_matrix", "q_bar_tm1_matrix"])
LangevinDynamics = namedtuple("LangevinDynamics", ["epsilon", "sqrt_2_epsilon"])


@dataclass
class NoiseParameters:
    """Noise schedule parameters."""

    total_time_steps: int
    time_delta: float = 1e-5  # the time schedule will cover the range [time_delta, 1]
    # As discussed in Appendix C of "SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS",
    # the time t = 0 is problematic.

    # Default values come from the paper:
    #   "Torsional Diffusion for Molecular Conformer Generation",
    # The original values in the paper are
    #   sigma_min = 0.01 pi , sigma_max = pi
    # However, they consider angles from 0 to 2pi as their coordinates:
    # here we divide by 2pi because our space is in the range [0, 1).
    sigma_min: float = 0.005
    sigma_max: float = 0.5

    # Default value comes from "Generative Modeling by Estimating Gradients of the Data Distribution"
    corrector_step_epsilon: float = 2e-5

    # Number of classes for the D3PM transition matrices
    num_classes: int = 3


class NoiseScheduler(torch.nn.Module):
    """Noise Scheduler.

    This class is responsible for creating all the quantities needed
    for noise generation for training and sampling.

    This implementation will use "exponential diffusion" and a "variance-preserving" diffusion as discussed in
    the following papers (no one paper presents everything clearly)
        - [1] "Torsional Diffusion for Molecular Conformer Generation".
        - [2] "SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS"
        - [3] "Generative Modeling by Estimating Gradients of the Data Distribution"
        - [4] "Denoising diffusion probabilistic models"
        - [5] "Deep unsupervised learning using nonequilibrium thermodynamics"

    The following quantities are defined:
        - total number of times steps, N

        - time steps:
            t in [delta, 1], on a discretized grid, t_i for i = 1, ..., N.
            We avoid t = 0 because sigma(t) is poorly defined there. See Appendix C of [2].

        - sigma and sigma^2:
            standard deviation, following the "exploding variance scheme",
                sigma(t) = sigma_min^(1-t) sigma_max^t.
                sigma_i for i = 1, ..., N.

        - g and g^2:
            g is the diffusion coefficient that appears in the stochastic differential equation (SDE).
                g^2(t) = d sigma^2(t)/ dt for the exploding variance scheme. This becomes discretized as
                g^2_i = sigma^2_{i} - sigma^2_{i-1} for i = 1, ..., N.

                --> The papers never clearly state what to do for sigma_{i=0}.
                    We CHOOSE sigma_{i=0} = sigma_min = sigma(t=0)

        - eps and sqrt_2_eps:
            This is for Langevin dynamics within a corrector step. Following [3], we define

                    eps_i = 0.5 epsilon_step * sigma^2_i / sigma^2_1 for i = 0, ..., N-1.

                --> Careful! eps_0 is needed for the corrector steps.

        - beta and alpha_bar:
            noise schedule following the "variance-preserving scheme",
                beta(t) = 1 / (t_{max} - t + 1)
                \bar{\alpha}(t) = \prod_{i=t}^t (1 - beta(i))

        - q_matrix, q_bar_matrix:
            transition matrix for D3PM - Q_t - and cumulative transition matrix \bar{Q}_t
            Q_t = (1 - beta(t)) I + beta(t) 1 e^T_m
            \bar{Q}_t = \prod_{i=i}^t Q_t
    """

    def __init__(self, noise_parameters: NoiseParameters, num_classes: int):
        """Init method.

        Args:
            noise_parameters: parameters that define the noise schedule.
            num_classes: number of discrete classes for the discrete diffusion
        """
        super().__init__()
        self.noise_parameters = noise_parameters
        self.num_classes = num_classes

        self._time_array = torch.nn.Parameter(
            self._get_time_array(noise_parameters), requires_grad=False
        )

        self._sigma_array = torch.nn.Parameter(
            self._create_sigma_array(noise_parameters, self._time_array),
            requires_grad=False,
        )
        self._sigma_squared_array = torch.nn.Parameter(
            self._sigma_array**2, requires_grad=False
        )

        self._g_squared_array = torch.nn.Parameter(
            self._create_g_squared_array(noise_parameters, self._sigma_squared_array),
            requires_grad=False,
        )
        self._g_array = torch.nn.Parameter(
            torch.sqrt(self._g_squared_array), requires_grad=False
        )

        self._epsilon_array = torch.nn.Parameter(
            self._create_epsilon_array(noise_parameters, self._sigma_squared_array),
            requires_grad=False,
        )
        self._sqrt_two_epsilon_array = torch.nn.Parameter(
            torch.sqrt(2.0 * self._epsilon_array), requires_grad=False
        )

        self._maximum_random_index = torch.nn.Parameter(
            torch.tensor(noise_parameters.total_time_steps - 1), requires_grad=False
        )
        self._minimum_random_index = torch.nn.Parameter(
            torch.tensor(0), requires_grad=False
        )

        self._beta_array = torch.nn.Parameter(
            self._create_beta_array(noise_parameters.total_time_steps), requires_grad=False
        )

        self._alpha_bar_array = torch.nn.Parameter(
            self._create_bar_alpha_array(self._beta_array)
        )

        self._q_matrix_array = torch.nn.Parameter(
            self._create_q_matrix_array(self._beta_array, num_classes), requires_grad=False
        )

        self._q_bar_matrix_array = torch.nn.Parameter(
            self._create_q_bar_matrix_array(self._q_matrix_array), requires_grad=False
        )

    @staticmethod
    def _get_time_array(noise_parameters: NoiseParameters) -> torch.Tensor:
        return torch.linspace(
            noise_parameters.time_delta, 1.0, noise_parameters.total_time_steps
        )

    @staticmethod
    def _create_sigma_array(
        noise_parameters: NoiseParameters, time_array: torch.Tensor
    ) -> torch.Tensor:
        sigma_min = noise_parameters.sigma_min
        sigma_max = noise_parameters.sigma_max

        sigma = sigma_min ** (1.0 - time_array) * sigma_max**time_array
        return sigma

    @staticmethod
    def _create_g_squared_array(
        noise_parameters: NoiseParameters, sigma_squared_array: torch.Tensor
    ) -> torch.Tensor:
        # g^2_{i} = sigma^2_{i} - sigma^2_{i-1}. For the first element (i=1), we set sigma_{0} = sigma_min.
        sigma_min = noise_parameters.sigma_min
        zeroth_value_tensor = torch.tensor([sigma_squared_array[0] - sigma_min**2])
        return torch.cat(
            [zeroth_value_tensor, sigma_squared_array[1:] - sigma_squared_array[:-1]]
        )

    @staticmethod
    def _create_epsilon_array(
        noise_parameters: NoiseParameters, sigma_squared_array: torch.Tensor
    ) -> torch.Tensor:

        sigma_squared_0 = noise_parameters.sigma_min**2
        sigma_squared_1 = sigma_squared_array[0]
        eps = noise_parameters.corrector_step_epsilon

        zeroth_value_tensor = torch.tensor(
            [0.5 * eps * sigma_squared_0 / sigma_squared_1]
        )
        return torch.cat(
            [
                zeroth_value_tensor,
                0.5 * eps * sigma_squared_array[:-1] / sigma_squared_1,
            ]
        )

    @staticmethod
    def _create_beta_array(num_time_steps: int) -> torch.Tensor:
        return 1.0 / (num_time_steps - torch.arange(1, num_time_steps + 1) + 1)

    @staticmethod
    def _create_alpha_bar_array(
        beta_array: torch.Tensor
    ) -> torch.Tensor:
        return torch.cumprod(1 - beta_array, 0)

    @staticmethod
    def _create_q_matrix_array(
        beta_array: torch.Tensor,
        num_classes: torch.Tensor
    ) -> torch.Tensor:
        beta_array_ = beta_array.unsqueeze(-1).unsqueeze(-1)
        qt = beta_array_ * torch.eye(num_classes)  # time step, num_classes, num_classes
        qt += (1 - beta_array_) * torch.outer(
            torch.ones(num_classes),
            torch.nn.functional.one_hot(torch.LongTensor([num_classes - 1]), num_classes=num_classes)
        )
        return qt

    @staticmethod
    def _create_q_bar_matrix_array(
        q_matrix_array: torch.Tensor
    ) -> torch.Tensor:
        q_bar_matrix_array = torch.empty_like(q_matrix_array)
        q_bar_matrix_array[0] = q_matrix_array[0]
        for i in range(1, q_matrix_array.size(0)):
            q_bar_matrix_array[i] = torch.matmul(q_bar_matrix_array[i - 1], q_matrix_array[i])
        return q_bar_matrix_array

    def _get_random_time_step_indices(self, shape: Tuple[int]) -> torch.Tensor:
        """Random time step indices.

        Generate random indices that correspond to valid time steps.

        Args:
            shape: shape of the random index array.

        Returns:
            time_step_indices: random time step indices in a tensor of shape "shape".
        """
        random_indices = torch.randint(
            self._minimum_random_index,
            self._maximum_random_index
            + 1,  # +1 because the maximum value is not sampled
            size=shape,
            device=self._minimum_random_index.device,
        )
        return random_indices

    def get_random_noise_sample(self, batch_size: int) -> Noise:
        """Get random noise sample.

        It is assumed that a batch is of the form [batch_size, (dimensions of a configuration)].
        In order to train a diffusion model, a configuration must be "noised" to a time t with a parameter sigma(t).
        Different values can be used for different configurations: correspondingly, this method returns
        one random time per element in the batch.


        Args:
            batch_size : number of configurations in a batch,

        Returns:
            noise_sample: a collection of all the noise parameters (t, sigma, sigma^2, g, g^2)
                for some random indices. All the arrays are of dimension [batch_size].
        """
        indices = self._get_random_time_step_indices((batch_size,))
        times = self._time_array.take(indices)
        sigmas = self._sigma_array.take(indices)
        sigmas_squared = self._sigma_squared_array.take(indices)
        gs = self._g_array.take(indices)
        gs_squared = self._g_squared_array.take(indices)
        betas = self._beta_array(indices)
        alpha_bars = self._alpha_bar_array(indices)
        q_matrices = self._q_matrix_array(indices)
        q_bar_matrices = self._q_bar_matrix_array(indices)
        # we also need the q_bar matrices for the previous time index (t-1) to compute the loss. We will use Q_{t-1}=1
        # for the case t=1 (special case in the loss or the last step of the sampling process
        q_bar_tm1_matrices = torch.where(
            indices.view(-1, 1, 1) == 0,  # condition
            torch.eye(self.num_classes).unsqueeze(-1),  # replace t=0 with identity matrix
            self._q_bar_matrix_array((indices - 1).clip(min=0))  # \bar{Q}_{t-1} otherwise
        )

        return Noise(
            time=times,
            sigma=sigmas,
            sigma_squared=sigmas_squared,
            g=gs,
            g_squared=gs_squared,
            beta=betas,
            alpha_bar=alpha_bars,
            q_matrix=q_matrices,
            q_bar_matrix=q_bar_matrices,
            q_bar_tm1_matrix=q_bar_tm1_matrices
        )

    def get_all_sampling_parameters(self) -> Tuple[Noise, LangevinDynamics]:
        """Get all sampling parameters.

        All the internal noise parameter arrays and Langevin dynamics arrays.

        Returns:
            all_noise: a collection of all the noise parameters (t, sigma, sigma^2, g, g^2)
                for all indices. The arrays are all of dimension [total_time_steps].
            langevin_dynamics: a collection of all the langevin dynamics parmaters (epsilon, sqrt{2epsilon})
                needed to apply a langevin dynamics corrector step.
        """
        q_bar_tm1_matrices = torch.cat(
            (torch.eye(self.num_classes).unsqueeze(0), self._q_bar_matrix_array[:-1]), dim=0)
        noise = Noise(
            time=self._time_array,
            sigma=self._sigma_array,
            sigma_squared=self._sigma_squared_array,
            g=self._g_array,
            g_squared=self._g_squared_array,
            beta=self._beta_array,
            alpha_bar=self._alpha_bar_array,
            q_matrix=self._q_matrix_array,
            q_bar_matrix=self._q_bar_matrix_array,
            q_bar_tm1_matrices=q_bar_tm1_matrices
        )
        langevin_dynamics = LangevinDynamics(
            epsilon=self._epsilon_array, sqrt_2_epsilon=self._sqrt_two_epsilon_array
        )

        return noise, langevin_dynamics
