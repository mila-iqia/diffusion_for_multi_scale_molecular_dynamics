import torch

from crystal_diffusion.samplers.variance_sampler import NoiseParameters


class ExplodingVariance(torch.nn.Module):
    """Exploding Variance.

    This class is responsible for calculating the various quantities related  to the diffusion variance.
    This implementation will use "exploding variance" scheme.
    """

    def __init__(self, noise_parameters: NoiseParameters):
        """Init method.

        Args:
            noise_parameters: parameters that define the noise schedule.
        """
        super(ExplodingVariance, self).__init__()

        self.sigma_min = torch.nn.Parameter(torch.tensor(noise_parameters.sigma_min))
        self.sigma_max = torch.nn.Parameter(torch.tensor(noise_parameters.sigma_max))

        self.ratio = self.sigma_max / self.sigma_min
        self.log_ratio = torch.log(self.sigma_max / self.sigma_min)

    def get_sigma(self, times: torch.Tensor) -> torch.Tensor:
        """Get sigma.

        Compute the exploding variance value of sigma(t).

        Args:
            times : diffusion times.

        Returns:
            sigmas: the standard deviation in the exploding variance scheme.
        """
        return self.sigma_min * self.ratio ** times

    def get_sigma_time_derivative(self, times: torch.Tensor) -> torch.Tensor:
        """Get sigma time derivative.

        Compute the analytical time derivative of sigma.

        Args:
            times : diffusion times.

        Returns:
            sigma_dot : time derivative of sigma(t).
        """
        return self.log_ratio * self.get_sigma(times)

    def get_g_squared(self, times: torch.Tensor) -> torch.Tensor:
        """Get g squared.

        Compute g(t)^2 = d sigma(t)^2 / dt

        Args:
            times : diffusion times.

        Returns:
            g_squared: g(t)^2
        """
        return 2.0 * self.get_sigma(times) * self.get_sigma_time_derivative(times)
