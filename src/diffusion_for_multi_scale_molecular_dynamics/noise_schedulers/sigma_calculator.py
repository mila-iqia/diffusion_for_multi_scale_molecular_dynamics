import torch
from torch import nn


def instantiate_sigma_calculator(sigma_min: float, sigma_max: float, schedule_type: str):
    """Instantiate a sigma calculator bvased on the schedule type."""
    match schedule_type:
        case "exponential":
            return ExponentialSigmaCalculator(sigma_min, sigma_max)
        case "linear":
            return LinearSigmaCalculator(sigma_min, sigma_max)
        case _:
            raise NotImplementedError(f"The schedule type {schedule_type} is not implemented")


class SigmaCalculator(nn.Module):
    """Sigma Calculator Base Class."""

    def __init__(self, sigma_min: float, sigma_max: float):
        """Init method.

        Args:
            sigma_min: minimum value of sigma
            sigma_max: maximum value of sigma
        """
        super().__init__()

        self.sigma_min = torch.nn.Parameter(
            torch.tensor(sigma_min), requires_grad=False
        )
        self.sigma_max = torch.nn.Parameter(
            torch.tensor(sigma_max), requires_grad=False
        )

    def get_sigma(self, times: torch.Tensor) -> torch.Tensor:
        """Get sigma."""
        raise NotImplementedError("This method must be implemented in a child class.")

    def get_sigma_time_derivative(self, times: torch.Tensor) -> torch.Tensor:
        """Get sigma time derivative."""
        raise NotImplementedError("This method must be implemented in a child class.")

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        return self.get_sigma(times)


class ExponentialSigmaCalculator(SigmaCalculator):
    """Exponential Sigma Calculator.

    The schedule is given by

        sigma(t) = sigma_min * (sigma_max / sigma_min)^t

    """

    def __init__(self, sigma_min: float, sigma_max: float):
        """Init method.

        Args:
            sigma_min: minimum value of sigma
            sigma_max: maximum value of sigma
        """
        super().__init__(sigma_min, sigma_max)
        self.ratio = torch.nn.Parameter(
            self.sigma_max / self.sigma_min, requires_grad=False
        )
        self.log_ratio = torch.nn.Parameter(
            torch.log(self.sigma_max / self.sigma_min), requires_grad=False
        )

    def get_sigma(self, times: torch.Tensor) -> torch.Tensor:
        """Get sigma."""
        return self.sigma_min * self.ratio**times

    def get_sigma_time_derivative(self, times: torch.Tensor) -> torch.Tensor:
        """Get sigma time derivative."""
        return self.log_ratio * self.get_sigma(times)


class LinearSigmaCalculator(SigmaCalculator):
    """Linear Sigma Calculator.

    The schedule is given by

        sigma(t) = sigma_min + (sigma_max - sigma_min) * t

    """

    def __init__(self, sigma_min: float, sigma_max: float):
        """Init method.

        Args:
            sigma_min: minimum value of sigma
            sigma_max: maximum value of sigma
        """
        super().__init__(sigma_min, sigma_max)
        self.sigma_difference = torch.nn.Parameter(
            self.sigma_max - self.sigma_min, requires_grad=False
        )

    def get_sigma(self, times: torch.Tensor) -> torch.Tensor:
        """Get sigma."""
        return self.sigma_min + self.sigma_difference * times

    def get_sigma_time_derivative(self, times: torch.Tensor) -> torch.Tensor:
        """Get sigma time derivative."""
        return self.sigma_difference * torch.ones_like(times)
