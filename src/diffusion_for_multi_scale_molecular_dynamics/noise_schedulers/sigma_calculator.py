import torch
from torch import nn


def instantiate_sigma_calculator(sigma_min: float, sigma_max: float, schedule_type: str, **kwargs):
    """Instantiate a sigma calculator bvased on the schedule type."""
    match schedule_type:
        case "exponential":
            return ExponentialSigmaCalculator(sigma_min, sigma_max)
        case "linear":
            return LinearSigmaCalculator(sigma_min, sigma_max)
        case "double_linear":
            return DoubleLinearSigmaCalculator(sigma_min, sigma_max, **kwargs)
        case _:
            raise NotImplementedError(f"The schedule type {schedule_type} is not implemented")


class SigmaCalculator(nn.Module):
    """Sigma Calculator Base Class."""

    def __init__(self, sigma_min: float, sigma_max: float, **kwargs):
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


class DoubleLinearSigmaCalculator(SigmaCalculator):
    """Double Linear Sigma Calculator.

    The schedule is given by

        sigma(t) = sigma_min + (sigma_max - sigma_min) * t

    if sigma_critical <= sigma(t) >= sigma_critical_max

    """

    def __init__(self,
                 sigma_min: float,
                 sigma_max: float,
                 sigma_critical_min: float,
                 sigma_critical_max: float,
                 fraction_in_critical_region : float = 0.5):
        """Init method.

        Args:
            sigma_min: minimum value of sigma
            sigma_max: maximum value of sigma
        """
        super().__init__(sigma_min, sigma_max)
        assert sigma_critical_min > sigma_min, "sigma_critical_min is expected to be larger than sigma_min"
        assert sigma_critical_min < sigma_critical_max,\
            "sigma_critical_min is expected to be larger than sigma_critical_max"
        assert sigma_critical_max <= sigma_max, "sigma_critical_max is expected to be smaller or equal to sigma_max"
        fraction_in_critical_region = torch.tensor(fraction_in_critical_region)
        self.low_sigma_transition_time = torch.nn.Parameter((1 - fraction_in_critical_region) / 2, requires_grad=False)
        self.high_sigma_transition_time = torch.nn.Parameter(
            self.low_sigma_transition_time + fraction_in_critical_region, requires_grad=False
        )
        self.sigma_critical_min = torch.nn.Parameter(torch.tensor(sigma_critical_min), requires_grad=False)
        self.sigma_critical_max = torch.nn.Parameter(torch.tensor(sigma_critical_max), requires_grad=False)
        self.low_sigma_slope = torch.nn.Parameter(
            (self.sigma_critical_min - self.sigma_min) / self.low_sigma_transition_time,
            requires_grad=False
        )
        self.transition_slope = torch.nn.Parameter(
            (self.sigma_critical_max - self.sigma_critical_min) / fraction_in_critical_region,
            requires_grad=False
        )

        self.high_sigma_slope = torch.nn.Parameter(
            (self.sigma_max - self.sigma_critical_max) / (1 - self.high_sigma_transition_time),
            requires_grad=False
        )

    def get_sigma(self, times: torch.Tensor) -> torch.Tensor:
        """Get sigma."""
        sigma_low = torch.where(
            times < self.low_sigma_transition_time,
            times * self.low_sigma_slope,
            torch.zeros_like(times)
        )
        sigma_transition = torch.where(
            (times <= self.high_sigma_transition_time) & (times >= self.low_sigma_transition_time),
            (times - self.low_sigma_transition_time) * self.transition_slope + self.sigma_critical_min,
            torch.zeros_like(times)
        )
        sigma_high = torch.where(
            times > self.high_sigma_transition_time,
            (times - self.high_sigma_transition_time) * self.high_sigma_slope + self.sigma_critical_max,
            torch.zeros_like(times)
        )

        return sigma_low + sigma_transition + sigma_high

    def get_sigma_time_derivative(self, times: torch.Tensor) -> torch.Tensor:
        """Get sigma time derivative."""
        sigma_low = torch.where(
            times < self.low_sigma_transition_time,
            self.low_sigma_slope,
            torch.zeros_like(times)
        )
        sigma_transition = torch.where(
            (times <= self.high_sigma_transition_time) & (times >= self.low_sigma_transition_time),
            self.transition_slope,
            torch.zeros_like(times)
        )
        sigma_high = torch.where(
            times > self.high_sigma_transition_time,
            self.high_sigma_slope,
            torch.zeros_like(times)
        )

        return sigma_low + sigma_transition + sigma_high
