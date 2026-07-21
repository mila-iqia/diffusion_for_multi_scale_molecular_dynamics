from dataclasses import dataclass


@dataclass
class NoiseParameters:
    """Noise schedule parameters."""
    total_time_steps: int

    # the kind of schedule that is created
    schedule_type: str = "exponential"

    time_delta: float = 1e-5  # the time schedule will cover the range [time_delta, 1]
    # As discussed in Appendix C of "SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS",
    # the time t = 0 is problematic.

    # sigma_min_cart and sigma_max_cart are in Angstroms (Cartesian units).
    # NoisingTransform and generators convert to relative coordinate units per sample using the cell dimensions,
    # so that the noise level is physically meaningful and size-independent.
    sigma_min_cart: float = 1e-4
    sigma_max_cart: float = 5.0

    # Default value comes from "Generative Modeling by Estimating Gradients of the Data Distribution"
    corrector_step_epsilon: float = 2e-5

    # Step size scaling for the Adaptive Corrector Generator. Default value comes from github implementation
    # https: // github.com / yang - song / score_sde / blob / main / configs / default_celeba_configs.py
    # for the celeba dataset. Note the suggested value for CIFAR10 is 0.16 in that repo.
    corrector_r: float = 0.17

    def __post_init__(self):
        """Check parameters."""
        assert self.schedule_type in ["exponential", "linear"], (
            f"The schedule type {self.schedule_type} is not supported.")
