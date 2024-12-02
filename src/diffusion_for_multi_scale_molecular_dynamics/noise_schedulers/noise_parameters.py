from dataclasses import dataclass


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
    #   sigma_min = 0.01 pi , sigma_σmax = pi
    # However, they consider angles from 0 to 2pi as their coordinates:
    # here we divide by 2pi because our space is in the range [0, 1).
    sigma_min: float = 0.005
    sigma_max: float = 0.5

    # Default value comes from "Generative Modeling by Estimating Gradients of the Data Distribution"
    corrector_step_epsilon: float = 2e-5

    # Step size scaling for the Adaptative Corrector Generator. Default value comes from github implementation
    # https: // github.com / yang - song / score_sde / blob / main / configs / default_celeba_configs.py
    # for the celeba dataset. Note the suggested value for CIFAR10 is 0.16 in that repo.
    corrector_r: float = 0.17
