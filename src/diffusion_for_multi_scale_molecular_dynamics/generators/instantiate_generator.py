from typing import Optional

from diffusion_for_multi_scale_molecular_dynamics.generators.adaptive_corrector import \
    AdaptiveCorrectorGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.axl_generator import \
    SamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.generators.constrained_langevin_generator import \
    ConstrainedLangevinGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.langevin_generator import \
    LangevinGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.ode_position_generator import \
    ExplodingVarianceODEAXLGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.sampling_constraint import \
    SamplingConstraint
from diffusion_for_multi_scale_molecular_dynamics.generators.sde_position_generator import \
    ExplodingVarianceSDEPositionGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.trajectory_initializer import \
    TrajectoryInitializer
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters


def instantiate_generator(
    sampling_parameters: SamplingParameters,
    noise_parameters: NoiseParameters,
    axl_network: ScoreNetwork,
    trajectory_initializer: TrajectoryInitializer,
    sampling_constraints: Optional[SamplingConstraint]
):
    """Instantiate generator."""
    assert sampling_parameters.algorithm in [
        "ode",
        "sde",
        "predictor_corrector",
        "adaptive_corrector",
    ], "Unknown algorithm. Possible choices are 'ode', 'sde', 'predictor_corrector' and 'adaptive_corrector'"

    if sampling_constraints is not None:
        assert sampling_parameters.algorithm == "predictor_corrector", \
            "Only the 'predictor_corrector' scheme supports sampling constraints."
        generator = ConstrainedLangevinGenerator(noise_parameters=noise_parameters,
                                                 sampling_parameters=sampling_parameters,
                                                 axl_network=axl_network,
                                                 sampling_constraints=sampling_constraints,
                                                 trajectory_initializer=trajectory_initializer)
        return generator

    match sampling_parameters.algorithm:
        case "predictor_corrector":
            generator = LangevinGenerator(
                sampling_parameters=sampling_parameters,
                noise_parameters=noise_parameters,
                axl_network=axl_network,
                trajectory_initializer=trajectory_initializer,
            )
        case "adaptive_corrector":
            generator = AdaptiveCorrectorGenerator(
                sampling_parameters=sampling_parameters,
                noise_parameters=noise_parameters,
                axl_network=axl_network,
                trajectory_initializer=trajectory_initializer,
            )
        case "ode":
            generator = ExplodingVarianceODEAXLGenerator(
                sampling_parameters=sampling_parameters,
                noise_parameters=noise_parameters,
                axl_network=axl_network,
            )
        case "sde":
            generator = ExplodingVarianceSDEPositionGenerator(
                sampling_parameters=sampling_parameters,
                noise_parameters=noise_parameters,
                axl_network=axl_network,
            )
        case _:
            raise NotImplementedError(
                f"algorithm '{sampling_parameters.algorithm}' is not implemented"
            )

    return generator
