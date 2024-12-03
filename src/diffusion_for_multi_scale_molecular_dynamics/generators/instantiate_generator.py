from diffusion_for_multi_scale_molecular_dynamics.generators.adaptive_corrector import \
    AdaptiveCorrectorGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.axl_generator import \
    SamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.generators.langevin_generator import \
    LangevinGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.ode_position_generator import \
    ExplodingVarianceODEAXLGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.sde_position_generator import \
    ExplodingVarianceSDEPositionGenerator
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters


def instantiate_generator(
    sampling_parameters: SamplingParameters,
    noise_parameters: NoiseParameters,
    axl_network: ScoreNetwork,
):
    """Instantiate generator."""
    assert sampling_parameters.algorithm in [
        "ode",
        "sde",
        "predictor_corrector",
        "adaptive_corrector",
    ], "Unknown algorithm. Possible choices are 'ode', 'sde', 'predictor_corrector' and 'adaptive_corrector'"

    match sampling_parameters.algorithm:
        case "predictor_corrector":
            generator = LangevinGenerator(
                sampling_parameters=sampling_parameters,
                noise_parameters=noise_parameters,
                axl_network=axl_network,
            )
        case "adaptive_corrector":
            generator = AdaptiveCorrectorGenerator(
                sampling_parameters=sampling_parameters,
                noise_parameters=noise_parameters,
                axl_network=axl_network,
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
