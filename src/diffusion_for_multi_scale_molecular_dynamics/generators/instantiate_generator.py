from diffusion_for_multi_scale_molecular_dynamics.generators.langevin_generator import \
    LangevinGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.ode_position_generator import \
    ExplodingVarianceODEPositionGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.position_generator import \
    SamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.generators.sde_position_generator import \
    ExplodingVarianceSDEPositionGenerator
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.variance_sampler import \
    NoiseParameters


def instantiate_generator(
    sampling_parameters: SamplingParameters,
    noise_parameters: NoiseParameters,
    sigma_normalized_score_network: ScoreNetwork,
):
    """Instantiate generator."""
    assert sampling_parameters.algorithm in [
        "ode",
        "sde",
        "predictor_corrector",
    ], "Unknown algorithm. Possible choices are 'ode', 'sde' and 'predictor_corrector'"

    match sampling_parameters.algorithm:
        case "predictor_corrector":
            generator = LangevinGenerator(
                sampling_parameters=sampling_parameters,
                noise_parameters=noise_parameters,
                sigma_normalized_score_network=sigma_normalized_score_network,
            )
        case "ode":
            generator = ExplodingVarianceODEPositionGenerator(
                sampling_parameters=sampling_parameters,
                noise_parameters=noise_parameters,
                sigma_normalized_score_network=sigma_normalized_score_network,
            )
        case "sde":
            generator = ExplodingVarianceSDEPositionGenerator(
                sampling_parameters=sampling_parameters,
                noise_parameters=noise_parameters,
                sigma_normalized_score_network=sigma_normalized_score_network,
            )
        case _:
            raise NotImplementedError(
                f"algorithm '{sampling_parameters.algorithm}' is not implemented"
            )

    return generator
