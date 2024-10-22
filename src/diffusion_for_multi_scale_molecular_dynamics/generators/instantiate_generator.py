from crystal_diffusion.generators.langevin_generator import LangevinGenerator
from crystal_diffusion.generators.ode_position_generator import \
    ExplodingVarianceODEPositionGenerator
from crystal_diffusion.generators.sde_position_generator import \
    ExplodingVarianceSDEPositionGenerator
from crystal_diffusion.models.score_networks import ScoreNetwork
from src.crystal_diffusion.generators.position_generator import \
    SamplingParameters
from src.crystal_diffusion.samplers.variance_sampler import NoiseParameters


def instantiate_generator(sampling_parameters: SamplingParameters,
                          noise_parameters: NoiseParameters,
                          sigma_normalized_score_network: ScoreNetwork):
    """Instantiate generator."""
    assert sampling_parameters.algorithm in ['ode', 'sde', 'predictor_corrector'], \
        "Unknown algorithm. Possible choices are 'ode', 'sde' and 'predictor_corrector'"

    match sampling_parameters.algorithm:
        case 'predictor_corrector':
            generator = LangevinGenerator(sampling_parameters=sampling_parameters,
                                          noise_parameters=noise_parameters,
                                          sigma_normalized_score_network=sigma_normalized_score_network)
        case 'ode':
            generator = ExplodingVarianceODEPositionGenerator(
                sampling_parameters=sampling_parameters,
                noise_parameters=noise_parameters,
                sigma_normalized_score_network=sigma_normalized_score_network)
        case 'sde':
            generator = ExplodingVarianceSDEPositionGenerator(
                sampling_parameters=sampling_parameters,
                noise_parameters=noise_parameters,
                sigma_normalized_score_network=sigma_normalized_score_network)
        case _:
            raise NotImplementedError(f"algorithm '{sampling_parameters.algorithm}' is not implemented")

    return generator
