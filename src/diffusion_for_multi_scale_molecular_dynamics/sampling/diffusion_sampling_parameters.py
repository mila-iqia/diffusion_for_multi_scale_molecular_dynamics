from dataclasses import dataclass
from typing import Any, AnyStr, Dict, Union

from diffusion_for_multi_scale_molecular_dynamics.generators.axl_generator import \
    SamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.generators.load_sampling_parameters import \
    load_sampling_parameters
from diffusion_for_multi_scale_molecular_dynamics.metrics.sampling_metrics_parameters import \
    SamplingMetricsParameters
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters


@dataclass(kw_only=True)
class DiffusionSamplingParameters:
    """Diffusion sampling parameters.

    This dataclass holds various configuration objects that define how
    samples should be generated and evaluated (ie, metrics) during training.
    """

    sampling_parameters: (
        SamplingParameters  # Define the algorithm and parameters to draw samples.
    )
    noise_parameters: (
        NoiseParameters  # Noise for sampling, which can be different from training!
    )
    metrics_parameters: (
        SamplingMetricsParameters  # what should be done with the generated samples?
    )


def load_diffusion_sampling_parameters(
    hyper_params: Dict[AnyStr, Any]
) -> Union[DiffusionSamplingParameters, None]:
    """Load diffusion sampling parameters.

    Extract the needed information from the configuration dictionary.

    Args:
        hyper_params: dictionary of hyperparameters loaded from a config file

    Returns:
        diffusion_sampling_parameters: the relevant configuration object.
    """
    if "diffusion_sampling" not in hyper_params:
        return None

    diffusion_sampling_dict = hyper_params["diffusion_sampling"]

    assert (
        "sampling" in diffusion_sampling_dict
    ), "The sampling parameters must be defined to draw samples."
    sampling_parameters = load_sampling_parameters(diffusion_sampling_dict["sampling"])

    assert (
        "noise" in diffusion_sampling_dict
    ), "The noise parameters must be defined to draw samples."
    noise_parameters = NoiseParameters(**diffusion_sampling_dict["noise"])

    assert (
        "metrics" in diffusion_sampling_dict
    ), "The metrics parameters must be defined to draw samples."
    metrics_parameters = SamplingMetricsParameters(**diffusion_sampling_dict["metrics"])

    return DiffusionSamplingParameters(
        sampling_parameters=sampling_parameters,
        noise_parameters=noise_parameters,
        metrics_parameters=metrics_parameters,
    )
