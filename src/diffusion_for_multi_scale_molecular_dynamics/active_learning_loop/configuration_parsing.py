from typing import Any, AnyStr, Dict, List, Optional, Tuple, Union

import torch

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.atom_selector.atom_selector_factory import \
    create_atom_selector_parameters
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.excisor_factory import \
    create_excisor_parameters
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.base_sample_maker import \
    BaseSampleMaker
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.sample_maker_factory import (
    create_sample_maker, create_sample_maker_parameters)
from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.sample_diffusion import \
    get_axl_network


def get_repaint_parameters(
    sampling_dictionary: Dict[AnyStr, Any],
    element_list: List[str],
    path_to_score_network_checkpoint: Optional[str] = None,
) -> Tuple[
    Union[NoiseParameters, None],
    Union[PredictorCorrectorSamplingParameters, None],
    Union[ScoreNetwork, None],
    str,
]:
    """Get repaint parameters.

    This convenience method is responsible for extracting the relevant configuration objects in the
    case that the sample maker algorithm is "Excise and Repaint", and to return a "None" default for
    these configuration objects if a different algorithm is used.

    Args:
        sampling_dictionary: Dictionary of sampling parameters, as read in from a yaml configuration file.
        element_list: List of element names.
        path_to_score_network_checkpoint: Path to score network checkpoint.

    Returns:
        noise_parameters: a NoiseParameters object if the config is present, otherwise None.
        sampling_parameters: a PredictorCorrectorSamplingParameters object if the config is present, otherwise None.
        axl_network: a Score Network object to draw constrained samples if the config is prseent, otherwise None.
        device: a string indicating which device should be used: either cpu or cuda.
    """
    algorithm = sampling_dictionary["algorithm"]
    # Default values
    device = "cpu"
    axl_network = None
    noise_parameters = None
    sampling_parameters = None
    if algorithm != "excise_and_repaint":
        return noise_parameters, sampling_parameters, axl_network, device

    if torch.cuda.is_available():
        device = "cuda"
    assert (
        path_to_score_network_checkpoint is not None
    ), "A path to a valid score network checkpoint must be provided to use 'excise_and_repaint'."
    axl_network = get_axl_network(path_to_score_network_checkpoint)

    assert (
        "noise" in sampling_dictionary
    ), "A 'noise' configuration must be defined in the 'sampling' field in order to use 'excise_and_repaint'."

    noise_dictionary = sampling_dictionary["noise"]
    noise_parameters = NoiseParameters(**noise_dictionary)

    assert "repaint_generator" in sampling_dictionary, (
        "A 'repaint_generator' configuration must be defined in the 'sampling' field in order to use "
        "'excise_and_repaint'."
    )

    sampling_generator_dictionary = sampling_dictionary["repaint_generator"]

    assert "algorithm" not in sampling_generator_dictionary, (
        "Do not specify the 'algorithm' for the repaint generator: only the predictor_corrector repaint generator "
        "algorithm is valid and will be automatically selected."
    )
    sampling_generator_dictionary["algorithm"] = "predictor_corrector"

    assert "num_atom_types" not in sampling_generator_dictionary, (
        "Do not specify the 'num_atom_types' for the repaint generator: the value will be inferred from "
        "the element list."
    )
    sampling_generator_dictionary["num_atom_types"] = len(element_list)

    assert "number_of_samples" not in sampling_generator_dictionary, (
        "Do not specify the 'number_of_samples' for the repaint generator: the value will be inferred from "
        "the 'number_of_samples_per_substructure' sampling field."
    )
    sampling_generator_dictionary["number_of_samples"] = sampling_dictionary.get(
        "number_of_samples_per_substructure", 1
    )

    assert (
        "use_fixed_lattice_parameters" not in sampling_generator_dictionary
        and "cell_dimensions" not in sampling_generator_dictionary
    ), (
        "Do not specify 'use_fixed_lattice_parameters' or 'cell_dimensions' for the repaint generator: these values "
        "will be inferred from the sampling field."
    )
    sampling_generator_dictionary["use_fixed_lattice_parameters"] = (
        sampling_dictionary.get("sample_box_strategy", "fixed")
    )

    if sampling_generator_dictionary["use_fixed_lattice_parameters"] == "fixed":
        sampling_generator_dictionary["cell_dimensions"] = sampling_dictionary[
            "sample_box_size"
        ]

    sampling_parameters = PredictorCorrectorSamplingParameters(
        **sampling_generator_dictionary
    )

    return noise_parameters, sampling_parameters, axl_network, device


def get_sample_maker_from_configuration(
    sampling_dictionary: Dict,
    uncertainty_threshold: float,
    element_list: List[str],
    path_to_score_network_checkpoint: Optional[str] = None,
) -> BaseSampleMaker:
    """Get sample maker from configuration.

    the sampling dictionary should have the following structure:

        sampling:
            algorithm: ...
            (other sample maker parameters)

            excision [Only if using Excise and *]:
                (excision parameters)

            noise [Only if using Excise and Repaint]:
                (noise parameters)

          repaint_generator [Only if using Excise and Repaint]:
                (constrained sampling parameters)

    Args:
        sampling_dictionary: Dictionary of sampling parameters, as read in from a yaml configuration file.
        uncertainty_threshold: Uncertainty threshold.
        element_list: List of element names.
        path_to_score_network_checkpoint: Path to score network checkpoint.

    Returns:
        sample_maker: A configured Sample Maker instance.
    """
    # Let's make sure we don't modify the input, which would lead to undesirable side effects!
    sampling_dict = sampling_dictionary.copy()

    noise_parameters, sampling_parameters, axl_network, device = get_repaint_parameters(
        sampling_dictionary=sampling_dict,
        element_list=element_list,
        path_to_score_network_checkpoint=path_to_score_network_checkpoint,
    )

    atom_selector_parameter_dictionary = dict(
        algorithm="threshold", uncertainty_threshold=uncertainty_threshold
    )
    atom_selector_parameters = create_atom_selector_parameters(
        atom_selector_parameter_dictionary
    )

    excisor_parameter_dictionary = sampling_dict.pop("excision", None)
    if excisor_parameter_dictionary is not None:
        excisor_parameters = create_excisor_parameters(excisor_parameter_dictionary)
    else:
        excisor_parameters = None

    # Let's extract only the sample_maker configuration, popping out components that don't belong.
    sample_maker_dictionary = sampling_dict.copy()
    sample_maker_dictionary["element_list"] = element_list
    sample_maker_dictionary.pop("noise", None)
    sample_maker_dictionary.pop("repaint_generator", None)

    sample_maker_parameters = create_sample_maker_parameters(sample_maker_dictionary)

    sample_maker = create_sample_maker(
        sample_maker_parameters=sample_maker_parameters,
        atom_selector_parameters=atom_selector_parameters,
        excisor_parameters=excisor_parameters,
        noise_parameters=noise_parameters,
        sampling_parameters=sampling_parameters,
        diffusion_model=axl_network,
        device=device,
    )
    return sample_maker
