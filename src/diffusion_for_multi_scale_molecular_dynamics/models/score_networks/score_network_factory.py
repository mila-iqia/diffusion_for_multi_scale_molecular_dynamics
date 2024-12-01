import dataclasses
from typing import Any, AnyStr, Dict

from diffusion_for_multi_scale_molecular_dynamics.models.score_networks import (
    ScoreNetwork, ScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.diffusion_mace_score_network import (
    DiffusionMACEScoreNetwork, DiffusionMACEScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.egnn_score_network import (
    EGNNScoreNetwork, EGNNScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.mace_score_network import (
    MACEScoreNetwork, MACEScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.mlp_score_network import (
    MLPScoreNetwork, MLPScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_prediction_head import (
    MaceEquivariantScorePredictionHeadParameters,
    MaceMLPScorePredictionHeadParameters)
from diffusion_for_multi_scale_molecular_dynamics.utils.configuration_parsing import \
    create_parameters_from_configuration_dictionary

SCORE_NETWORKS_BY_ARCH = dict(
    mlp=MLPScoreNetwork,
    mace=MACEScoreNetwork,
    diffusion_mace=DiffusionMACEScoreNetwork,
    egnn=EGNNScoreNetwork,
)
SCORE_NETWORK_PARAMETERS_BY_ARCH = dict(
    mlp=MLPScoreNetworkParameters,
    mace=MACEScoreNetworkParameters,
    diffusion_mace=DiffusionMACEScoreNetworkParameters,
    egnn=EGNNScoreNetworkParameters,
)
MACE_PREDICTION_HEAD_BY_NAME = dict(
    mlp=MaceMLPScorePredictionHeadParameters,
    equivariant=MaceEquivariantScorePredictionHeadParameters,
)


def create_score_network(score_network_parameters: ScoreNetworkParameters):
    """Create Score Network.

    This is a factory method responsible for instantiating the score network.
    """
    architecture = score_network_parameters.architecture
    assert (
        architecture in SCORE_NETWORKS_BY_ARCH.keys()
    ), f"Architecture {architecture} is not implemented. Possible choices are {SCORE_NETWORKS_BY_ARCH.keys()}"

    instantiated_score_network: ScoreNetwork = SCORE_NETWORKS_BY_ARCH[architecture](
        score_network_parameters
    )

    return instantiated_score_network


def create_score_network_parameters(
    score_network_dictionary: Dict[AnyStr, Any],
    global_parameters_dictionary: Dict[AnyStr, Any],
) -> ScoreNetworkParameters:
    """Create the score network parameters.

    Args:
        score_network_dictionary : input parameters that describe the score network.
        global_parameters_dictionary : global hyperparameters.

    Returns:
        score_network_parameters: the dataclass configuration object describing the score network.
    """
    assert len(global_parameters_dictionary["elements"]) == score_network_dictionary["num_atom_types"], \
        "There should be 'num_atom_types' entries in the 'elements' list."

    assert (
        "architecture" in score_network_dictionary
    ), "The architecture of the score network must be specified."
    score_network_architecture = score_network_dictionary["architecture"]

    assert score_network_architecture in SCORE_NETWORK_PARAMETERS_BY_ARCH.keys(), (
        f"Architecture {score_network_architecture} is not implemented. "
        f"Possible choices are {SCORE_NETWORK_PARAMETERS_BY_ARCH.keys()}"
    )

    score_network_dataclass = SCORE_NETWORK_PARAMETERS_BY_ARCH[
        score_network_architecture
    ]

    # Augment the configuration dictionary with a head (if present) and relevant global parameters.
    augmented_score_network_dictionary = dict(score_network_dictionary)

    if "prediction_head_parameters" in score_network_dictionary:
        head_config = score_network_dictionary["prediction_head_parameters"]
        prediction_head_parameters = create_parameters_from_configuration_dictionary(
            configuration=head_config,
            identifier="name",
            options=MACE_PREDICTION_HEAD_BY_NAME,
        )
        augmented_score_network_dictionary["prediction_head_parameters"] = (
            prediction_head_parameters
        )

    # Validate that there are no contradictions between the score config and global parameters
    for key, value in augmented_score_network_dictionary.items():
        if key in global_parameters_dictionary:
            assert (
                global_parameters_dictionary[key] == value
            ), f"inconsistent configuration values for {key}"

    # Complete the score config with global values
    all_fields = [field.name for field in dataclasses.fields(score_network_dataclass)]
    for key, value in global_parameters_dictionary.items():
        if key in all_fields:
            augmented_score_network_dictionary[key] = value

    score_network_parameters = score_network_dataclass(
        **augmented_score_network_dictionary
    )

    return score_network_parameters
