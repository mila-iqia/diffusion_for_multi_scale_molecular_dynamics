from crystal_diffusion.models.score_networks.diffusion_mace_score_network import \
    DiffusionMACEScoreNetwork
from crystal_diffusion.models.score_networks.mace_score_network import \
    MACEScoreNetwork
from crystal_diffusion.models.score_networks.mlp_score_network import \
    MLPScoreNetwork
from crystal_diffusion.models.score_networks.score_network import \
    ScoreNetworkParameters

_SCORE_NETWORKS_BY_ARCH = dict(mlp=MLPScoreNetwork, mace=MACEScoreNetwork, diffusion_mace=DiffusionMACEScoreNetwork)


def create_score_network(score_network_parameters: ScoreNetworkParameters):
    """Create Score Network.

    This is a factory method responsible for instantiating the score network.
    """
    architecture = score_network_parameters.architecture
    assert architecture in _SCORE_NETWORKS_BY_ARCH.keys(), \
        f"Architecture {architecture} is not implemented. Possible choices are {_SCORE_NETWORKS_BY_ARCH.keys()}"

    return _SCORE_NETWORKS_BY_ARCH[architecture](score_network_parameters)
