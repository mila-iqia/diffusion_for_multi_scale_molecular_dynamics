from dataclasses import dataclass, field
from typing import AnyStr, Dict

import torch
from faenet.fa_forward import model_forward
from faenet.transforms import FrameAveraging

from crystal_diffusion.models.faenet import FAENetWithSigma
from crystal_diffusion.models.faenet_utils import input_to_faenet
from crystal_diffusion.models.score_networks.score_network import (
    ScoreNetwork, ScoreNetworkParameters)
from crystal_diffusion.namespace import (NOISY_CARTESIAN_POSITIONS,
                                         NOISY_RELATIVE_COORDINATES, UNIT_CELL)


@dataclass(kw_only=True)
class FAENetScoreNetworkParameters(ScoreNetworkParameters):
    """Specific Hyper-parameters for FAENet score networks."""
    architecture: str = 'faenet'
    number_of_atoms: int  # the number of atoms in a cframonfiguration.
    r_max: float = 5.0
    hidden_channels: int = 256
    num_filters: int = 480
    num_interactions: int = 7
    num_gaussians: int = 136
    regress_forces: str = "direct_with_gradient_target"  # this can be changed to compute forces (score) from a gradient
    max_num_neighbors: int = 30
    tag_hidden_channels: int = 0  # 32  # only for OC2
    pg_hidden_channels: int = 64  # period & group embedding hidden channels
    phys_embeds: bool = False  # physics-aware embeddings for atoms
    phys_hidden_channels: int = 0
    energy_head: str = "weighted-av-final-embeds"
    skip_co: bool = False  # Skip connections {False, "add", "concat"}
    second_layer_MLP: bool = False  # in EmbeddingBlock
    complex_mp: bool = True  # 2-layer MLP in Interaction blocks
    mp_type: str = "updownscale_base"  # Message Passing type {'base', 'simple', 'updownscale', 'updownscale_base'}
    graph_norm: bool = True  # graph normalization layer
    sigma_hidden_channels: int = 0  # embedding for sigma
    force_decoder_type: str = "mlp"  # force head (`"simple"`, `"mlp"`, `"res"`, `"res_updown"`)
    force_decoder_model_config: dict = field(default=lambda: dict(
        simple=dict(hidden_channels=128, norm="batch1d"),  # norm = batch1d, layer or null
        mlp=dict(hidden_channels=256, norm="batch1d"),
        res=dict(hidden_channels=128, norm="batch1d"),
        res_updown=dict(hidden_channels=128, norm="batch1d")
    ))
    frame_averaging: str = '3D'
    fa_method: str = 'stochastic'


class FAENetScoreNetwork(ScoreNetwork):
    """Score network using FAENet as the score estimator.

    Inherits from the given framework's model class.
    """

    def __init__(self, hyper_params: FAENetScoreNetworkParameters):
        """__init__.

        Args:
            hyper_params : hyper parameters from the config file.
        """
        super(FAENetScoreNetwork, self).__init__(hyper_params)

        self.r_max = hyper_params.r_max

        faenet_config = dict(
            hidden_channels=hyper_params.hidden_channels,
            num_filters=hyper_params.num_filters,
            num_interactions=hyper_params.num_interactions,
            num_gaussians=hyper_params.num_gaussians,
            regress_forces=hyper_params.regress_forces,
            max_num_neighbors=hyper_params.max_num_neighbors,
            tag_hidden_channels=hyper_params.tag_hidden_channels,
            pg_hidden_channels=hyper_params.pg_hidden_channels,
            phys_embeds=hyper_params.phys_embeds,
            phys_hidden_channels=hyper_params.phys_hidden_channels,
            energy_head=hyper_params.energy_head,
            skip_co=hyper_params.skip_co,
            second_layer_MLP=hyper_params.second_layer_MLP,
            complex_mp=hyper_params.complex_mp,
            mp_type=hyper_params.mp_type,
            graph_norm=hyper_params.graph_norm,
            force_decoder_type=hyper_params.force_decoder_type,
            force_decoder_model_config=hyper_params.force_decoder_model_config,
            sigma_hidden_channels=hyper_params.sigma_hidden_channels,
        )

        self._natoms = hyper_params.number_of_atoms

        self.faenet_network = FAENetWithSigma(**faenet_config)
        self.fa_transform = FrameAveraging(hyper_params.frame_averaging, hyper_params.fa_method)

    def _check_batch(self, batch: Dict[AnyStr, torch.Tensor]):
        super(FAENetScoreNetwork, self)._check_batch(batch)
        number_of_atoms = batch[NOISY_RELATIVE_COORDINATES].shape[1]
        assert (
            number_of_atoms == self._natoms
        ), "The dimension corresponding to the number of atoms is not consistent with the configuration."

    def _forward_unchecked(self, batch: Dict[AnyStr, torch.Tensor], conditional: bool = False) -> torch.Tensor:
        """Forward unchecked.

        This method assumes that the input data has already been checked with respect to expectations
        and computes the scores assuming that the data is in the correct format.

        Args:
            batch : dictionary containing the data to be processed by the model.
            conditional (optional): if True, do a forward as though the model was conditional on the forces.
                Defaults to False.

        Returns:
            output : the scores computed by the model as a [batch_size, n_atom, spatial_dimension] tensor.
        """
        del conditional  # TODO implement conditional
        relative_coordinates = batch[NOISY_RELATIVE_COORDINATES]
        batch[NOISY_CARTESIAN_POSITIONS] = torch.bmm(relative_coordinates, batch[UNIT_CELL])  # positions in Angstrom
        graph_input = input_to_faenet(batch, radial_cutoff=self.r_max)
        mode = "train" if self.training else "eval"
        graph_input = self.fa_transform(graph_input)

        faenet_output = model_forward(
            batch=graph_input,
            model=self.faenet_network,
            frame_averaging='3D',  # TODO do not hard-code
            mode=mode,
            crystal_task=True  # use pbc for crystals - TODO do not hard code
        )

        flat_scores = faenet_output['forces']

        # Reshape the scores to have an explicit batch dimension
        scores = flat_scores.reshape(-1, self._natoms, self.spatial_dimension)

        return scores
