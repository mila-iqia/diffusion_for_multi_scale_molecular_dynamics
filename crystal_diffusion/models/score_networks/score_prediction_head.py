from dataclasses import dataclass

import e3nn
import torch
from e3nn import o3
from e3nn.nn import Activation
from mace.modules import LinearNodeEmbeddingBlock, gate_dict
from torch import nn

from crystal_diffusion.models.mace_utils import \
    get_normalized_irreps_permutation_indices


@dataclass(kw_only=True)
class MaceScorePredictionHeadParameters:
    """Base Hyper-parameters for score networks."""
    name: str  # this must be overloaded to identify the type of prediction head
    spatial_dimension: int = 3  # the dimension of Euclidean space where atoms live.


class MaceScorePredictionHead(nn.Module):
    """A Base class for the head that predicts the scores given node features from MACE."""
    def __init__(self, output_node_features_irreps: o3.Irreps, hyper_params: MaceScorePredictionHeadParameters):
        """Init method."""
        super().__init__()
        self.output_node_features_irreps = output_node_features_irreps
        self.hyper_params = hyper_params

    def forward(self, flat_node_features: torch.Tensor, flat_times: torch.Tensor) -> torch.Tensor:
        """Forward method.

        Here, 'flat' means that the batch dimension and the number_of_atoms dimensions are combined (flattened).

        Args:
            flat_node_features: mace node features. Dimension [batch_size * number_of_atoms, number_of_mace_features]
            flat_times : diffusion time. Dimension [batch_size * number_of_atoms, 1]

        Returns:
            flat_scores: scores computed using the MLP. Dimension [batch_size * number_of_atoms, spatial_dimension]
        """
        raise NotImplementedError("This method must be implemented in a derived class")


@dataclass(kw_only=True)
class MaceMLPScorePredictionHeadParameters(MaceScorePredictionHeadParameters):
    """Parameters for a MLP prediction head."""
    name: str = 'mlp'
    hidden_dimensions_size: int   # dimension of a linear layer
    n_hidden_dimensions: int  # number of linear layers in the MLP


class MaceMLPScorePredictionHead(MaceScorePredictionHead):
    """A MLP head to predict scores given node features from MACE."""
    def __init__(self, output_node_features_irreps: o3.Irreps, hyper_params: MaceMLPScorePredictionHeadParameters):
        """Init method."""
        super().__init__(output_node_features_irreps, hyper_params)
        hidden_dimensions = [hyper_params.hidden_dimensions_size] * hyper_params.n_hidden_dimensions
        self.mlp_layers = torch.nn.Sequential()
        # TODO we could add a linear layer to the times before concat with mace_output
        input_dimensions = [output_node_features_irreps.dim + 1] + hidden_dimensions  # add 1 for the times
        output_dimensions = hidden_dimensions + [hyper_params.spatial_dimension]
        add_relus = len(input_dimensions) * [True]
        add_relus[-1] = False

        for input_dimension, output_dimension, add_relu in zip(input_dimensions, output_dimensions, add_relus):
            self.mlp_layers.append(nn.Linear(input_dimension, output_dimension))
            if add_relu:
                self.mlp_layers.append(nn.ReLU())

    def forward(self, flat_node_features: torch.Tensor, flat_times: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        mlp_input = torch.cat([flat_node_features, flat_times], dim=-1)
        # pass through the final MLP layers
        flat_scores = self.mlp_layers(mlp_input)
        return flat_scores


@dataclass(kw_only=True)
class MaceEquivariantScorePredictionHeadParameters(MaceScorePredictionHeadParameters):
    """Parameters for an equivariant prediction head."""
    name: str = 'equivariant'
    time_embedding_irreps: str = "16x0e"
    gate: str = "silu"  # non linearity for last readout - choices: ["silu", "tanh", "abs", "None"]
    number_of_layers: int


class MaceEquivariantScorePredictionHead(MaceScorePredictionHead):
    """An Equivariant head to predict scores given node features from MACE."""
    def __init__(self, output_node_features_irreps: o3.Irreps,
                 hyper_params: MaceEquivariantScorePredictionHeadParameters):
        """Init method."""
        super().__init__(output_node_features_irreps, hyper_params)

        # raise the dimensionality of the time scalar
        time_irreps_in = o3.Irreps("1x0e")  # time is a scalar
        time_irreps_out = o3.Irreps(hyper_params.time_embedding_irreps)

        self.time_embedding_linear_layer = LinearNodeEmbeddingBlock(irreps_in=time_irreps_in,
                                                                    irreps_out=time_irreps_out)

        # The concatenated node features have representation 'output_node_features_irreps', which
        # is potentially out of order. We will pre-concatenate the time embedding to this data.
        # It is important to then sort the data columns to make sure all 'like' channels are mixed
        # by the various subsequent layers.
        head_input_irreps = time_irreps_out + output_node_features_irreps

        sorted_irreps, _ = (
            get_normalized_irreps_permutation_indices(head_input_irreps))

        # mix the time embedding 0e irrep with the different components of the MACE output and avoid a dimensionality
        # explosion with FullyConnectedTensorProduct that only computes the specified channels in the output and applies
        # linear layers to control the dimensionality across each irrep
        self.time_mixing_layer = o3.FullyConnectedTensorProduct(
            irreps_in1=time_irreps_out,
            irreps_in2=output_node_features_irreps,
            irreps_out=sorted_irreps
        )

        self.head = torch.nn.Sequential()

        for _ in range(hyper_params.number_of_layers):
            linear = o3.Linear(irreps_in=sorted_irreps, irreps_out=sorted_irreps)
            self.head.append(linear)

            batch_norm = e3nn.nn.BatchNorm(irreps=sorted_irreps)
            self.head.append(batch_norm)

            gate = gate_dict[hyper_params.gate]
            # There must be one activation per L-channel, and that activation must be 'None'  for l !=0.
            # The first channel is l = 0.
            number_of_l_channels = len(sorted_irreps)
            activations = [gate] + (number_of_l_channels - 1) * [None]
            non_linearity = Activation(irreps_in=sorted_irreps, acts=activations)
            self.head.append(non_linearity)

        # the output is a single vector.
        linear_readout = o3.Linear(irreps_in=sorted_irreps, irreps_out=o3.Irreps("1x1o"))
        self.head.append(linear_readout)

    def forward(self, flat_node_features: torch.Tensor, flat_times: torch.Tensor) -> torch.Tensor:
        """Forward method.

        Here, 'flat' means that the batch dimension and the number_of_atoms dimensions are combined (flattened).

        Args:
            flat_node_features: mace node features. Dimension [batch_size * number_of_atoms, number_of_mace_features]
            flat_times : diffusion time. Dimension [batch_size * number_of_atoms, 1]

        Returns:
            flat_scores: scores computed using the head. Dimension [batch_size * number_of_atoms, spatial_dimension]
        """
        embedded_times = self.time_embedding_linear_layer(flat_times)
        mixed_features = self.time_mixing_layer(embedded_times, flat_node_features)
        flat_scores = self.head(mixed_features)

        return flat_scores


# Register the possible MACE prediction heads as  key:  model class
MACE_PREDICTION_HEADS = dict(mlp=MaceMLPScorePredictionHead, equivariant=MaceEquivariantScorePredictionHead)


def instantiate_mace_prediction_head(output_node_features_irreps: o3.Irreps,
                                     prediction_head_parameters: MaceScorePredictionHeadParameters) \
        -> MaceScorePredictionHead:
    """Instantiate MACE prediction head.

    Args:
        output_node_features_irreps : irreps of the node features.
        prediction_head_parameters : the hyperparameters defining the prediction head.

    Returns:
        prediction_head: torch module to predict the scores from the output of MACE.
    """
    head_name = prediction_head_parameters.name
    assert head_name in MACE_PREDICTION_HEADS, f"MACE prediction head '{head_name}' is not implemented"

    head_class = MACE_PREDICTION_HEADS[head_name]
    prediction_head = head_class(output_node_features_irreps, prediction_head_parameters)
    return prediction_head
