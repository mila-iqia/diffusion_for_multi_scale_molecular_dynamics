import torch
from torch import nn


class MaceMLPScorePredictionHead(nn.Module):
    """A MLP head to predict scores given node features from MACE."""
    def __init__(self, mace_output_size: int, hidden_dimensions_size: int,
                 n_hidden_dimensions: int, spatial_dimension: int = 3):
        """Init method."""
        super(self).__init__()
        hidden_dimensions = [hidden_dimensions_size] * n_hidden_dimensions
        self.mlp_layers = nn.Sequential()
        # TODO we could add a linear layer to the times before concat with mace_output
        input_dimensions = [mace_output_size + 1] + hidden_dimensions  # add 1 for the times
        output_dimensions = hidden_dimensions + [spatial_dimension]
        add_relus = len(input_dimensions) * [True]
        add_relus[-1] = False

        for input_dimension, output_dimension, add_relu in zip(input_dimensions, output_dimensions, add_relus):
            self.mlp_layers.append(nn.Linear(input_dimension, output_dimension))
            if add_relu:
                self.mlp_layers.append(nn.ReLU())

    def forward(self, node_features: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Forward method.

        Args:
            node_features: mace node features. Dimension [batch_size, number_of_atoms, number_of_mace_features]
            times : diffusion time. Dimension [batch_size, number_of_atoms, 1]

        Returns:
            scores: scores computed using the MLP. Dimension [batch_size, number_of_atoms, spatial_dimension]
        """
        mlp_input = torch.cat([node_features, times], dim=-1)
        # pass through the final MLP layers
        scores = self.mlp_layers(mlp_input)
        return scores
