import pytest
import torch

from crystal_diffusion.models.optimizer import (OptimizerParameters,
                                                ValidOptimizerName,
                                                load_optimizer)


class FakeNeuralNet(torch.nn.Module):
    """A fake neural net for testing that we can attach an optimizer."""
    def __init__(self):
        super(FakeNeuralNet, self).__init__()
        self.linear_layer = torch.nn.Linear(in_features=4, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


@pytest.fixture()
def model():
    return FakeNeuralNet()


@pytest.fixture()
def optimizer_parameters(optimizer_name, weight_decay):
    valid_optimizer_name = ValidOptimizerName(optimizer_name)
    if weight_decay is False:
        return OptimizerParameters(name=valid_optimizer_name, learning_rate=0.01)
    if weight_decay is True:
        return OptimizerParameters(name=valid_optimizer_name, learning_rate=0.01, weight_decay=1e-6)


@pytest.mark.parametrize("optimizer_name", [option.value for option in list(ValidOptimizerName)])
@pytest.mark.parametrize("weight_decay", [True, False])
def test_load_optimizer(optimizer_name, optimizer_parameters, model):
    # This is more of a "smoke test": can the optimizer be instantiated and run without crashing.
    optimizer = load_optimizer(optimizer_parameters, model)
    optimizer.zero_grad()
    optimizer.step()
