import pytest
import torch

from crystal_diffusion.models.optimizer import (OptimizerParameters,
                                                ValidOptimizerNames,
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
def optimizer_parameters(optimizer_name):
    valid_optimizer_name = ValidOptimizerNames(optimizer_name)
    return OptimizerParameters(name=valid_optimizer_name, learning_rate=0.01)


@pytest.mark.parametrize("optimizer_name", [option.value for option in list(ValidOptimizerNames)])
def test_load_optimizer(optimizer_name, optimizer_parameters, model):
    # This is more of a "smoke test": can the optimizer be instantiated and run without crashing.
    optimizer = load_optimizer(optimizer_parameters, model)
    optimizer.zero_grad()
    optimizer.step()
