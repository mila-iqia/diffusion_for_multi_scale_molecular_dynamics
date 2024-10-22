import pytest
import torch
from crystal_diffusion.models.optimizer import (OptimizerParameters,
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


@pytest.fixture(params=[None, 1e-6])
def weight_decay(request):
    return request.param


@pytest.fixture(params=['adam', 'adamw'])
def optimizer_name(request):
    return request.param


@pytest.fixture()
def optimizer_parameters(optimizer_name, weight_decay):
    if weight_decay:
        return OptimizerParameters(name=optimizer_name, learning_rate=0.01, weight_decay=weight_decay)
    else:
        return OptimizerParameters(name=optimizer_name, learning_rate=0.01)


def test_load_optimizer(optimizer_name, optimizer_parameters, model):
    # This is more of a "smoke test": can the optimizer be instantiated and run without crashing.
    optimizer = load_optimizer(optimizer_parameters, model)
    optimizer.zero_grad()
    optimizer.step()
