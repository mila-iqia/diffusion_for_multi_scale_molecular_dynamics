import pytest
import torch

from crystal_diffusion.models.optimizer import (OptimizerParameters,
                                                load_optimizer)
from crystal_diffusion.models.scheduler import (
    CosineAnnealingLRSchedulerParameters, ReduceLROnPlateauSchedulerParameters,
    load_scheduler_dictionary)


class FakeNeuralNet(torch.nn.Module):
    """A fake neural net for testing that we can attach an optimizer."""
    def __init__(self):
        super(FakeNeuralNet, self).__init__()
        self.linear_layer = torch.nn.Linear(in_features=4, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


@pytest.fixture
def optimizer():
    model = FakeNeuralNet()
    optimizer_parameters = OptimizerParameters(name='adam', learning_rate=0.001, weight_decay=1e-6)
    optimizer = load_optimizer(optimizer_parameters, model)
    return optimizer


@pytest.fixture(params=['CosineAnnealingLR', 'ReduceLROnPlateau'])
def scheduler_name(request):
    return request.param


@pytest.fixture
def scheduler_parameters(scheduler_name):
    match scheduler_name:
        case 'CosineAnnealingLR':
            parameters = CosineAnnealingLRSchedulerParameters(name=scheduler_name, T_max=1000, eta_min=0.001)
        case 'ReduceLROnPlateau':
            parameters = ReduceLROnPlateauSchedulerParameters(name=scheduler_name, factor=0.243, patience=17)
        case _:
            raise ValueError(f"Untested case: {scheduler_name}")
    return parameters


def test_load_scheduler(optimizer, scheduler_parameters):
    _ = load_scheduler_dictionary(scheduler_parameters=scheduler_parameters, optimizer=optimizer)
