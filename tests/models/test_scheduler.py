import pytest
import torch

from crystal_diffusion.models.optimizer import (OptimizerParameters,
                                                ValidOptimizerName,
                                                load_optimizer)
from crystal_diffusion.models.scheduler import (
    CosineAnnealingLRSchedulerParameters, ReduceLROnPlateauSchedulerParameters,
    ValidSchedulerName, load_scheduler)


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
    optimizer_parameters = OptimizerParameters(name=ValidOptimizerName.adam, learning_rate=0.001, weight_decay=1e-6)
    optimizer = load_optimizer(optimizer_parameters, model)
    return optimizer


@pytest.fixture
def scheduler_parameters(scheduler_name: ValidSchedulerName):
    valid_scheduler_name = ValidSchedulerName(scheduler_name)

    if valid_scheduler_name is ValidSchedulerName.reduce_lr_on_plateau:
        parameters = ReduceLROnPlateauSchedulerParameters(name=valid_scheduler_name, factor=0.243, patience=17)
        pass
    elif valid_scheduler_name is ValidSchedulerName.cosine_annealing_lr:
        parameters = CosineAnnealingLRSchedulerParameters(name=valid_scheduler_name, T_max=1000, eta_min=0.001)
    else:
        raise ValueError(f"Untested case: {valid_scheduler_name}")
    return parameters


@pytest.mark.parametrize("scheduler_name", [option.value for option in list(ValidSchedulerName)])
def test_load_scheduler(optimizer, scheduler_parameters):
    _ = load_scheduler(hyper_params=scheduler_parameters, optimizer=optimizer)
