import pytest
import torch

from crystal_diffusion.models.optimizer import (OptimizerParameters,
                                                ValidOptimizerNames)
from crystal_diffusion.models.position_diffusion_lightning_model import (
    PositionDiffusionLightningModel, PositionDiffusionParameters)
from crystal_diffusion.models.score_network import MLPScoreNetworkParameters
from crystal_diffusion.samplers.variance_sampler import NoiseParameters


@pytest.mark.parametrize("spatial_dimension", [2, 3])
class TestPositionDiffusionLightningModel:

    @pytest.fixture()
    def batch_size(self):
        return 4

    @pytest.fixture()
    def number_of_atoms(self):
        return 8

    @pytest.fixture()
    def hyper_params(self, number_of_atoms, spatial_dimension):

        score_network_parameters = MLPScoreNetworkParameters(number_of_atoms=number_of_atoms,
                                                             hidden_dim=16,
                                                             spatial_dimension=spatial_dimension)

        optimizer_parameters = OptimizerParameters(name=ValidOptimizerNames('adam'), learning_rate=0.01)
        noise_parameters = NoiseParameters(total_time_steps=15)

        hyper_params = PositionDiffusionParameters(score_network_parameters=score_network_parameters,
                                                   optimizer_parameters=optimizer_parameters,
                                                   noise_parameters=noise_parameters,
                                                   kmax_target_score=4)
        return hyper_params

    @pytest.fixture()
    def batch(self, batch_size, number_of_atoms, spatial_dimension):
        torch.manual_seed(23452342)
        relative_positions = torch.rand(batch_size, number_of_atoms, spatial_dimension)
        return dict(relative_positions=relative_positions)

    @pytest.fixture()
    def lightning_model(self, hyper_params):
        lightning_model = PositionDiffusionLightningModel(hyper_params)
        return lightning_model

    def test_smoke_training_step(self, lightning_model, batch):
        _ = lightning_model.training_step(batch, batch_idx=0)
