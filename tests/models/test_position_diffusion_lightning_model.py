import pytest
import torch
from pytorch_lightning import LightningDataModule, Trainer
from torch.utils.data import DataLoader, random_split

from crystal_diffusion.models.optimizer import (OptimizerParameters,
                                                ValidOptimizerName)
from crystal_diffusion.models.position_diffusion_lightning_model import (
    PositionDiffusionLightningModel, PositionDiffusionParameters)
from crystal_diffusion.models.scheduler import (
    CosineAnnealingLRSchedulerParameters, ReduceLROnPlateauSchedulerParameters,
    ValidSchedulerName)
from crystal_diffusion.models.score_network import (MLPScoreNetwork,
                                                    MLPScoreNetworkParameters)
from crystal_diffusion.namespace import RELATIVE_COORDINATES
from crystal_diffusion.samplers.variance_sampler import NoiseParameters
from crystal_diffusion.score.wrapped_gaussian_score import \
    get_sigma_normalized_score_brute_force
from crystal_diffusion.utils.tensor_utils import \
    broadcast_batch_tensor_to_all_dimensions

available_accelerators = ["cpu"]
if torch.cuda.is_available():
    available_accelerators.append("gpu")


class FakePositionsDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 4,
        dataset_size: int = 33,
        number_of_atoms: int = 8,
        spatial_dimension: int = 2,
    ):
        super().__init__()
        self.batch_size = batch_size
        all_relative_coordinates = torch.rand(dataset_size, number_of_atoms, spatial_dimension)
        box = torch.rand(spatial_dimension)
        self.data = [
            {RELATIVE_COORDINATES: configuration, 'box': box} for configuration in all_relative_coordinates
        ]
        self.train_data, self.val_data, self.test_data = None, None, None

    def setup(self, stage: str):
        self.train_data, self.val_data, self.test_data = random_split(
            self.data, lengths=[0.5, 0.3, 0.2]
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)


@pytest.mark.parametrize("spatial_dimension", [2, 3])
class TestPositionDiffusionLightningModel:
    @pytest.fixture(scope="class", autouse=True)
    def set_random_seed(self):
        torch.manual_seed(23452342)

    @pytest.fixture()
    def batch_size(self):
        return 4

    @pytest.fixture()
    def number_of_atoms(self):
        return 8

    @pytest.fixture()
    def unit_cell_size(self):
        return 10

    @pytest.fixture()
    def optimizer_name(self):
        return 'adam'

    @pytest.fixture()
    def scheduler_name(self):
        return None

    @pytest.fixture()
    def hyper_params(self, number_of_atoms, spatial_dimension, optimizer_name, scheduler_name):
        score_network_parameters = MLPScoreNetworkParameters(
            number_of_atoms=number_of_atoms,
            n_hidden_dimensions=3,
            hidden_dimensions_size=8,
            spatial_dimension=spatial_dimension,
        )

        optimizer_parameters = OptimizerParameters(
            name=ValidOptimizerName(optimizer_name), learning_rate=0.01, weight_decay=1e-6
        )

        if scheduler_name is None:
            scheduler_parameters = None
        elif scheduler_name == 'ReduceLROnPlateau':
            scheduler_parameters = ReduceLROnPlateauSchedulerParameters(name=ValidSchedulerName(scheduler_name),
                                                                        factor=0.5,
                                                                        patience=2)
        elif scheduler_name == 'CosineAnnealingLR':
            scheduler_parameters = CosineAnnealingLRSchedulerParameters(name=ValidSchedulerName(scheduler_name),
                                                                        T_max=5,
                                                                        eta_min=1e-5)
        else:
            raise ValueError(f"Untested case {scheduler_name}")

        noise_parameters = NoiseParameters(total_time_steps=15)

        hyper_params = PositionDiffusionParameters(
            score_network_parameters=score_network_parameters,
            optimizer_parameters=optimizer_parameters,
            scheduler_parameters=scheduler_parameters,
            noise_parameters=noise_parameters,
        )
        return hyper_params

    @pytest.fixture()
    def real_relative_coordinates(self, batch_size, number_of_atoms, spatial_dimension):
        relative_coordinates = torch.rand(batch_size, number_of_atoms, spatial_dimension)
        return relative_coordinates

    @pytest.fixture()
    def noisy_relative_coordinates(self, batch_size, number_of_atoms, spatial_dimension):
        noisy_relative_coordinates = torch.rand(
            batch_size, number_of_atoms, spatial_dimension
        )
        return noisy_relative_coordinates

    @pytest.fixture()
    def fake_datamodule(self, batch_size, number_of_atoms, spatial_dimension):
        data_module = FakePositionsDataModule(
            batch_size=batch_size,
            number_of_atoms=number_of_atoms,
            spatial_dimension=spatial_dimension,
        )
        return data_module

    @pytest.fixture()
    def times(self, batch_size):
        times = torch.rand(batch_size)
        return times

    @pytest.fixture()
    def sigmas(self, batch_size, number_of_atoms, spatial_dimension):
        sigma_values = 0.5 * torch.rand(batch_size)  # smaller sigmas for harder tests!
        sigmas = broadcast_batch_tensor_to_all_dimensions(
            sigma_values, final_shape=(batch_size, number_of_atoms, spatial_dimension)
        )
        return sigmas

    @pytest.fixture()
    def lightning_model(self, hyper_params):
        lightning_model = PositionDiffusionLightningModel(hyper_params)
        return lightning_model

    @pytest.fixture()
    def brute_force_target_normalized_score(
        self, noisy_relative_coordinates, real_relative_coordinates, sigmas
    ):
        shape = noisy_relative_coordinates.shape

        expected_scores = []
        for xt, x0, sigma in zip(
            noisy_relative_coordinates.flatten(),
            real_relative_coordinates.flatten(),
            sigmas.flatten(),
        ):
            u = torch.remainder(xt - x0, 1.0)

            # Note that the brute force algorithm is not robust and can sometimes produce NaNs in single precision!
            # Let's compute in double precision to avoid NaNs.
            expected_score = get_sigma_normalized_score_brute_force(
                u.to(torch.double), sigma.to(torch.double), kmax=20
            ).to(torch.float)
            expected_scores.append(expected_score)

        expected_scores = torch.tensor(expected_scores).reshape(shape)
        assert not torch.any(
            expected_scores.isnan()
        ), "The brute force algorithm produced NaN scores. Review input."
        return expected_scores

    @pytest.fixture()
    def unit_cell_sample(self, unit_cell_size, spatial_dimension, batch_size):
        return torch.diag(torch.Tensor([unit_cell_size] * spatial_dimension)).repeat(batch_size, 1, 1)

    def test_get_target_normalized_score(
        self,
        lightning_model,
        noisy_relative_coordinates,
        real_relative_coordinates,
        sigmas,
        brute_force_target_normalized_score,
        unit_cell_sample
    ):
        computed_target_normalized_scores = (
            lightning_model._get_target_normalized_score(
                noisy_relative_coordinates, real_relative_coordinates, sigmas
            )
        )

        torch.testing.assert_close(computed_target_normalized_scores,
                                   brute_force_target_normalized_score,
                                   atol=1e-7,
                                   rtol=1e-4)

    def test_get_predicted_normalized_score(
        self, mocker, lightning_model, noisy_relative_coordinates, times, unit_cell_sample
    ):
        mocker.patch.object(MLPScoreNetwork, "_forward_unchecked")

        _ = lightning_model._get_predicted_normalized_score(
            noisy_relative_coordinates, times, unit_cell_sample
        )

        list_calls = MLPScoreNetwork._forward_unchecked.mock_calls
        assert len(list_calls) == 1
        input_batch = list_calls[0][1][0]

        assert MLPScoreNetwork.position_key in input_batch
        torch.testing.assert_close(input_batch[MLPScoreNetwork.position_key], noisy_relative_coordinates)

        assert MLPScoreNetwork.timestep_key in input_batch
        torch.testing.assert_close(input_batch[MLPScoreNetwork.timestep_key], times.reshape(-1, 1))

    @pytest.mark.parametrize("accelerator", available_accelerators)
    @pytest.mark.parametrize("optimizer_name", ['adam', 'adamw'])
    @pytest.mark.parametrize("scheduler_name", [None, 'ReduceLROnPlateau', 'CosineAnnealingLR'])
    def test_smoke_test(self, lightning_model, fake_datamodule, accelerator):
        trainer = Trainer(fast_dev_run=3, accelerator=accelerator)
        trainer.fit(lightning_model, fake_datamodule)
        trainer.test(lightning_model, fake_datamodule)
