import pytest
import torch
from pytorch_lightning import LightningDataModule, Trainer
from torch.utils.data import DataLoader, random_split

from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_position_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.metrics.sampling_metrics_parameters import \
    SamplingMetricsParameters
from diffusion_for_multi_scale_molecular_dynamics.models.loss import \
    create_loss_parameters
from diffusion_for_multi_scale_molecular_dynamics.models.optimizer import \
    OptimizerParameters
from diffusion_for_multi_scale_molecular_dynamics.models.position_diffusion_lightning_model import (
    PositionDiffusionLightningModel, PositionDiffusionParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.scheduler import (
    CosineAnnealingLRSchedulerParameters, ReduceLROnPlateauSchedulerParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.mlp_score_network import \
    MLPScoreNetworkParameters
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    CARTESIAN_FORCES, RELATIVE_COORDINATES)
from diffusion_for_multi_scale_molecular_dynamics.samples.diffusion_sampling_parameters import \
    DiffusionSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.score.wrapped_gaussian_score import \
    get_sigma_normalized_score_brute_force
from diffusion_for_multi_scale_molecular_dynamics.utils.tensor_utils import \
    broadcast_batch_tensor_to_all_dimensions
from src.diffusion_for_multi_scale_molecular_dynamics.samplers.variance_sampler import \
    NoiseParameters


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
        all_relative_coordinates = torch.rand(
            dataset_size, number_of_atoms, spatial_dimension
        )
        box = torch.rand(spatial_dimension)
        self.data = [
            {
                RELATIVE_COORDINATES: configuration,
                "box": box,
                CARTESIAN_FORCES: torch.zeros_like(configuration),
            }
            for configuration in all_relative_coordinates
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
        torch.manual_seed(2345234)

    @pytest.fixture()
    def batch_size(self):
        return 4

    @pytest.fixture()
    def number_of_atoms(self):
        return 8

    @pytest.fixture()
    def unit_cell_size(self):
        return 10.1

    @pytest.fixture(params=["adam", "adamw"])
    def optimizer_parameters(self, request):
        return OptimizerParameters(
            name=request.param, learning_rate=0.01, weight_decay=1e-6
        )

    @pytest.fixture(params=[None, "ReduceLROnPlateau", "CosineAnnealingLR"])
    def scheduler_parameters(self, request):
        match request.param:
            case None:
                scheduler_parameters = None
            case "ReduceLROnPlateau":
                scheduler_parameters = ReduceLROnPlateauSchedulerParameters(
                    factor=0.5, patience=2
                )
            case "CosineAnnealingLR":
                scheduler_parameters = CosineAnnealingLRSchedulerParameters(
                    T_max=5, eta_min=1e-5
                )
            case _:
                raise ValueError(f"Untested case {request.param}")

        return scheduler_parameters

    @pytest.fixture(params=["mse", "weighted_mse"])
    def loss_parameters(self, request):
        model_dict = dict(loss=dict(algorithm=request.param))
        return create_loss_parameters(model_dictionary=model_dict)

    @pytest.fixture()
    def number_of_samples(self):
        return 12

    @pytest.fixture()
    def cell_dimensions(self, unit_cell_size, spatial_dimension):
        return spatial_dimension * [unit_cell_size]

    @pytest.fixture()
    def sampling_parameters(
        self, number_of_atoms, spatial_dimension, number_of_samples, cell_dimensions
    ):
        sampling_parameters = PredictorCorrectorSamplingParameters(
            number_of_atoms=number_of_atoms,
            spatial_dimension=spatial_dimension,
            number_of_samples=number_of_samples,
            cell_dimensions=cell_dimensions,
        )
        return sampling_parameters

    @pytest.fixture()
    def diffusion_sampling_parameters(self, sampling_parameters):
        noise_parameters = NoiseParameters(total_time_steps=5)
        metrics_parameters = SamplingMetricsParameters(
            structure_factor_max_distance=1.0
        )
        diffusion_sampling_parameters = DiffusionSamplingParameters(
            sampling_parameters=sampling_parameters,
            noise_parameters=noise_parameters,
            metrics_parameters=metrics_parameters,
        )
        return diffusion_sampling_parameters

    @pytest.fixture()
    def hyper_params(
        self,
        number_of_atoms,
        spatial_dimension,
        optimizer_parameters,
        scheduler_parameters,
        loss_parameters,
        sampling_parameters,
        diffusion_sampling_parameters,
    ):
        score_network_parameters = MLPScoreNetworkParameters(
            number_of_atoms=number_of_atoms,
            n_hidden_dimensions=3,
            embedding_dimensions_size=8,
            hidden_dimensions_size=8,
            spatial_dimension=spatial_dimension,
        )

        noise_parameters = NoiseParameters(total_time_steps=15)

        hyper_params = PositionDiffusionParameters(
            score_network_parameters=score_network_parameters,
            optimizer_parameters=optimizer_parameters,
            scheduler_parameters=scheduler_parameters,
            noise_parameters=noise_parameters,
            loss_parameters=loss_parameters,
            diffusion_sampling_parameters=diffusion_sampling_parameters,
        )
        return hyper_params

    @pytest.fixture()
    def real_relative_coordinates(self, batch_size, number_of_atoms, spatial_dimension):
        relative_coordinates = torch.rand(
            batch_size, number_of_atoms, spatial_dimension
        )
        return relative_coordinates

    @pytest.fixture()
    def noisy_relative_coordinates(
        self, batch_size, number_of_atoms, spatial_dimension
    ):
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
        return torch.diag(torch.Tensor([unit_cell_size] * spatial_dimension)).repeat(
            batch_size, 1, 1
        )

    # The brute force target normalized scores are *fragile*; they can return NaNs easily.
    # There is no point in running this test for all possible component combinations.
    @pytest.mark.parametrize("loss_parameters", ["mse"], indirect=True)
    @pytest.mark.parametrize("optimizer_parameters", ["adam"], indirect=True)
    @pytest.mark.parametrize("scheduler_parameters", [None], indirect=True)
    def test_get_target_normalized_score(
        self,
        lightning_model,
        noisy_relative_coordinates,
        real_relative_coordinates,
        sigmas,
        brute_force_target_normalized_score,
        unit_cell_sample,
    ):
        computed_target_normalized_scores = (
            lightning_model._get_target_normalized_score(
                noisy_relative_coordinates, real_relative_coordinates, sigmas
            )
        )

        torch.testing.assert_close(
            computed_target_normalized_scores,
            brute_force_target_normalized_score,
            atol=1e-7,
            rtol=1e-4,
        )

    def test_smoke_test(self, lightning_model, fake_datamodule, accelerator):
        trainer = Trainer(fast_dev_run=3, accelerator=accelerator)
        trainer.fit(lightning_model, fake_datamodule)
        trainer.test(lightning_model, fake_datamodule)

    def test_generate_sample(
        self, lightning_model, number_of_samples, number_of_atoms, spatial_dimension
    ):
        samples_batch = lightning_model.generate_samples()
        assert samples_batch[RELATIVE_COORDINATES].shape == (
            number_of_samples,
            number_of_atoms,
            spatial_dimension,
        )
