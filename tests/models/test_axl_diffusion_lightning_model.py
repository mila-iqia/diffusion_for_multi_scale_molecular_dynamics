from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pytest
import torch
from lightning import LightningDataModule, Trainer
from torch.utils.data import DataLoader, default_collate, random_split

from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.noising_transform import \
    NoisingTransform
from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.loss.loss_parameters import \
    create_loss_parameters
from diffusion_for_multi_scale_molecular_dynamics.metrics.sampling_metrics_parameters import \
    SamplingMetricsParameters
from diffusion_for_multi_scale_molecular_dynamics.models.axl_diffusion_lightning_model import (
    AXLDiffusionLightningModel, AXLDiffusionParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.optimizer import \
    OptimizerParameters
from diffusion_for_multi_scale_molecular_dynamics.models.scheduler import (
    CosineAnnealingLRSchedulerParameters, ReduceLROnPlateauSchedulerParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.egnn_score_network import \
    EGNNScoreNetworkParameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.mlp_score_network import \
    MLPScoreNetworkParameters
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    ATOM_TYPES, AXL_COMPOSITION, CARTESIAN_FORCES, LATTICE_PARAMETERS, NOISE,
    NOISY_ATOM_TYPES, NOISY_AXL_COMPOSITION, NOISY_LATTICE_PARAMETERS,
    NOISY_RELATIVE_COORDINATES, PADDED_ATOM_TYPE, Q_BAR_MATRICES,
    Q_BAR_TM1_MATRICES, Q_MATRICES, RELATIVE_COORDINATES, TIME, TIME_INDICES)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_scheduler import \
    NoiseScheduler
from diffusion_for_multi_scale_molecular_dynamics.noisers.lattice_noiser import \
    LatticeDataParameters
from diffusion_for_multi_scale_molecular_dynamics.oracle.energy_oracle import (
    EnergyOracle, OracleParameters)
from diffusion_for_multi_scale_molecular_dynamics.oracle.energy_oracle_factory import (
    ENERGY_ORACLE_BY_NAME, ORACLE_PARAMETERS_BY_NAME)
from diffusion_for_multi_scale_molecular_dynamics.sampling.diffusion_sampling_parameters import \
    DiffusionSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.score.wrapped_gaussian_score import \
    get_sigma_normalized_score_brute_force
from diffusion_for_multi_scale_molecular_dynamics.utils.tensor_utils import (
    broadcast_batch_matrix_tensor_to_all_dimensions,
    broadcast_batch_tensor_to_all_dimensions)
from tests.fake_data_utils import generate_random_string


@dataclass(kw_only=True)
class FakeOracleParameters(OracleParameters):
    name = "test"


class FakeEnergyOracle(EnergyOracle):

    def _compute_one_configuration_energy_and_forces(
        self,
        cartesian_positions: np.ndarray,
        basis_vectors: np.ndarray,
        atom_types: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        return np.random.rand(), torch.rand(*cartesian_positions.shape)


class FakeAXLDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 4,
        dataset_size: int = 33,
        number_of_atoms: int = 8,
        spatial_dimension: int = 2,
        num_atom_types: int = 2,
    ):
        super().__init__()

        noising_transform = NoisingTransform(noise_parameters=NoiseParameters(total_time_steps=10),
                                             num_atom_types=num_atom_types,
                                             spatial_dimension=spatial_dimension,
                                             use_optimal_transport=False)

        self.batch_size = batch_size
        all_relative_coordinates = torch.rand(
            dataset_size, number_of_atoms, spatial_dimension
        )
        potential_energies = torch.rand(dataset_size)
        all_atom_types = torch.randint(
            0, num_atom_types, (dataset_size, number_of_atoms)
        )
        box = torch.rand(int(spatial_dimension * (spatial_dimension + 1) / 2))

        raw_data = [
            {
                RELATIVE_COORDINATES: coordinate_configuration,
                ATOM_TYPES: atom_configuration,
                LATTICE_PARAMETERS: box,
                CARTESIAN_FORCES: torch.zeros_like(coordinate_configuration),
                "potential_energy": potential_energy,
                "natom": torch.tensor(number_of_atoms),
            }
            for coordinate_configuration, atom_configuration, potential_energy in zip(
                all_relative_coordinates, all_atom_types, potential_energies
            )
        ]
        raw_data = default_collate(raw_data)

        batched_data = noising_transform.transform(raw_data)

        keys = batched_data.keys()

        self.data = [{key: batched_data[key][idx] for key in keys} for idx in range(batch_size)]

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


class FakeVariableNatomAXLDataModule(LightningDataModule):
    """DataModule for variable-natom training smoke tests.

    Samples have different numbers of real atoms; padded atoms get NaN coordinates
    and PADDED_ATOM_TYPE. The lattice is fixed (not noised) so it stays safely above
    any radial cutoff.
    """

    def __init__(
        self,
        spatial_dimension: int = 3,
        num_atom_types: int = 2,
        max_atoms: int = 8,
        lattice_size: float = 10.0,
    ):
        super().__init__()
        natoms_list = [3, 5, max_atoms, 4, 6, max_atoms, 5, 3]
        dataset_size = len(natoms_list)
        self.batch_size = 4

        noising_transform = NoisingTransform(
            noise_parameters=NoiseParameters(total_time_steps=10),
            num_atom_types=num_atom_types,
            spatial_dimension=spatial_dimension,
            use_optimal_transport=False,
            use_fixed_lattice_parameters=True,
        )

        lattice_params_dim = spatial_dimension * (spatial_dimension + 1) // 2
        lattice_params = torch.zeros(lattice_params_dim)
        lattice_params[:spatial_dimension] = lattice_size

        x0 = torch.rand(dataset_size, max_atoms, spatial_dimension)
        a0 = torch.randint(0, num_atom_types, (dataset_size, max_atoms))
        for b, n in enumerate(natoms_list):
            x0[b, n:] = float('nan')
            a0[b, n:] = PADDED_ATOM_TYPE

        raw_data = {
            RELATIVE_COORDINATES: x0,
            ATOM_TYPES: a0,
            LATTICE_PARAMETERS: lattice_params.unsqueeze(0).expand(dataset_size, -1).clone(),
            CARTESIAN_FORCES: torch.zeros(dataset_size, max_atoms, spatial_dimension),
            "natom": torch.tensor(natoms_list),
        }
        batched_data = noising_transform.transform(raw_data)
        keys = batched_data.keys()
        self.data = [{key: batched_data[key][idx] for key in keys} for idx in range(dataset_size)]
        self.train_data = self.val_data = self.test_data = None

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
        torch.manual_seed(234523)

    @pytest.fixture()
    def batch_size(self):
        return 4

    @pytest.fixture()
    def number_of_atoms(self):
        return 8

    @pytest.fixture()
    def num_atom_types(self):
        return 4

    @pytest.fixture
    def unique_elements(self, num_atom_types):
        return [generate_random_string(size=8) for _ in range(num_atom_types)]

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
        model_dict = dict(loss=dict(coordinates_algorithm=request.param))
        return create_loss_parameters(model_dictionary=model_dict)

    @pytest.fixture()
    def number_of_samples(self):
        return 12

    @pytest.fixture()
    def lattice_parameters(self, spatial_dimension):
        lattice_params = LatticeDataParameters(
            spatial_dimension=spatial_dimension,
        )
        return lattice_params

    @pytest.fixture()
    def sampling_parameters(
        self,
        number_of_atoms,
        spatial_dimension,
        number_of_samples,
        num_atom_types,
    ):
        sampling_parameters = PredictorCorrectorSamplingParameters(
            number_of_atoms=number_of_atoms,
            spatial_dimension=spatial_dimension,
            number_of_samples=number_of_samples,
            num_atom_types=num_atom_types,
        )
        return sampling_parameters

    @pytest.fixture()
    def diffusion_sampling_parameters(self, sampling_parameters):
        noise_parameters = NoiseParameters(total_time_steps=5)
        metrics_parameters = SamplingMetricsParameters(
            structure_factor_max_distance=1.0,
            compute_energies=True,
            compute_structure_factor=False,
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
        unique_elements,
        num_atom_types,
        spatial_dimension,
        optimizer_parameters,
        scheduler_parameters,
        loss_parameters,
        sampling_parameters,
        diffusion_sampling_parameters,
    ):
        score_network_parameters = MLPScoreNetworkParameters(
            number_of_atoms=number_of_atoms,
            num_atom_types=num_atom_types,
            n_hidden_dimensions=3,
            relative_coordinates_embedding_dimensions_size=8,
            noise_embedding_dimensions_size=8,
            time_embedding_dimensions_size=8,
            atom_type_embedding_dimensions_size=8,
            lattice_parameters_embedding_dimensions_size=8,
            hidden_dimensions_size=8,
            spatial_dimension=spatial_dimension,
        )

        oracle_parameters = OracleParameters(name="test", elements=unique_elements)

        hyper_params = AXLDiffusionParameters(
            score_network_parameters=score_network_parameters,
            optimizer_parameters=optimizer_parameters,
            scheduler_parameters=scheduler_parameters,
            loss_parameters=loss_parameters,
            diffusion_sampling_parameters=diffusion_sampling_parameters,
            oracle_parameters=oracle_parameters,
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
    def fake_datamodule(
        self, batch_size, number_of_atoms, spatial_dimension, num_atom_types
    ):
        data_module = FakeAXLDataModule(
            batch_size=batch_size,
            number_of_atoms=number_of_atoms,
            spatial_dimension=spatial_dimension,
            num_atom_types=num_atom_types,
        )
        return data_module

    @pytest.fixture()
    def times(self, batch_size):
        times = torch.rand(batch_size)
        return times

    @pytest.fixture()
    def sigma_cart(self, batch_size):
        return 0.5 + 4.5 * torch.rand(batch_size)

    @pytest.fixture()
    def unit_cell_lattice_diagonals(self, unit_cell_size, batch_size, spatial_dimension):
        return torch.full((batch_size, spatial_dimension), unit_cell_size)

    @pytest.fixture()
    def lightning_model(self, mocker, hyper_params):
        fake_oracle_parameters_by_name = dict(test=FakeOracleParameters)
        fake_energy_oracle_by_name = dict(test=FakeEnergyOracle)

        mocker.patch.dict(ORACLE_PARAMETERS_BY_NAME, fake_oracle_parameters_by_name)
        mocker.patch.dict(ENERGY_ORACLE_BY_NAME, fake_energy_oracle_by_name)

        lightning_model = AXLDiffusionLightningModel(hyper_params)
        return lightning_model

    @pytest.fixture()
    def brute_force_target_normalized_score(
        self, noisy_relative_coordinates, real_relative_coordinates, sigma_cart, unit_cell_size,
        number_of_atoms, spatial_dimension, batch_size,
    ):
        shape = noisy_relative_coordinates.shape

        # sigma_rel_d = sigma_cart / unit_cell_size (same for all directions for a cubic cell)
        sigma_rel_values = sigma_cart / unit_cell_size  # shape [batch_size]
        sigma_rel_broadcast = broadcast_batch_tensor_to_all_dimensions(
            sigma_rel_values, final_shape=(batch_size, number_of_atoms, spatial_dimension)
        )

        expected_scores = []
        for xt, x0, sigma_rel in zip(
            noisy_relative_coordinates.flatten(),
            real_relative_coordinates.flatten(),
            sigma_rel_broadcast.flatten(),
        ):
            u = torch.remainder(xt - x0, 1.0)

            # Note that the brute force algorithm is not robust and can sometimes produce NaNs in single precision!
            # Let's compute in double precision to avoid NaNs.
            expected_score = get_sigma_normalized_score_brute_force(
                u.to(torch.double), sigma_rel.to(torch.double), kmax=20
            ).to(torch.float)
            expected_scores.append(expected_score)

        expected_scores = torch.tensor(expected_scores).reshape(shape)
        assert not torch.any(
            expected_scores.isnan()
        ), "The brute force algorithm produced NaN scores. Review input."
        return expected_scores

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
        sigma_cart,
        unit_cell_lattice_diagonals,
        brute_force_target_normalized_score,
    ):
        computed_target_normalized_scores = (
            lightning_model._get_relative_coordinates_target_cartesian_normalized_score(
                noisy_relative_coordinates,
                real_relative_coordinates,
                sigma_cart,
                unit_cell_lattice_diagonals,
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
        assert samples_batch[AXL_COMPOSITION].X.shape == (
            number_of_samples,
            number_of_atoms,
            spatial_dimension,
        )
        assert samples_batch[AXL_COMPOSITION].A.shape == (
            number_of_samples,
            number_of_atoms,
        )


@pytest.mark.parametrize("num_atom_types", [1, 2])
@pytest.mark.parametrize("edges_connection", ["radial_cutoff", "fully_connected"])
class TestAXLDiffusionLightningModelWithPadding:
    """Tests for _generic_step correctness under variable-natom padding."""

    @pytest.fixture(scope="class", autouse=True)
    def set_random_seed(self):
        torch.manual_seed(98765432)

    @pytest.fixture()
    def spatial_dimension(self):
        return 3

    @pytest.fixture()
    def loss_parameters(self):
        return create_loss_parameters(model_dictionary={})

    @pytest.fixture()
    def optimizer_parameters(self):
        return OptimizerParameters(name="adam", learning_rate=0.001, weight_decay=0.0)

    @pytest.fixture(params=["radial_cutoff", "fully_connected"])
    def edges_connection(self, request):
        return request.param

    @pytest.fixture()
    def hyper_params(self, num_atom_types, spatial_dimension, loss_parameters, optimizer_parameters, edges_connection):
        if edges_connection == "radial_cutoff":
            score_network_parameters = EGNNScoreNetworkParameters(
                num_atom_types=num_atom_types,
                spatial_dimension=spatial_dimension,
                edges="radial_cutoff",
                radial_cutoff=3.0,
            )
        elif edges_connection == "fully_connected":
            score_network_parameters = EGNNScoreNetworkParameters(
                num_atom_types=num_atom_types,
                spatial_dimension=spatial_dimension,
                edges="fully_connected",
            )
        return AXLDiffusionParameters(
            score_network_parameters=score_network_parameters,
            loss_parameters=loss_parameters,
            optimizer_parameters=optimizer_parameters,
        )

    @pytest.fixture()
    def lightning_model(self, hyper_params):
        return AXLDiffusionLightningModel(hyper_params)

    @pytest.fixture()
    def noising_transform(self, num_atom_types, spatial_dimension):
        # Fix the lattice so the noisy cell never shrinks below radial_cutoff.
        # With sigma_max_cart=5.0 and natoms=4, sigma_n ~ 3.15 Å — enough to push an 8 Å
        # diagonal below radial_cutoff=3.0. This test targets variable-natom padding, not
        # lattice noising, so fixing the lattice is the right approach.
        return NoisingTransform(
            noise_parameters=NoiseParameters(total_time_steps=10),
            num_atom_types=num_atom_types,
            spatial_dimension=spatial_dimension,
            use_optimal_transport=False,
            use_fixed_lattice_parameters=True,
        )

    @pytest.fixture()
    def lattice_parameters_orthogonal(self, spatial_dimension):
        # Orthogonal box with diagonal > 2.2 * radial_cutoff (6.6) so the EGNN doesn't clip
        lattice_dim = spatial_dimension * (spatial_dimension + 1) // 2
        lp = torch.zeros(lattice_dim)
        lp[:spatial_dimension] = 8.0
        return lp

    @pytest.fixture()
    def mixed_natom_noised_batch(
        self, spatial_dimension, num_atom_types, lattice_parameters_orthogonal, noising_transform
    ):
        batch_size = 4
        max_atom = 8
        natoms_list = [4, 8, 6, 8]
        x0 = torch.rand(batch_size, max_atom, spatial_dimension)
        a0 = torch.randint(0, num_atom_types, (batch_size, max_atom))
        for b, n in enumerate(natoms_list):
            x0[b, n:, :] = float('nan')
            a0[b, n:] = PADDED_ATOM_TYPE
        batch = {
            RELATIVE_COORDINATES: x0,
            ATOM_TYPES: a0,
            LATTICE_PARAMETERS: lattice_parameters_orthogonal.unsqueeze(0).expand(batch_size, -1).clone(),
            CARTESIAN_FORCES: torch.zeros(batch_size, max_atom, spatial_dimension),
            "natom": torch.tensor(natoms_list),
        }
        return noising_transform.transform(batch)

    def test_no_padded_atom_type_reaches_score_network(self, mocker, lightning_model, mixed_natom_noised_batch):
        """PADDED_ATOM_TYPE (-1) must be replaced with 0 before reaching the score network.

        On CPU, nn.Embedding(-1) silently wraps to the last element, so checking the loss
        is not sufficient. On CUDA the same index produces NaN. This test catches the
        regression on any device by inspecting the atom types the network actually receives.
        """
        spy = mocker.spy(lightning_model.axl_network, 'forward')

        with torch.no_grad():
            lightning_model._generic_step(mixed_natom_noised_batch, batch_idx=0)

        assert spy.called, "Score network forward was never called"
        batch_arg = spy.call_args[0][0]
        at_seen = batch_arg[NOISY_AXL_COMPOSITION].A
        assert (at_seen != PADDED_ATOM_TYPE).all(), \
            f"PADDED_ATOM_TYPE (-1) reached the score network in positions: {(at_seen == PADDED_ATOM_TYPE).nonzero()}"

    def test_loss_is_finite_with_variable_natom(self, lightning_model, mixed_natom_noised_batch):
        with torch.no_grad():
            output = lightning_model._generic_step(mixed_natom_noised_batch, batch_idx=0)
        assert output["loss"].isfinite()

    def test_loss_is_independent_of_padding(self, lightning_model, num_atom_types, spatial_dimension):
        n_real = 5
        batch_size = 2
        max_atom_B = 10
        num_classes = num_atom_types + 1

        # Orthogonal lattice, above EGNN radial-cutoff clip threshold
        lattice_dim = spatial_dimension * (spatial_dimension + 1) // 2
        l0 = torch.zeros(batch_size, lattice_dim)
        l0[:, :spatial_dimension] = 8.0

        torch.manual_seed(42)
        x0 = torch.rand(batch_size, n_real, spatial_dimension)
        a0 = torch.randint(0, num_atom_types, (batch_size, n_real))
        xt = torch.rand(batch_size, n_real, spatial_dimension)
        at = torch.randint(0, num_atom_types, (batch_size, n_real))
        lt = l0.clone()

        sigma = torch.full((batch_size, 1), 0.1)
        time = torch.full((batch_size, 1), 0.5)
        time_indices = torch.tensor([5, 5])

        noise_scheduler = NoiseScheduler(NoiseParameters(total_time_steps=10), num_classes=num_classes)
        noise_sample = noise_scheduler.get_noise_from_indices(time_indices)
        q = broadcast_batch_matrix_tensor_to_all_dimensions(
            noise_sample.q_matrix, final_shape=(batch_size, n_real)
        )
        q_bar = broadcast_batch_matrix_tensor_to_all_dimensions(
            noise_sample.q_bar_matrix, final_shape=(batch_size, n_real)
        )
        q_bar_tm1 = broadcast_batch_matrix_tensor_to_all_dimensions(
            noise_sample.q_bar_tm1_matrix, final_shape=(batch_size, n_real)
        )

        # batch_A: 2 samples, each with 5 real atoms, no padding
        batch_A = {
            RELATIVE_COORDINATES: x0,
            ATOM_TYPES: a0,
            LATTICE_PARAMETERS: l0,
            NOISY_RELATIVE_COORDINATES: xt,
            NOISY_ATOM_TYPES: at,
            NOISY_LATTICE_PARAMETERS: lt,
            NOISE: sigma,
            TIME: time,
            TIME_INDICES: time_indices,
            Q_MATRICES: q,
            Q_BAR_MATRICES: q_bar,
            Q_BAR_TM1_MATRICES: q_bar_tm1,
            CARTESIAN_FORCES: torch.zeros(batch_size, n_real, spatial_dimension),
            "natom": torch.tensor([n_real, n_real]),
        }

        pad_size = max_atom_B - n_real
        q_pad = torch.zeros(batch_size, pad_size, num_classes, num_classes)
        # batch_B: same 2 samples, each with 5 real atoms + 5 padded atoms (NaN/PADDED_ATOM_TYPE)
        batch_B = {
            RELATIVE_COORDINATES: torch.cat(
                [x0, torch.full((batch_size, pad_size, spatial_dimension), float('nan'))], dim=1
            ),
            ATOM_TYPES: torch.cat(
                [a0, torch.full((batch_size, pad_size), PADDED_ATOM_TYPE, dtype=a0.dtype)], dim=1
            ),
            LATTICE_PARAMETERS: l0,
            NOISY_RELATIVE_COORDINATES: torch.cat(
                [xt, torch.full((batch_size, pad_size, spatial_dimension), float('nan'))], dim=1
            ),
            NOISY_ATOM_TYPES: torch.cat(
                [at, torch.full((batch_size, pad_size), PADDED_ATOM_TYPE, dtype=at.dtype)], dim=1
            ),
            NOISY_LATTICE_PARAMETERS: lt,
            NOISE: sigma,
            TIME: time,
            TIME_INDICES: time_indices,
            Q_MATRICES: torch.cat([q, q_pad], dim=1),
            Q_BAR_MATRICES: torch.cat([q_bar, q_pad], dim=1),
            Q_BAR_TM1_MATRICES: torch.cat([q_bar_tm1, q_pad], dim=1),
            CARTESIAN_FORCES: torch.zeros(batch_size, max_atom_B, spatial_dimension),
            "natom": torch.tensor([n_real, n_real]),
        }

        with torch.no_grad():
            output_A = lightning_model._generic_step(batch_A, batch_idx=0)
            output_B = lightning_model._generic_step(batch_B, batch_idx=0)

        torch.testing.assert_close(output_A["loss"], output_B["loss"])

    @pytest.fixture()
    def fake_datamodule(self, num_atom_types, spatial_dimension):
        return FakeVariableNatomAXLDataModule(
            spatial_dimension=spatial_dimension,
            num_atom_types=num_atom_types,
        )

    def test_smoke_test(self, lightning_model, fake_datamodule, accelerator):
        trainer = Trainer(fast_dev_run=3, accelerator=accelerator)
        trainer.fit(lightning_model, fake_datamodule)
        trainer.test(lightning_model, fake_datamodule)
