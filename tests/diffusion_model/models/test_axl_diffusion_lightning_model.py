import dataclasses
import math
import os
import pickle
import sys

import ase
import ase.io
import numpy as np
import pytest
import torch
from ase.calculators.singlepoint import SinglePointCalculator
from lightning import LightningDataModule, Trainer
from torch.utils.data import DataLoader, default_collate, random_split

from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.data_module.diffusion.noising_transform import \
    NoisingTransform
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.loss.loss_parameters import (
    AtomTypeLossParameters, MSELossParameters, create_loss_parameters)
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.models.axl_diffusion_lightning_model import (
    AXLDiffusionLightningModel, AXLDiffusionParameters)
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.models.optimizer import \
    OptimizerParameters
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.models.scheduler import (
    CosineAnnealingLRSchedulerParameters, ReduceLROnPlateauSchedulerParameters)
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.models.target_score_matching import (
    TargetScoreMatchingParameters, get_one_sided_squared_error,
    read_bad_frames)
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.noise_schedulers.noise_scheduler import \
    NoiseScheduler
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.noisers.lattice_noiser import \
    LatticeDataParameters
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    ATOM_TYPES, AXL, AXL_COMPOSITION, CARTESIAN_FORCES, LATTICE_PARAMETERS,
    NOISE, NOISY_ATOM_TYPES, NOISY_AXL_COMPOSITION, NOISY_LATTICE_PARAMETERS,
    NOISY_RELATIVE_COORDINATES, NUMBER_OF_ATOMS, PADDED_ATOM_TYPE,
    Q_BAR_MATRICES, Q_BAR_TM1_MATRICES, Q_MATRICES, RELATIVE_COORDINATES, TIME,
    TIME_INDICES)
from diffusion_for_multi_scale_molecular_dynamics.oracle.base_single_point_calculator import (
    BaseSinglePointCalculator, SinglePointCalculation)
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.generator.predictor_corrector_axl_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.metrics.sampling_metrics_parameters import \
    SamplingMetricsParameters
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.sampling.diffusion_sampling_parameters import \
    DiffusionSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.score_network.cartesian_egnn_score_network import \
    CartesianEGNNScoreNetworkParameters
from diffusion_for_multi_scale_molecular_dynamics.score_network.mlp_score_network import \
    MLPScoreNetworkParameters
from diffusion_for_multi_scale_molecular_dynamics.score_network.target_scores.wrapped_gaussian_score import (
    get_coordinates_sigma_normalized_score_cartesian,
    get_sigma_normalized_score_brute_force)
from diffusion_for_multi_scale_molecular_dynamics.score_network.torus_egnn_score_network import \
    TorusEGNNScoreNetworkParameters
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from diffusion_for_multi_scale_molecular_dynamics.utils.tensor_utils import (
    broadcast_batch_matrix_tensor_to_all_dimensions,
    broadcast_batch_tensor_to_all_dimensions)


class FakeSinglePointCalculator(BaseSinglePointCalculator):
    """A single-point calculator that returns random energies and forces (no LAMMPS)."""

    def __init__(self):
        pass

    def calculate(self, atoms, results_path=None) -> SinglePointCalculation:
        return SinglePointCalculation(
            calculation_type="fake",
            atoms=atoms,
            forces=np.random.rand(len(atoms), 3),
            energy=float(np.random.rand()),
        )


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
        # Real chemical symbols: the energy path builds pymatgen structures, which reject non-elements.
        return ["Si", "Ge", "C", "O"][:num_atom_types]

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
    def diffusion_sampling_parameters(self, sampling_parameters, spatial_dimension):
        noise_parameters = NoiseParameters(total_time_steps=5)
        metrics_parameters = SamplingMetricsParameters(
            structure_factor_max_distance=1.0,
            # The energy oracle builds pymatgen structures, so it is only meaningful in 3D.
            compute_energies=(spatial_dimension == 3),
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

        hyper_params = AXLDiffusionParameters(
            score_network_parameters=score_network_parameters,
            optimizer_parameters=optimizer_parameters,
            scheduler_parameters=scheduler_parameters,
            loss_parameters=loss_parameters,
            diffusion_sampling_parameters=diffusion_sampling_parameters,
            single_point_calculator_configuration={"name": "stillinger_weber", "sw_coeff_filename": "unused"},
            elements=unique_elements,
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
        # The model builds its single-point calculator through the factory; swap in a fake (no LAMMPS).
        mocker.patch(
            "diffusion_for_multi_scale_molecular_dynamics.diffusion_model.models."
            "axl_diffusion_lightning_model.instantiate_single_point_calculator",
            return_value=FakeSinglePointCalculator(),
        )
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

    @pytest.fixture(params=[CartesianEGNNScoreNetworkParameters, TorusEGNNScoreNetworkParameters],
                    ids=["cartesian", "torus"])
    def egnn_score_network_parameters_class(self, request):
        return request.param

    @pytest.fixture()
    def hyper_params(self, num_atom_types, spatial_dimension, loss_parameters, optimizer_parameters,
                     egnn_score_network_parameters_class):
        score_network_parameters = egnn_score_network_parameters_class(
            num_atom_types=num_atom_types,
            spatial_dimension=spatial_dimension,
            radial_cutoff=3.0,
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


@pytest.mark.parametrize(
    "score_network_parameters",
    [
        CartesianEGNNScoreNetworkParameters(num_atom_types=1, radial_cutoff=3.0, n_layers=2),
        TorusEGNNScoreNetworkParameters(num_atom_types=1, radial_cutoff=3.0, n_layers=2),
        TorusEGNNScoreNetworkParameters(num_atom_types=1, edges="fully_connected", radial_cutoff=None,
                                        smooth_cutoff=False, n_layers=2),
    ],
    ids=["cartesian", "torus_radial_cutoff", "torus_fully_connected"],
)
class TestEGNNCheckpointRoundTrip:
    """A saved checkpoint loads back the same EGNN score network, with the same outputs."""

    @pytest.fixture()
    def lightning_model(self, score_network_parameters):
        hyper_params = AXLDiffusionParameters(
            score_network_parameters=score_network_parameters,
            loss_parameters=create_loss_parameters(model_dictionary={}),
            optimizer_parameters=OptimizerParameters(name="adam", learning_rate=0.001),
        )
        return AXLDiffusionLightningModel(hyper_params).eval()

    @pytest.fixture()
    def batch(self):
        torch.manual_seed(1234)
        batch_size, number_of_atoms = 2, 8
        return {
            NOISY_AXL_COMPOSITION: AXL(A=torch.zeros(batch_size, number_of_atoms, dtype=torch.long),
                                       X=torch.rand(batch_size, number_of_atoms, 3),
                                       L=torch.tensor([[5.43, 5.43, 5.43, 0.0, 0.0, 0.0]] * batch_size)),
            NOISE: torch.tensor([[0.05], [0.5]]),
            TIME: torch.rand(batch_size, 1),
            NUMBER_OF_ATOMS: torch.full((batch_size,), number_of_atoms),
        }

    def test_checkpoint_round_trip(self, lightning_model, batch, score_network_parameters, tmp_path):
        trainer = Trainer(accelerator="cpu", logger=False, enable_checkpointing=False)
        trainer.strategy.connect(lightning_model)
        checkpoint_path = tmp_path / "model.ckpt"
        trainer.save_checkpoint(checkpoint_path)

        loaded_model = AXLDiffusionLightningModel.load_from_checkpoint(checkpoint_path, map_location="cpu").eval()
        assert type(loaded_model.axl_network) is type(lightning_model.axl_network)
        assert loaded_model.hyper_params.score_network_parameters == score_network_parameters

        def copy(batch):
            composition = batch[NOISY_AXL_COMPOSITION]
            return {**batch, NOISY_AXL_COMPOSITION: composition._replace(L=composition.L.clone())}

        with torch.no_grad():
            scores = lightning_model.axl_network(copy(batch), conditional=False)
            loaded_scores = loaded_model.axl_network(copy(batch), conditional=False)
        torch.testing.assert_close(loaded_scores.X, scores.X)
        torch.testing.assert_close(loaded_scores.A, scores.A)


TARGET_SCORE_MATCHING_CELL_LENGTH = 5.43
AXL_DIFFUSION_LIGHTNING_MODEL_MODULE = sys.modules[AXLDiffusionLightningModel.__module__]


def write_bad_frames(bad_frames_path, number_of_frames, number_of_atoms=4):
    """Write bad frames whose first atom has a force above any maximum_force used in these tests."""
    with ase.io.Trajectory(str(bad_frames_path), "w") as trajectory:
        for seed in range(number_of_frames):
            rng = np.random.default_rng(seed)
            atoms = ase.Atoms(symbols=["Si"] * number_of_atoms,
                              positions=rng.random((number_of_atoms, 3)) * TARGET_SCORE_MATCHING_CELL_LENGTH,
                              cell=np.diag([TARGET_SCORE_MATCHING_CELL_LENGTH] * 3), pbc=True)
            forces = rng.normal(size=(number_of_atoms, 3))
            forces[0] = [100.0, 0.0, 0.0]
            atoms.calc = SinglePointCalculator(atoms, energy=0.0, forces=forces)
            trajectory.write(atoms)


class TestTargetScoreMatchingTraining:
    """Target score matching in the lightning model: the training target and loss, and the bad frames."""

    @pytest.fixture(autouse=True)
    def set_random_seed(self):
        torch.manual_seed(2024)

    @pytest.fixture()
    def number_of_atoms(self):
        return 4

    @pytest.fixture()
    def batch_size(self):
        return 3

    @pytest.fixture()
    def bad_frames_path(self, tmp_path):
        return tmp_path / "bad_frames.traj"

    @pytest.fixture()
    def target_score_matching_parameters(self, bad_frames_path):
        return TargetScoreMatchingParameters(
            temperature=300.0, thermal_width={"Si": 0.044}, minimum_sigma=0.01, maximum_sigma=0.044,
            bad_frames_path=bad_frames_path, maximum_force=5.0, bad_frames_lambda_weight=1.0,
        )

    @pytest.fixture(params=[CartesianEGNNScoreNetworkParameters, TorusEGNNScoreNetworkParameters],
                    ids=["cartesian", "torus"])
    def egnn_score_network_parameters_class(self, request):
        return request.param

    @pytest.fixture()
    def hyper_params(self, target_score_matching_parameters, egnn_score_network_parameters_class):
        return AXLDiffusionParameters(
            score_network_parameters=egnn_score_network_parameters_class(
                num_atom_types=1, radial_cutoff=3.0, n_layers=1, message_hidden_dimensions_size=4,
                node_hidden_dimensions_size=4, coordinate_hidden_dimensions_size=4,
            ),
            loss_parameters=AXL(A=AtomTypeLossParameters(lambda_weight=0.0), X=MSELossParameters(lambda_weight=1.0),
                                L=MSELossParameters(lambda_weight=0.0)),
            optimizer_parameters=OptimizerParameters(name="adamw", learning_rate=1e-3),
            elements=["Si"],
            target_score_matching_parameters=target_score_matching_parameters,
        )

    @pytest.fixture()
    def lightning_model(self, hyper_params):
        return AXLDiffusionLightningModel(hyper_params)

    @pytest.fixture()
    def batch(self, batch_size, number_of_atoms):
        """A noised training batch with forces; the last configuration has one padded atom."""
        noising_transform = NoisingTransform(
            noise_parameters=NoiseParameters(total_time_steps=10, sigma_min_cart=0.01, sigma_max_cart=1.0),
            num_atom_types=1, spatial_dimension=3, use_fixed_lattice_parameters=True, use_optimal_transport=False,
        )
        raw_data = []
        for sample_index in range(batch_size):
            number_of_real_atoms = number_of_atoms - 1 if sample_index == batch_size - 1 else number_of_atoms
            relative_coordinates = torch.rand(number_of_atoms, 3)
            atom_types = torch.zeros(number_of_atoms, dtype=torch.long)
            forces = torch.randn(number_of_atoms, 3)
            relative_coordinates[number_of_real_atoms:] = torch.nan
            atom_types[number_of_real_atoms:] = PADDED_ATOM_TYPE
            forces[number_of_real_atoms:] = torch.nan
            raw_data.append({
                RELATIVE_COORDINATES: relative_coordinates,
                ATOM_TYPES: atom_types,
                LATTICE_PARAMETERS: torch.tensor([TARGET_SCORE_MATCHING_CELL_LENGTH] * 3 + [0.0] * 3),
                CARTESIAN_FORCES: forces,
                "natom": torch.tensor(number_of_real_atoms),
            })
        return noising_transform.transform(default_collate(raw_data))

    @pytest.fixture()
    def expected_target(self, batch, target_score_matching_parameters):
        """kappa sigma beta F + (1 - kappa) target_dsm, computed independently from the batch."""
        pad_mask = batch[ATOM_TYPES] == PADDED_ATOM_TYPE
        x0 = batch[RELATIVE_COORDINATES].masked_fill(pad_mask.unsqueeze(-1), 0.0)
        xt = batch[NOISY_RELATIVE_COORDINATES].masked_fill(pad_mask.unsqueeze(-1), 0.0)
        forces = batch[CARTESIAN_FORCES].masked_fill(pad_mask.unsqueeze(-1), 0.0)
        sigma = batch[NOISE].squeeze(-1)
        lattice_diagonals = batch[LATTICE_PARAMETERS][:, :3]
        denoising_target = get_coordinates_sigma_normalized_score_cartesian(
            map_relative_coordinates_to_unit_cell(xt - x0), sigma, lattice_diagonals, kmax=4
        )
        kappa = (0.044**2 / (0.044**2 + sigma**2))[:, None, None]
        force_target = sigma[:, None, None] * target_score_matching_parameters.beta * forces
        return kappa * force_target + (1.0 - kappa) * denoising_target

    @pytest.fixture()
    def recorded_network_inputs(self, mocker, lightning_model):
        """Replace the network by a fixed prediction of 0.3, recording the batches it receives."""
        network_inputs = []

        def fake_forward(batch, conditional=None):
            composition = batch[NOISY_AXL_COMPOSITION]
            network_inputs.append(batch)
            return AXL(A=torch.zeros(*composition.A.shape, 2), X=torch.full_like(composition.X, 0.3),
                       L=torch.zeros_like(composition.L))

        mocker.patch.object(lightning_model.axl_network, "forward", side_effect=fake_forward)
        return network_inputs

    @pytest.fixture()
    def lightning_model_with_bad_frames(self, lightning_model, bad_frames_path):
        write_bad_frames(bad_frames_path, number_of_frames=3)
        lightning_model._read_bad_frames_if_modified()
        return lightning_model

    def test_elements_are_required(self, hyper_params):
        with pytest.raises(AssertionError):
            AXLDiffusionLightningModel(dataclasses.replace(hyper_params, elements=None))

    def test_thermal_widths_buffer(self, lightning_model):
        torch.testing.assert_close(lightning_model.thermal_widths_by_atom_type, torch.tensor([0.044]))
        assert "thermal_widths_by_atom_type" not in lightning_model.state_dict()
        lightning_model.to(torch.float64)
        assert lightning_model.thermal_widths_by_atom_type.dtype == torch.float64

    def test_parameters_pickled_without_target_score_matching_load_as_none(self, hyper_params):
        old_hyper_params = dataclasses.replace(hyper_params, target_score_matching_parameters=None)
        del old_hyper_params.__dict__["target_score_matching_parameters"]
        restored_hyper_params = pickle.loads(pickle.dumps(old_hyper_params))
        assert restored_hyper_params.target_score_matching_parameters is None

    def test_training_target(self, lightning_model, batch, expected_target):
        output = lightning_model._generic_step(batch, batch_idx=0)
        target = output["target_coordinates_normalized_conditional_scores"]
        torch.testing.assert_close(target, expected_target)
        pad_mask = batch[ATOM_TYPES] == PADDED_ATOM_TYPE
        torch.testing.assert_close(target[pad_mask], torch.zeros_like(target[pad_mask]))

    def test_training_loss(self, lightning_model, batch):
        output = lightning_model._generic_step(batch, batch_idx=0)
        target = output["target_coordinates_normalized_conditional_scores"]
        predictions = output["model_predictions"].X
        torch.testing.assert_close(output["unreduced_loss"].X, (predictions - target) ** 2)

        pad_mask = batch[ATOM_TYPES] == PADDED_ATOM_TYPE
        squared_error = ((predictions - target) ** 2).masked_fill(pad_mask.unsqueeze(-1), 0.0)
        expected_loss = (squared_error.sum(dim=(-2, -1)) / (batch["natom"] * 3)).mean()
        torch.testing.assert_close(output["loss"], expected_loss)

    def test_batch_without_forces_raises(self, lightning_model, batch):
        del batch[CARTESIAN_FORCES]
        with pytest.raises(AssertionError):
            lightning_model._generic_step(batch, batch_idx=0)

    def test_bad_frames_are_read_again_only_when_modified(self, mocker, lightning_model, bad_frames_path):
        read_spy = mocker.patch.object(AXL_DIFFUSION_LIGHTNING_MODEL_MODULE, "read_bad_frames",
                                       wraps=read_bad_frames)
        lightning_model._read_bad_frames_if_modified()
        assert lightning_model.bad_frames is None

        write_bad_frames(bad_frames_path, number_of_frames=2)
        lightning_model._read_bad_frames_if_modified()
        lightning_model._read_bad_frames_if_modified()
        assert read_spy.call_count == 1
        assert len(lightning_model.bad_frames["forces"]) == 2

        write_bad_frames(bad_frames_path, number_of_frames=3)
        later_time = bad_frames_path.stat().st_mtime + 10.0
        os.utime(bad_frames_path, (later_time, later_time))
        lightning_model._read_bad_frames_if_modified()
        assert read_spy.call_count == 2
        assert len(lightning_model.bad_frames["forces"]) == 3

    @pytest.mark.parametrize(
        "number_of_training_batches, expected_number_of_bad_frames_per_step",
        [(1, [3, 0]), (2, [2, 1, 0]), (5, [1, 1, 1, 0, 0])],
    )
    def test_each_bad_frame_is_used_once_per_epoch(self, lightning_model_with_bad_frames, number_of_training_batches,
                                                   expected_number_of_bad_frames_per_step):
        lightning_model_with_bad_frames._start_bad_frames_epoch(number_of_training_batches)
        bad_frame_indices = [lightning_model_with_bad_frames._get_next_bad_frame_indices()
                             for _ in expected_number_of_bad_frames_per_step]
        assert [len(indices) for indices in bad_frame_indices] == expected_number_of_bad_frames_per_step
        torch.testing.assert_close(torch.cat(bad_frame_indices).sort().values, torch.arange(3))

    def test_no_bad_frames_gives_no_epoch_order(self, lightning_model):
        lightning_model._read_bad_frames_if_modified()
        lightning_model._start_bad_frames_epoch(number_of_training_batches=2)
        assert lightning_model.bad_frames_order is None

    def test_bad_frames_loss_scaling(self, lightning_model_with_bad_frames, recorded_network_inputs):
        bad_frame_indices = torch.arange(3)
        torch.manual_seed(7)
        loss = lightning_model_with_bad_frames._compute_bad_frames_loss(bad_frame_indices,
                                                                        number_of_training_configurations=8)
        assert torch.isfinite(loss)

        torch.manual_seed(7)
        loss_with_half_the_configurations = lightning_model_with_bad_frames._compute_bad_frames_loss(
            bad_frame_indices, number_of_training_configurations=4)
        torch.testing.assert_close(loss_with_half_the_configurations, 2.0 * loss)

        lightning_model_with_bad_frames.target_score_matching_parameters.bad_frames_lambda_weight = 3.0
        torch.manual_seed(7)
        weighted_loss = lightning_model_with_bad_frames._compute_bad_frames_loss(bad_frame_indices,
                                                                                 number_of_training_configurations=8)
        torch.testing.assert_close(weighted_loss, 3.0 * loss)

    def test_bad_frames_loss_is_zero_for_an_exact_prediction(self, mocker, lightning_model_with_bad_frames):
        target_spy = mocker.spy(AXL_DIFFUSION_LIGHTNING_MODEL_MODULE, "get_target_score_matching_target")

        def exact_forward(batch, conditional=None):
            composition = batch[NOISY_AXL_COMPOSITION]
            return AXL(A=torch.zeros(*composition.A.shape, 2), X=target_spy.spy_return,
                       L=torch.zeros_like(composition.L))

        mocker.patch.object(lightning_model_with_bad_frames.axl_network, "forward", side_effect=exact_forward)
        loss = lightning_model_with_bad_frames._compute_bad_frames_loss(torch.arange(3),
                                                                        number_of_training_configurations=8)
        torch.testing.assert_close(loss, torch.tensor(0.0))

    def test_bad_frames_loss_value(self, lightning_model_with_bad_frames, recorded_network_inputs,
                                   target_score_matching_parameters):
        """The network sees the unnoised bad frames with a noise level sigma, and the target is kappa sigma beta F."""
        bad_frame_indices = torch.tensor([2, 0])
        torch.manual_seed(11)
        loss = lightning_model_with_bad_frames._compute_bad_frames_loss(bad_frame_indices,
                                                                        number_of_training_configurations=8)

        bad_frames = lightning_model_with_bad_frames.bad_frames
        torch.manual_seed(11)
        sigma = torch.exp(torch.empty(2).uniform_(math.log(0.01), math.log(0.044)))
        kappa = (0.044**2 / (0.044**2 + sigma**2))[:, None, None]
        forces = bad_frames["forces"][bad_frame_indices]
        target = kappa * sigma[:, None, None] * target_score_matching_parameters.beta * forces
        saturated_atoms = bad_frames["saturated_atoms"][bad_frame_indices]
        assert saturated_atoms[:, 0].all()
        error = get_one_sided_squared_error(torch.full_like(target, 0.3), target, saturated_atoms)
        expected_loss = (error.mean(dim=-1) / 3).sum() / 8
        torch.testing.assert_close(loss, expected_loss)

        [network_input] = recorded_network_inputs
        torch.testing.assert_close(network_input[NOISY_AXL_COMPOSITION].X,
                                   bad_frames["composition"].X[bad_frame_indices])
        torch.testing.assert_close(network_input[NOISE], sigma[:, None])
