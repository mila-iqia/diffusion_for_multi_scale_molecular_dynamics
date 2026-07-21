import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.lammps_for_diffusion_data_module import \
    LammpsForDiffusionDataModule
from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.noising_transform import \
    NoisingTransform
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    ATOM_TYPES, CARTESIAN_FORCES, LATTICE_PARAMETERS, NOISE, NOISY_ATOM_TYPES,
    NOISY_RELATIVE_COORDINATES, PADDED_ATOM_TYPE, Q_BAR_MATRICES,
    Q_BAR_TM1_MATRICES, Q_MATRICES, RELATIVE_COORDINATES, TIME, TIME_INDICES)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from tests.data.diffusion.conftest import TestLammpsForDiffusionDataModuleBase


class TestNoisingTransform(TestLammpsForDiffusionDataModuleBase):

    @pytest.fixture()
    def noising_transform(self, num_atom_types, spatial_dimension):
        return NoisingTransform(noise_parameters=NoiseParameters(total_time_steps=10),
                                num_atom_types=num_atom_types,
                                spatial_dimension=spatial_dimension)

    @pytest.fixture()
    def raw_batch(self, batched_input_data, element_types):
        return LammpsForDiffusionDataModule.dataset_transform(batched_input_data, element_types)

    def test_noising_transform(self, raw_batch, noising_transform, num_atom_types):
        batch_size, number_of_atoms, spatial_dimension = raw_batch[RELATIVE_COORDINATES].shape

        augmented_batch = noising_transform.transform(raw_batch)

        for key in raw_batch.keys():
            assert key in augmented_batch.keys()
            torch.testing.assert_allclose(augmented_batch[key], raw_batch[key])

        assert NOISY_RELATIVE_COORDINATES in augmented_batch.keys()
        assert augmented_batch[NOISY_RELATIVE_COORDINATES].shape == (batch_size, number_of_atoms, spatial_dimension)

        assert TIME in augmented_batch.keys()
        assert augmented_batch[TIME].shape == (batch_size, 1)

        assert NOISE in augmented_batch.keys()
        assert augmented_batch[NOISE].shape == (batch_size, 1)

        assert TIME_INDICES in augmented_batch.keys()
        assert augmented_batch[TIME_INDICES].shape == (batch_size, )

        for key in [Q_MATRICES, Q_BAR_MATRICES, Q_BAR_TM1_MATRICES]:
            assert key in augmented_batch.keys()
            assert augmented_batch[key].shape == (batch_size, number_of_atoms, num_atom_types + 1, num_atom_types + 1)


@pytest.mark.parametrize("spatial_dimension", [2, 3])
class TestNoisingTransformWithPadding:
    """Tests that NoisingTransform preserves NaN/PADDED_ATOM_TYPE sentinels for padded slots."""

    @pytest.fixture(scope="class", autouse=True)
    def set_random_seed(self):
        torch.manual_seed(12345678)

    @pytest.fixture()
    def num_atom_types(self):
        return 2

    @pytest.fixture()
    def batch_size(self):
        return 4

    @pytest.fixture()
    def max_atom(self):
        return 10

    @pytest.fixture()
    def natoms_list(self, max_atom):
        # One fully-occupied sample (no padding), others partially padded
        return [3, 5, max_atom, 7]

    @pytest.fixture()
    def padded_batch(self, batch_size, max_atom, num_atom_types, spatial_dimension, natoms_list):
        lattice_dim = spatial_dimension * (spatial_dimension + 1) // 2
        x0 = torch.rand(batch_size, max_atom, spatial_dimension)
        a0 = torch.randint(0, num_atom_types, (batch_size, max_atom))
        for b, n in enumerate(natoms_list):
            x0[b, n:, :] = float('nan')
            a0[b, n:] = PADDED_ATOM_TYPE
        return {
            RELATIVE_COORDINATES: x0,
            ATOM_TYPES: a0,
            LATTICE_PARAMETERS: torch.ones(batch_size, lattice_dim),
            CARTESIAN_FORCES: torch.zeros(batch_size, max_atom, spatial_dimension),
            "natom": torch.tensor(natoms_list),
        }

    @pytest.fixture()
    def noising_transform(self, num_atom_types, spatial_dimension):
        return NoisingTransform(
            noise_parameters=NoiseParameters(total_time_steps=10),
            num_atom_types=num_atom_types,
            spatial_dimension=spatial_dimension,
            use_optimal_transport=False,
        )

    @pytest.fixture()
    def transformed_batch(self, padded_batch, noising_transform):
        return noising_transform.transform(padded_batch)

    def test_padded_coordinates_stay_nan(self, transformed_batch, natoms_list):
        noisy_coords = transformed_batch[NOISY_RELATIVE_COORDINATES]
        for b, n in enumerate(natoms_list):
            assert not torch.any(torch.isnan(noisy_coords[b, :n, :]))
            if n < noisy_coords.shape[1]:
                assert torch.all(torch.isnan(noisy_coords[b, n:, :]))

    def test_padded_atom_types_stay_padded(self, transformed_batch, natoms_list):
        # atom types after noising: real atoms get diffused, padded should remain PADDED_ATOM_TYPE
        noisy_types = transformed_batch[NOISY_ATOM_TYPES]
        for b, n in enumerate(natoms_list):
            # Before the sample natoms must not contain the PADDED_ATOM_TYPE
            assert PADDED_ATOM_TYPE not in noisy_types[b, :n]
            if n < noisy_types.shape[1]:
                # After sample natoms must only contain PADDED_ATOM_TYPE
                assert torch.all(noisy_types[b, n:] == PADDED_ATOM_TYPE)
