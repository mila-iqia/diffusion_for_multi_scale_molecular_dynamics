import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.lammps_for_diffusion_data_module import \
    LammpsForDiffusionDataModule
from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.noising_transform import \
    NoisingTransform
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    NOISE, NOISY_RELATIVE_COORDINATES, Q_BAR_MATRICES, Q_BAR_TM1_MATRICES,
    Q_MATRICES, RELATIVE_COORDINATES, TIME, TIME_INDICES)
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
