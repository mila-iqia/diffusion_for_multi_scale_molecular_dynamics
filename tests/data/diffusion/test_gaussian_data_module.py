import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.gaussian_data_module import (
    GaussianDataModule, GaussianDataModuleParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import \
    RELATIVE_COORDINATES


class TestGaussianDataModule:

    @pytest.fixture()
    def batch_size(self):
        return 4

    @pytest.fixture()
    def train_dataset_size(self):
        return 16

    @pytest.fixture()
    def valid_dataset_size(self):
        return 8

    @pytest.fixture()
    def number_of_atoms(self):
        return 4

    @pytest.fixture()
    def spatial_dimension(self):
        return 2

    @pytest.fixture()
    def sigma_d(self):
        return 0.01

    @pytest.fixture()
    def equilibrium_relative_coordinates(self, number_of_atoms, spatial_dimension):
        list_x = torch.rand(number_of_atoms, spatial_dimension)
        equilibrium_relative_coordinates = [list(x) for x in list_x.numpy()]
        return equilibrium_relative_coordinates

    @pytest.fixture
    def data_module_hyperparameters(self, batch_size, train_dataset_size, valid_dataset_size,
                                    number_of_atoms, spatial_dimension, sigma_d, equilibrium_relative_coordinates):
        return GaussianDataModuleParameters(
            batch_size=batch_size,
            random_seed=42,
            num_workers=0,
            sigma_d=sigma_d,
            number_of_atoms=number_of_atoms,
            spatial_dimension=spatial_dimension,
            equilibrium_relative_coordinates=equilibrium_relative_coordinates,
            train_dataset_size=train_dataset_size,
            valid_dataset_size=valid_dataset_size,
            elements=['DUMMY']
        )

    @pytest.fixture()
    def data_module(self, data_module_hyperparameters):

        data_module = GaussianDataModule(hyper_params=data_module_hyperparameters)
        data_module.setup()

        return data_module

    def test_train_data_loader(self, data_module, train_dataset_size, number_of_atoms, spatial_dimension):
        self._check_data_loader(data_module.train_dataloader(), number_of_atoms, spatial_dimension, train_dataset_size)

    def test_validation_data_loader(self, data_module, valid_dataset_size, number_of_atoms, spatial_dimension):
        self._check_data_loader(data_module.val_dataloader(), number_of_atoms, spatial_dimension, valid_dataset_size)

    def _check_data_loader(self, dataloader, number_of_atoms, spatial_dimension, dataset_size):
        count = 0
        for batch in dataloader:
            x = batch[RELATIVE_COORDINATES]
            assert torch.all(x >= 0.)
            assert torch.all(x < 1.)
            batch_size, natoms, space = x.shape
            count += batch_size
            assert natoms == number_of_atoms
            assert space == spatial_dimension

        assert count == dataset_size
