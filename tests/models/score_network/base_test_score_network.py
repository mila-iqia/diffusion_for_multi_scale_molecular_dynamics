import pytest
import torch


class BaseTestScoreNetwork:
    """Base class defining common fixtures for all tests."""

    @pytest.fixture(scope="class", autouse=True)
    def set_seed(self):
        """Set the random seed."""
        torch.manual_seed(234233)

    @pytest.fixture()
    def score_network(self, *args):
        raise NotImplementedError("This fixture must be implemented in the derived class.")

    @pytest.fixture()
    def batch_size(self, *args, **kwargs):
        return 16

    @pytest.fixture()
    def number_of_atoms(self):
        return 8

    @pytest.fixture()
    def spatial_dimension(self):
        return 3

    @pytest.fixture()
    def num_atom_types(self):
        return 5

    @pytest.fixture()
    def atom_types(self, batch_size, number_of_atoms, num_atom_types):
        atom_types = torch.randint(0, num_atom_types + 1, (batch_size, number_of_atoms))
        return atom_types
