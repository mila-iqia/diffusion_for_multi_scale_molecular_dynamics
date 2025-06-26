import pytest

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.no_op_excisor import (
    NoOpExcision, NoOpExcisionArguments)
from tests.active_learning_loop.excisor.base_test_excision import \
    BaseTestExcision


class TestNoOpExcision(BaseTestExcision):

    @pytest.fixture()
    def excisor(self):
        return NoOpExcision(NoOpExcisionArguments())

    @pytest.fixture
    def expected_excised_environment(self, structure_axl):
        return structure_axl

    @pytest.fixture
    def expected_excised_atom_index(self, central_atom_idx):
        return central_atom_idx
