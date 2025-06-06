import numpy as np
import pytest

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.base_excisor import (
    NoOpEnvironmentExcision, NoOpEnvironmentExcisionArguments)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.base_sample_maker import (
    NoOpExciseSampleMaker, NoOpExciseSampleMakerArguments)
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    get_number_of_lattice_parameters


class TestBaseExciseSampleMaker:
    @pytest.fixture(params=[-1, 1, 2, 3])
    def max_constrained_substructure(self, request):
        return request.param

    @pytest.fixture(params=[1, 2, 3])
    def number_of_samples_per_substructure(self, request):
        return request.param

    @staticmethod
    def trivial_make_samples_from_constraints_method(substructure, num_samples):
        return [substructure] * num_samples, [{}] * num_samples

    @pytest.fixture(params=[1, 2, 3])
    def spatial_dimension(self, request):
        return request.param

    @pytest.fixture
    def excise_sample_maker_arguments(
        self,
        max_constrained_substructure,
        number_of_samples_per_substructure,
        spatial_dimension,
    ):
        return NoOpExciseSampleMakerArguments(
            element_list=["fire", "earth", "air", "water"],
            max_constrained_substructure=max_constrained_substructure,
            number_of_samples_per_substructure=number_of_samples_per_substructure,
            sample_box_size=[1.0] * spatial_dimension,
        )

    @pytest.fixture
    def environment_excisor(self, max_constrained_substructure):
        noop_excisor = NoOpEnvironmentExcision(
            NoOpEnvironmentExcisionArguments(
                excise_top_k_environment=max(1, max_constrained_substructure + 1)
            )
        )
        return noop_excisor

    @pytest.fixture
    def noop_base_excise_sample_maker(
        self, excise_sample_maker_arguments, environment_excisor
    ):
        sample_maker = NoOpExciseSampleMaker(
            excise_sample_maker_arguments, environment_excisor
        )
        sample_maker.make_samples_from_constrained_substructure = (
            self.trivial_make_samples_from_constraints_method
        )
        return sample_maker

    @pytest.fixture
    def number_of_atoms(self):
        return 20

    @pytest.fixture
    def structure_axl(self, number_of_atoms, spatial_dimension):
        atom_types = np.ones((number_of_atoms,))
        coordinates = np.random.rand(number_of_atoms, spatial_dimension)
        num_lattice_parameters = get_number_of_lattice_parameters(spatial_dimension)
        lattice_params = np.concatenate(
            [
                np.random.rand(spatial_dimension),
                np.zeros((num_lattice_parameters - spatial_dimension,)),
            ]
        )
        return AXL(A=atom_types, X=coordinates, L=lattice_params)

    @pytest.fixture
    def uncertainty_per_atom(self, number_of_atoms):
        return np.random.rand(number_of_atoms)

    @pytest.mark.parametrize("num_samples", [1, 4, 20])
    def test_make_samples(
        self,
        noop_base_excise_sample_maker,
        max_constrained_substructure,
        number_of_samples_per_substructure,
        structure_axl,
        uncertainty_per_atom,
        num_samples,
    ):
        made_samples, _ = noop_base_excise_sample_maker.make_samples(
            structure_axl, uncertainty_per_atom
        )
        assert (
            len(made_samples)
            == abs(max_constrained_substructure) * number_of_samples_per_substructure
        )
