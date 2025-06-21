from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.excise_and_repaint_sample_maker import (  # noqa
    ExciseAndRepaintSampleMaker, ExciseAndRepaintSampleMakerArguments)
from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.generators.sampling_constraint import \
    SamplingConstraint
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from tests.active_learning_loop.sample_maker.base_test_sample_maker import \
    BaseTestExciseSampleMaker


# TODO: a "fake" or "mock" ScoreNetwork should be introduced in order to make it possible to call
#   the "make_samples" method on the sample_maker; this is necessary to run the base class tests.
class TestExciseAndRepaintSampleMaker(BaseTestExciseSampleMaker):

    @pytest.fixture(scope="class", autouse=True)
    def set_random_seed(self):
        np.random.seed(42)

    @pytest.fixture
    def batch_size(self):
        return 3

    @pytest.fixture
    def number_of_atoms(self):
        return 14

    @pytest.fixture
    def batch_atom_types(self, batch_size, number_of_atoms, num_atom_types):
        return np.random.randint(0, num_atom_types, (batch_size, number_of_atoms))

    @pytest.fixture
    def batch_relative_coordinates(self, batch_size, number_of_atoms, spatial_dimension):
        return np.random.rand(batch_size, number_of_atoms, spatial_dimension)

    @pytest.fixture
    def batch_lattice_parameters(self, batch_size):
        return np.random.rand(batch_size, 6)

    @pytest.fixture
    def torch_batch_axl(self, batch_atom_types, batch_relative_coordinates, batch_lattice_parameters):
        return AXL(
            A=torch.tensor(batch_atom_types),
            X=torch.tensor(batch_relative_coordinates),
            L=torch.tensor(batch_lattice_parameters),
        )

    @pytest.fixture()
    def sample_maker_arguments(self, element_list, sample_box_strategy,
                               sample_box_size, number_of_samples_per_substructure):
        return ExciseAndRepaintSampleMakerArguments(
            element_list=element_list,
            sample_box_strategy=sample_box_strategy,
            sample_box_size=sample_box_size,
            number_of_samples_per_substructure=number_of_samples_per_substructure)

    @pytest.fixture
    def noise_parameters(self):
        return NoiseParameters(
            total_time_steps=2,
            schedule_type="exponential",
            sigma_min=0.01,
            sigma_max=0.5,
        )

    @pytest.fixture
    def cell_dimensions_for_samping(self, sample_box_strategy, sample_box_size, basis_vectors):
        if sample_box_size is None:
            return np.diag(basis_vectors)
        else:
            return sample_box_size

    @pytest.fixture
    def sampling_parameters(
        self,
        spatial_dimension,
        number_of_atoms,
        num_atom_types,
        cell_dimensions_for_samping,
    ):
        return PredictorCorrectorSamplingParameters(
            number_of_samples=1,
            spatial_dimension=spatial_dimension,
            number_of_corrector_steps=1,
            num_atom_types=num_atom_types,
            number_of_atoms=number_of_atoms,
            use_fixed_lattice_parameters=True,
            cell_dimensions=cell_dimensions_for_samping,
            record_samples=False,
        )

    @pytest.fixture
    def axl_score_network(self):
        return None

    @pytest.fixture
    def sample_maker(
        self,
        sample_maker_arguments,
        atom_selector,
        excisor,
        noise_parameters,
        sampling_parameters,
        axl_score_network,
    ):
        return ExciseAndRepaintSampleMaker(
            sample_maker_arguments=sample_maker_arguments,
            atom_selector=atom_selector,
            environment_excisor=excisor,
            noise_parameters=noise_parameters,
            sampling_parameters=sampling_parameters,
            diffusion_model=axl_score_network,
        )

    def test_torch_batch_axl_to_list_of_numpy_axl(
        self,
        batch_size,
        batch_atom_types,
        batch_relative_coordinates,
        batch_lattice_parameters,
        torch_batch_axl,
        sample_maker,
    ):
        calculated_list_of_numpy_axl = (
            sample_maker.torch_batch_axl_to_list_of_numpy_axl(
                torch_batch_axl
            )
        )
        assert len(calculated_list_of_numpy_axl) == batch_size

        for b in range(batch_size):
            current_numpy_axl = calculated_list_of_numpy_axl[b]
            assert np.array_equal(current_numpy_axl.A, batch_atom_types[b, :])
            assert np.allclose(current_numpy_axl.X, batch_relative_coordinates[b, :, :])
            assert np.allclose(current_numpy_axl.L, batch_lattice_parameters[b, :])

    def test_create_sampling_constraints(self, structure_axl, element_list, sample_maker):

        calculated_sampling_constraint = sample_maker.create_sampling_constraints(structure_axl)

        expected_sampling_constraint = SamplingConstraint(
            elements=element_list,
            constrained_relative_coordinates=torch.FloatTensor(structure_axl.X),
            constrained_atom_types=torch.LongTensor(structure_axl.A),
        )

        assert all(
            [
                x == y
                for x, y in zip(
                    calculated_sampling_constraint.elements,
                    expected_sampling_constraint.elements,
                )
            ]
        )
        assert torch.allclose(
            calculated_sampling_constraint.constrained_relative_coordinates,
            expected_sampling_constraint.constrained_relative_coordinates,
        )
        assert torch.allclose(
            calculated_sampling_constraint.constrained_atom_types,
            expected_sampling_constraint.constrained_atom_types,
        )

    def test_make_samples_from_constrained_substructure(
        self,
        structure_axl,
        batch_atom_types,
        batch_relative_coordinates,
        batch_lattice_parameters,
        torch_batch_axl,
        batch_size,
        sample_maker,
    ):

        mock_create_batch_of_samples = MagicMock(
            return_value={"original_axl": torch_batch_axl}
        )
        with (((patch(
            "diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker."
            "excise_and_repaint_sample_maker.create_batch_of_samples",
            new=mock_create_batch_of_samples,
        )))):
            # The inputs don't matter since we force the output with a mock.
            calculated_new_samples, _, _ = sample_maker.make_samples_from_constrained_substructure(
                constrained_structure=structure_axl,
                active_atom_index=0)

        mock_create_batch_of_samples.assert_called_once()
        assert len(calculated_new_samples) == batch_size
        for b in range(batch_size):
            assert np.array_equal(calculated_new_samples[b].A, batch_atom_types[b, :])
            assert np.array_equal(
                calculated_new_samples[b].X, batch_relative_coordinates[b, :, :]
            )
            assert np.array_equal(calculated_new_samples[b].L, batch_lattice_parameters[b, :])

    @pytest.fixture()
    def samples_and_indices(self):
        # TODO: Remove this stub once we have a proper way of drawing fake samples.
        return None

    @pytest.fixture()
    def calculated_pymatgen_sample_structures_and_indices(self):
        # TODO: Remove this stub once we have a proper way of drawing fake samples.
        return None

    @pytest.fixture()
    def reference_pymatgen_excised_substructures_and_indices(self):
        # TODO: Remove this stub once we have a proper way of drawing fake samples.
        return None

    def test_sample_lattice_parameters(self, samples_and_indices, sample_box_strategy,
                                       sample_box_size, lattice_parameters):
        # TODO: Remove this stub once we have a proper way of drawing fake samples.
        pytest.skip("Skipping this test until a fake make_sample method is implemented.")

    def test_excised_environments_are_present(
            self,
            calculated_pymatgen_sample_structures_and_indices,
            reference_pymatgen_excised_substructures_and_indices,
            number_of_samples_per_substructure
    ):
        # TODO: Remove this stub once we have a proper way of drawing fake samples.
        pytest.skip("Skipping this test until a fake make_sample method is implemented.")
