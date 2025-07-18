from unittest.mock import MagicMock, patch

import einops
import numpy as np
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.excise_and_repaint_sample_maker import (  # noqa
    ExciseAndRepaintSampleMaker, ExciseAndRepaintSampleMakerArguments)
from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.generators.sampling_constraint import \
    SamplingConstraint
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.egnn_score_network import (
    EGNNScoreNetwork, EGNNScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import (
    get_reciprocal_basis_vectors,
    get_relative_coordinates_from_cartesian_positions,
    map_relative_coordinates_to_unit_cell)
from tests.active_learning_loop.sample_maker.base_test_sample_maker import \
    BaseTestExciseSampleMaker


class TestExciseAndRepaintSampleMaker(BaseTestExciseSampleMaker):

    @pytest.fixture(scope="class", autouse=True)
    def set_random_seed(self):
        np.random.seed(42)

    @pytest.fixture
    def batch_size(self):
        return 3

    @pytest.fixture
    def score_network_parameters(self, num_atom_types):
        return EGNNScoreNetworkParameters(
            number_of_bloch_wave_shells=1,
            edges="radial_cutoff",
            radial_cutoff=5.0,
            num_atom_types=num_atom_types,
            normalize=False,
        )

    @pytest.fixture()
    def axl_score_network(self, score_network_parameters):
        score_network = EGNNScoreNetwork(score_network_parameters)
        return score_network

    @pytest.fixture
    def batch_atom_types(self, batch_size, number_of_atoms, num_atom_types):
        return np.random.randint(0, num_atom_types, (batch_size, number_of_atoms))

    @pytest.fixture
    def batch_relative_coordinates(
        self, batch_size, number_of_atoms, spatial_dimension
    ):
        return np.random.rand(batch_size, number_of_atoms, spatial_dimension)

    @pytest.fixture
    def batch_lattice_parameters(self, batch_size):
        batch_lattice_parameters = np.zeros((batch_size, 6))
        batch_lattice_parameters[:, :3] = 10 + 10 * np.random.rand(batch_size, 3)
        return batch_lattice_parameters

    @pytest.fixture
    def torch_batch_axl(
        self, batch_atom_types, batch_relative_coordinates, batch_lattice_parameters
    ):
        return AXL(
            A=torch.tensor(batch_atom_types),
            X=torch.tensor(batch_relative_coordinates),
            L=torch.tensor(batch_lattice_parameters),
        )

    @pytest.fixture(params=[True, False])
    def sample_edit_radius(self, request, radial_cutoff):
        if request.param:
            return radial_cutoff
        else:
            return None

    @pytest.fixture()
    def sample_maker_arguments(
        self,
        element_list,
        sample_box_strategy,
        sample_edit_radius,
        sample_box_size,
        number_of_samples_per_substructure,
    ):
        return ExciseAndRepaintSampleMakerArguments(
            element_list=element_list,
            sample_edit_radius=sample_edit_radius,
            sample_box_strategy=sample_box_strategy,
            sample_box_size=sample_box_size,
            number_of_samples_per_substructure=number_of_samples_per_substructure,
        )

    @pytest.fixture
    def noise_parameters(self):
        return NoiseParameters(
            total_time_steps=5,
            schedule_type="exponential",
            sigma_min=0.01,
            sigma_max=0.5,
        )

    @pytest.fixture
    def cell_dimensions_for_samping(
        self, sample_box_strategy, sample_box_size, basis_vectors
    ):
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
        number_of_samples_per_substructure,
        cell_dimensions_for_samping,
    ):
        # The EGNN model works with float32: the code crashes if we mix float32 and float64.
        cell_dimensions = [np.float32(c) for c in cell_dimensions_for_samping]
        return PredictorCorrectorSamplingParameters(
            number_of_samples=number_of_samples_per_substructure,
            spatial_dimension=spatial_dimension,
            number_of_corrector_steps=1,
            num_atom_types=num_atom_types,
            number_of_atoms=number_of_atoms,
            use_fixed_lattice_parameters=True,
            cell_dimensions=cell_dimensions,
            record_samples=False,
        )

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
            sample_maker.torch_batch_axl_to_list_of_numpy_axl(torch_batch_axl)
        )
        assert len(calculated_list_of_numpy_axl) == batch_size

        for b in range(batch_size):
            current_numpy_axl = calculated_list_of_numpy_axl[b]
            assert np.array_equal(current_numpy_axl.A, batch_atom_types[b, :])
            assert np.allclose(current_numpy_axl.X, batch_relative_coordinates[b, :, :])
            assert np.allclose(current_numpy_axl.L, batch_lattice_parameters[b, :])

    def test_create_sampling_constraints(
        self, structure_axl, element_list, sample_maker
    ):

        calculated_sampling_constraint = sample_maker.create_sampling_constraints(
            structure_axl
        )

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
        with patch(
            "diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker."
            "excise_and_repaint_sample_maker.create_batch_of_samples",
            new=mock_create_batch_of_samples,
        ):
            # The inputs don't matter since we force the output with a mock.
            calculated_new_samples, _, _ = (
                sample_maker.make_samples_from_constrained_substructure(
                    substructure=structure_axl, active_atom_index=0
                )
            )

        mock_create_batch_of_samples.assert_called_once()
        assert len(calculated_new_samples) == batch_size
        for b in range(batch_size):
            assert np.array_equal(calculated_new_samples[b].A, batch_atom_types[b, :])
            assert np.array_equal(
                calculated_new_samples[b].X, batch_relative_coordinates[b, :, :]
            )
            assert np.array_equal(
                calculated_new_samples[b].L, batch_lattice_parameters[b, :]
            )

    def test_edit_generated_structure(
        self, structure_axl, basis_vectors, num_atom_types, radial_cutoff
    ):

        number_of_constrained_atoms = len(structure_axl.X)
        active_atom_index = np.random.randint(0, number_of_constrained_atoms)

        central_atom_relative_coordinates = structure_axl.X[active_atom_index]

        central_atom_cartesian_position = np.matmul(
            central_atom_relative_coordinates, basis_vectors
        )

        number_of_new_atoms = 10
        new_directions = np.random.rand(number_of_new_atoms, 3)
        new_directions /= einops.repeat(
            np.linalg.norm(new_directions, axis=1), "b -> b d", d=3
        )

        lengths = radial_cutoff * np.random.rand(number_of_new_atoms)
        new_cartesian_positions = (
            central_atom_cartesian_position + lengths[:, np.newaxis] * new_directions
        )

        reciprocal_lattice_vectors = get_reciprocal_basis_vectors(
            torch.from_numpy(basis_vectors)
        )

        new_relative_coordinates = get_relative_coordinates_from_cartesian_positions(
            torch.from_numpy(new_cartesian_positions), reciprocal_lattice_vectors
        )
        new_relative_coordinates = map_relative_coordinates_to_unit_cell(
            new_relative_coordinates
        ).numpy()
        new_atom_types = np.random.randint(num_atom_types, size=number_of_new_atoms)

        overloaded_structure_axl = AXL(
            A=np.concatenate([structure_axl.A, new_atom_types]),
            X=np.vstack([structure_axl.X, new_relative_coordinates]),
            L=structure_axl.L,
        )

        edited_structure_axl = ExciseAndRepaintSampleMaker.edit_generated_structure(
            overloaded_structure_axl,
            active_atom_index,
            number_of_constrained_atoms,
            radial_cutoff,
        )

        np.testing.assert_allclose(edited_structure_axl.A, structure_axl.A)
        np.testing.assert_allclose(edited_structure_axl.X, structure_axl.X)
        np.testing.assert_allclose(edited_structure_axl.L, structure_axl.L)
