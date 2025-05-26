from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.base_excisor import (
    NoOpEnvironmentExcision, NoOpEnvironmentExcisionArguments)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.excise_and_repaint_sample_maker import (  # noqa
    ExciseAndRepaintSampleMaker, ExciseAndRepaintSampleMakerArguments)
from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.generators.sampling_constraint import \
    SamplingConstraint
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    get_number_of_lattice_parameters


class TestExciseAndRepaintSampleMaker:
    @pytest.fixture
    def batch_size(self):
        return 3

    @pytest.fixture
    def number_of_atoms(self):
        return 14

    @pytest.fixture(params=[1, 2, 3])
    def spatial_dimension(self, request):
        return request.param

    @pytest.fixture
    def relative_coordinates(self, batch_size, number_of_atoms, spatial_dimension):
        return np.random.rand(batch_size, number_of_atoms, spatial_dimension)

    @pytest.fixture
    def num_atom_types(self):
        return 5

    @pytest.fixture
    def atoms_type(self, batch_size, number_of_atoms, num_atom_types):
        return np.random.randint(0, num_atom_types, (batch_size, number_of_atoms))

    @pytest.fixture
    def lattice_parameters(self, batch_size, spatial_dimension):
        num_lattice_parameters = get_number_of_lattice_parameters(spatial_dimension)
        lattice_params = np.zeros(
            (
                batch_size,
                num_lattice_parameters,
            )
        )
        lattice_params[:, :spatial_dimension] = np.random.rand(
            batch_size, spatial_dimension
        )
        return lattice_params

    @pytest.fixture
    def torch_batch_axl(self, atoms_type, relative_coordinates, lattice_parameters):
        return AXL(
            A=torch.tensor(atoms_type),
            X=torch.tensor(relative_coordinates),
            L=torch.tensor(lattice_parameters),
        )

    @pytest.fixture
    def e_and_r_samplemaker_arguments(self, num_atom_types, spatial_dimension):
        return ExciseAndRepaintSampleMakerArguments(
            element_list=list(range(num_atom_types)),
            max_constrained_substructure=1,
            number_of_samples_per_substructure=1,
            sample_box_size=[1.0] * spatial_dimension,
        )

    @pytest.fixture
    def environment_excisor(self):
        noop_excisor = NoOpEnvironmentExcision(
            NoOpEnvironmentExcisionArguments(excise_top_k_environment=1)
        )
        return noop_excisor

    @pytest.fixture
    def noise_parameters(self):
        return NoiseParameters(
            total_time_steps=2,
            schedule_type="exponential",
            sigma_min=0.01,
            sigma_max=0.5,
        )

    @pytest.fixture
    def sampling_parameters(
        self,
        spatial_dimension,
        number_of_atoms,
        num_atom_types,
        e_and_r_samplemaker_arguments,
    ):
        return PredictorCorrectorSamplingParameters(
            number_of_samples=1,
            spatial_dimension=spatial_dimension,
            number_of_corrector_steps=1,
            num_atom_types=num_atom_types,
            number_of_atoms=number_of_atoms,
            use_fixed_lattice_parameters=True,
            cell_dimensions=e_and_r_samplemaker_arguments.sample_box_size,
            record_samples=False,
        )

    @pytest.fixture
    def axl_score_network(self):
        return None

    @pytest.fixture
    def excise_and_repaint_sample_maker(
        self,
        e_and_r_samplemaker_arguments,
        environment_excisor,
        noise_parameters,
        sampling_parameters,
        axl_score_network,
    ):
        return ExciseAndRepaintSampleMaker(
            sample_maker_arguments=e_and_r_samplemaker_arguments,
            environment_excisor=environment_excisor,
            noise_parameters=noise_parameters,
            sampling_parameters=sampling_parameters,
            diffusion_model=axl_score_network,
        )

    def test_torch_batch_axl_to_list_of_numpy_axl(
        self,
        batch_size,
        relative_coordinates,
        atoms_type,
        lattice_parameters,
        torch_batch_axl,
        excise_and_repaint_sample_maker,
    ):
        calculated_list_of_numpy_axl = (
            excise_and_repaint_sample_maker.torch_batch_axl_to_list_of_numpy_axl(
                torch_batch_axl
            )
        )
        assert len(calculated_list_of_numpy_axl) == batch_size

        for b in range(batch_size):
            current_numpy_axl = calculated_list_of_numpy_axl[b]
            assert np.array_equal(current_numpy_axl.A, atoms_type[b, :])
            assert np.allclose(current_numpy_axl.X, relative_coordinates[b, :, :])
            assert np.allclose(current_numpy_axl.L, lattice_parameters[b, :])

    def test_create_sampling_constraints(
        self,
        relative_coordinates,
        atoms_type,
        num_atom_types,
        excise_and_repaint_sample_maker,
        spatial_dimension,
    ):
        constrained_axl = AXL(
            A=atoms_type[0, :],
            X=relative_coordinates[0, :],
            L=torch.zeros(
                get_number_of_lattice_parameters(spatial_dimension)
            ),  # not used
        )

        calculated_sampling_constraint = (
            excise_and_repaint_sample_maker.create_sampling_constraints(constrained_axl)
        )

        expected_sampling_constraint = SamplingConstraint(
            elements=list(range(num_atom_types)),
            constrained_relative_coordinates=torch.FloatTensor(
                relative_coordinates[0, :]
            ),
            constrained_atom_types=torch.LongTensor(atoms_type[0, :]),
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
        atoms_type,
        relative_coordinates,
        lattice_parameters,
        spatial_dimension,
        torch_batch_axl,
        batch_size,
        excise_and_repaint_sample_maker,
    ):
        constrained_axl = AXL(
            A=atoms_type[0, :],
            X=relative_coordinates[0, :],
            L=torch.zeros(
                get_number_of_lattice_parameters(spatial_dimension)
            ),  # not used
        )

        mock_create_batch_of_samples = MagicMock(
            return_value={"original_axl": torch_batch_axl}
        )
        with patch(
            "diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker."
            "excise_and_repaint_sample_maker.create_batch_of_samples",
            new=mock_create_batch_of_samples,
        ):
            calculated_new_samples = excise_and_repaint_sample_maker.make_samples_from_constrained_substructure(
                constrained_axl
            )

        mock_create_batch_of_samples.assert_called_once()
        assert len(calculated_new_samples) == batch_size
        for b in range(batch_size):
            assert np.array_equal(calculated_new_samples[b].A, atoms_type[b, :])
            assert np.array_equal(
                calculated_new_samples[b].X, relative_coordinates[b, :, :]
            )
            assert np.array_equal(calculated_new_samples[b].L, lattice_parameters[b, :])
