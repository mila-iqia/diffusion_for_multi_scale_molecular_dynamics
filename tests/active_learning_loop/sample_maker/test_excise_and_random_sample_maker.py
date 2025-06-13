from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.base_excisor import (
    NoOpEnvironmentExcision, NoOpEnvironmentExcisionArguments)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.excise_and_random_sample_maker import (  # noqa
    ExciseAndRandomSampleMaker, ExciseAndRandomSampleMakerArguments)
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    get_number_of_lattice_parameters


class TestExciseAndRandomSampleMaker:
    @pytest.fixture(params=[1, 2, 3])
    def spatial_dimension(self, request):
        return request.param

    @pytest.fixture
    def num_atom_types(self):
        return 5

    @pytest.fixture(params=[4, 6])
    def total_number_of_atoms(self, request):
        return request.param

    @pytest.fixture
    def num_constrained_atoms(self):
        return 3

    @pytest.fixture
    def constrained_relative_coordinates(
        self, num_constrained_atoms, spatial_dimension
    ):
        return np.random.rand(num_constrained_atoms, spatial_dimension)

    @pytest.fixture
    def constrained_atoms_type(self, num_atom_types, num_constrained_atoms):
        return np.random.randint(0, num_atom_types, (num_constrained_atoms,))

    @pytest.fixture
    def lattice_parameters(self, spatial_dimension):
        num_lattice_parameters = get_number_of_lattice_parameters(spatial_dimension)
        lattice_params = np.zeros((num_lattice_parameters,))
        lattice_params[:spatial_dimension] = np.random.rand(spatial_dimension)
        return lattice_params

    @pytest.fixture
    def constrained_axl_structure(
        self,
        constrained_relative_coordinates,
        constrained_atoms_type,
        lattice_parameters,
    ):
        return AXL(
            A=constrained_atoms_type,
            X=constrained_relative_coordinates,
            L=lattice_parameters,
        )

    @pytest.fixture(params=["true_random", "voxel_random"])
    def random_algorithm(self, request):
        return request.param

    @pytest.fixture(params=[1, 2, 3])
    def num_generated_samples(self, request):
        return request.param

    @pytest.fixture
    def e_and_r_samplemaker_arguments(
        self,
        num_atom_types,
        spatial_dimension,
        random_algorithm,
        total_number_of_atoms,
        num_generated_samples,
    ):
        return ExciseAndRandomSampleMakerArguments(
            element_list=list(range(num_atom_types)),
            max_constrained_substructure=1,
            number_of_samples_per_substructure=num_generated_samples,
            sample_box_size=[1.0] * spatial_dimension,
            total_number_of_atoms=total_number_of_atoms,
            random_coordinates_algorithm=random_algorithm,
            minimal_interatomic_distance=1e-5,
            max_attempts=1,
        )

    @pytest.fixture
    def environment_excisor(self):
        noop_excisor = NoOpEnvironmentExcision(
            NoOpEnvironmentExcisionArguments(excise_top_k_environment=1)
        )
        return noop_excisor

    @pytest.fixture
    def excise_and_random_sample_maker(
        self,
        e_and_r_samplemaker_arguments,
        environment_excisor,
    ):
        return ExciseAndRandomSampleMaker(
            sample_maker_arguments=e_and_r_samplemaker_arguments,
            environment_excisor=environment_excisor,
        )

    @pytest.fixture
    def expected_atom_types(self, total_number_of_atoms, num_atom_types):
        return np.random.randint(0, num_atom_types, (total_number_of_atoms,))

    @pytest.fixture
    def expected_relative_coordinates(self, total_number_of_atoms, spatial_dimension):
        return np.random.rand(total_number_of_atoms, spatial_dimension)

    @pytest.fixture
    def expected_axl_after_replacing_constrained_atoms(
        self,
        expected_relative_coordinates,
        expected_atom_types,
        constrained_axl_structure,
        spatial_dimension,
    ):
        box_size = constrained_axl_structure.L[:spatial_dimension][
            np.newaxis, :
        ]  # 1, spatial_dimension
        final_relative_coordinates = expected_relative_coordinates.copy()
        final_atom_types = expected_atom_types.copy()
        replaced_atom_history = []
        for a, x in zip(constrained_axl_structure.A, constrained_axl_structure.X):
            distance_between_constrained_atom_and_others = (
                expected_relative_coordinates - x[np.newaxis, :]
            ) * box_size
            distance_between_constrained_atom_and_others = (
                distance_between_constrained_atom_and_others**2
            )
            distance_between_constrained_atom_and_others_plus_shift = (
                expected_relative_coordinates - x[np.newaxis, :] + 1
            ) * box_size
            distance_between_constrained_atom_and_others_plus_shift = (
                distance_between_constrained_atom_and_others_plus_shift**2
            )
            distance_between_constrained_atom_and_others_minus_shift = (
                expected_relative_coordinates - x[np.newaxis, :] - 1
            ) * box_size
            distance_between_constrained_atom_and_others_minus_shift = (
                distance_between_constrained_atom_and_others_minus_shift**2
            )
            distance_between_constrained_atom_and_others = np.minimum(
                distance_between_constrained_atom_and_others,
                distance_between_constrained_atom_and_others_plus_shift,
            )
            distance_between_constrained_atom_and_others = np.minimum(
                distance_between_constrained_atom_and_others,
                distance_between_constrained_atom_and_others_minus_shift,
            )
            sorted_idx_by_distance = np.argsort(
                distance_between_constrained_atom_and_others.sum(axis=-1)
            )
            for idx in sorted_idx_by_distance:
                if idx not in replaced_atom_history:
                    replaced_atom_history.append(idx)
                    break
            final_relative_coordinates[replaced_atom_history[-1], :] = x
            final_atom_types[replaced_atom_history[-1]] = a
        return AXL(
            A=final_atom_types,
            X=final_relative_coordinates,
            L=constrained_axl_structure.L,
        )

    def test_make_samples_from_constrained_substructure(
        self,
        constrained_axl_structure,
        excise_and_random_sample_maker,
        expected_relative_coordinates,
        expected_atom_types,
        random_algorithm,
        num_generated_samples,
        expected_axl_after_replacing_constrained_atoms,
    ):
        mock_generate_atom_types = MagicMock(return_value=expected_atom_types)
        mock_generate_relative_coordinates = MagicMock(
            return_value=expected_relative_coordinates
        )
        if random_algorithm == "true_random":
            relative_coordinates_mock_target = (
                "diffusion_for_multi_scale_molecular_dynamics."
                "active_learning_loop.sample_maker.excise_and_random_sample_maker."
                "ExciseAndRandomSampleMaker.generate_relative_coordinates_true_random"
            )
        else:
            relative_coordinates_mock_target = (
                "diffusion_for_multi_scale_molecular_dynamics.active_learning_loop."
                "sample_maker.excise_and_random_sample_maker."
                "ExciseAndRandomSampleMaker.generate_relative_coordinates_voxel_random"
            )

        with patch(
            "diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker."
            "excise_and_random_sample_maker.ExciseAndRandomSampleMaker.generate_atom_types",
            new=mock_generate_atom_types,
        ):
            with patch(
                relative_coordinates_mock_target, new=mock_generate_relative_coordinates
            ):
                calculated_new_samples, _ = (
                    excise_and_random_sample_maker.make_samples_from_constrained_substructure(
                        constrained_axl_structure, num_samples=num_generated_samples
                    )
                )

        assert len(calculated_new_samples) == num_generated_samples
        for new_sample in calculated_new_samples:
            assert np.array_equal(
                new_sample.A, expected_axl_after_replacing_constrained_atoms.A
            )
            assert np.allclose(
                new_sample.X, expected_axl_after_replacing_constrained_atoms.X
            )
            assert np.allclose(
                new_sample.L, expected_axl_after_replacing_constrained_atoms.L
            )

    def test_sort_atoms_indices_by_distance(
        self,
        expected_relative_coordinates,
        lattice_parameters,
        spatial_dimension,
        excise_and_random_sample_maker,
    ):
        target_point = np.random.rand(spatial_dimension)
        target_point_cartesian = target_point * lattice_parameters[:spatial_dimension]
        cartesian_coordinates = (
            expected_relative_coordinates * lattice_parameters[:spatial_dimension]
        )
        distance_to_atom = (cartesian_coordinates - target_point_cartesian) ** 2
        distance_to_atom_shift_plus = (
            cartesian_coordinates
            - target_point_cartesian
            + lattice_parameters[:spatial_dimension]
        ) ** 2
        distance_to_atom_shift_minus = (
            cartesian_coordinates
            - target_point_cartesian
            - lattice_parameters[:spatial_dimension]
        ) ** 2
        distance_to_atom = np.minimum(distance_to_atom, distance_to_atom_shift_plus)
        distance_to_atom = np.minimum(distance_to_atom, distance_to_atom_shift_minus)
        distances = distance_to_atom.sum(axis=-1)

        calculated_order = (
            excise_and_random_sample_maker.sort_atoms_indices_by_distance(
                target_point, expected_relative_coordinates, lattice_parameters
            )
        )

        for calculated_index in calculated_order:
            expected_index = np.argmin(distances)
            assert calculated_index == expected_index
            distances[expected_index] = np.infty

    def test_get_shortest_distance_between_atoms(
        self,
        expected_relative_coordinates,
        lattice_parameters,
        spatial_dimension,
        excise_and_random_sample_maker,
    ):
        shortest_distance_between_atoms = np.infty
        all_cartesian_coordinates = (
            expected_relative_coordinates * lattice_parameters[:spatial_dimension]
        )
        for atom_coordinates in expected_relative_coordinates:
            atom_cartesian_coordinates = (
                atom_coordinates * lattice_parameters[:spatial_dimension]
            )
            distance_to_atom = (
                all_cartesian_coordinates - atom_cartesian_coordinates
            ) ** 2
            distance_to_atom_shift_plus = (
                all_cartesian_coordinates
                - atom_cartesian_coordinates
                + lattice_parameters[:spatial_dimension]
            ) ** 2
            distance_to_atom_shift_minus = (
                all_cartesian_coordinates
                - atom_cartesian_coordinates
                - lattice_parameters[:spatial_dimension]
            ) ** 2
            distance_to_atom = np.minimum(distance_to_atom, distance_to_atom_shift_plus)
            distance_to_atom = np.minimum(
                distance_to_atom, distance_to_atom_shift_minus
            )
            distances = np.sqrt(distance_to_atom.sum(axis=-1))
            sorted_distances = np.sort(distances)
            shortest_distance_between_atoms = min(
                shortest_distance_between_atoms, sorted_distances[1]
            )

        calculated_distance = (
            excise_and_random_sample_maker.get_shortest_distance_between_atoms(
                expected_relative_coordinates, lattice_parameters
            )
        )
        assert np.allclose(calculated_distance, shortest_distance_between_atoms)

    @pytest.fixture
    def num_voxel(self):
        return 2

    @pytest.fixture
    def num_voxels_per_dimension(self, spatial_dimension, num_voxel):
        return np.ones(spatial_dimension).astype(int) * num_voxel

    @pytest.fixture
    def box_partition(self, num_voxels_per_dimension):
        grid_points = [
            np.linspace(0, 1, p, endpoint=False) for p in num_voxels_per_dimension
        ]
        meshes = np.meshgrid(*grid_points, indexing="ij")
        stacked_meshes = np.stack(meshes).reshape(len(meshes), -1)
        return stacked_meshes

    @pytest.fixture
    def mocked_voxel_occupation(self, total_number_of_atoms, num_voxels_per_dimension):
        num_voxel = np.prod(num_voxels_per_dimension)
        return np.arange(total_number_of_atoms) % num_voxel

    @pytest.fixture
    def expected_atom_coordinates_in_voxels(
        self,
        mocked_voxel_occupation,
        box_partition,
        expected_relative_coordinates,
        num_voxels_per_dimension,
    ):
        num_atom, spatial_dimension = expected_relative_coordinates.shape
        updated_coordinates = expected_relative_coordinates.copy()
        for i in range(spatial_dimension):
            updated_coordinates[:, i] /= num_voxels_per_dimension[i]
        for atom in range(num_atom):
            voxel_index = mocked_voxel_occupation[atom]
            voxel_coordinates = box_partition[:, voxel_index]
            updated_coordinates[atom, :] += voxel_coordinates
        return updated_coordinates

    def test_generate_relative_coordinates_voxel_random(
        self,
        lattice_parameters,
        expected_relative_coordinates,
        excise_and_random_sample_maker,
        box_partition,
        num_voxels_per_dimension,
        total_number_of_atoms,
        mocked_voxel_occupation,
        expected_atom_coordinates_in_voxels,
    ):
        # mock utils to make this test easy
        mock_generate_relative_coordinates = MagicMock(
            return_value=expected_relative_coordinates
        )
        mock_partition_relative_coordinates = MagicMock(
            return_value=(box_partition, num_voxels_per_dimension)
        )
        mock_select_voxels = MagicMock(return_value=mocked_voxel_occupation)
        with patch(
            "diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker."
            "excise_and_random_sample_maker.ExciseAndRandomSampleMaker.generate_random_relative_coordinates",
            new=mock_generate_relative_coordinates,
        ):
            with patch(
                "diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker."
                "excise_and_random_sample_maker.partition_relative_coordinates_for_voxels",
                new=mock_partition_relative_coordinates,
            ):
                with patch(
                    "diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker."
                    "excise_and_random_sample_maker.select_occupied_voxels",
                    new=mock_select_voxels,
                ):
                    calculated_atom_coordinates_in_voxels = \
                        excise_and_random_sample_maker.generate_relative_coordinates_voxel_random(lattice_parameters)
        assert np.allclose(
            calculated_atom_coordinates_in_voxels, expected_atom_coordinates_in_voxels
        )
