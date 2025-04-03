import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import \
    PredictorCorrectorAXLGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.trajectory_initializer import (
    StartFromGivenConfigurationTrajectoryInitializer, TrajectoryInitializer,
    TrajectoryInitializerParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, NOISY_AXL_COMPOSITION)
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import (
    map_axl_composition_to_unit_cell, map_relative_coordinates_to_unit_cell)
from tests.generators.conftest import BaseTestGenerator


class FakePCGenerator(PredictorCorrectorAXLGenerator):
    """A dummy PC generator for the purpose of testing."""

    def __init__(
        self,
        number_of_discretization_steps: int,
        number_of_corrector_steps: int,
        spatial_dimension: int,
        num_atom_types: int,
        number_of_atoms: int,
        trajectory_initializer: TrajectoryInitializer,
    ):
        super().__init__(
            number_of_discretization_steps,
            number_of_corrector_steps,
            spatial_dimension,
            num_atom_types,
            number_of_atoms,
            use_fixed_lattice_parameters=False,
            fixed_lattice_parameters=None,
            trajectory_initializer=trajectory_initializer
        )

    def predictor_step(
        self,
        axl_ip1: AXL,
        ip1: int,
        forces: torch.Tensor,
    ) -> torch.Tensor:
        updated_axl = AXL(
            A=axl_ip1.A,
            X=map_relative_coordinates_to_unit_cell(
                1.2 * axl_ip1.X + 3.4 + ip1 / 111.0
            ),
            L=1.2 * axl_ip1.L + 3.4 + ip1 / 111.0,
        )
        return updated_axl

    def corrector_step(
        self, axl_i: torch.Tensor, i: int, forces: torch.Tensor
    ) -> torch.Tensor:
        updated_axl = AXL(
            A=axl_i.A,
            X=map_relative_coordinates_to_unit_cell(0.56 * axl_i.X + 7.89 + i / 117.0),
            L=0.56 * axl_i.L + 7.89 + i / 117.0,
        )
        return updated_axl


@pytest.mark.parametrize("number_of_discretization_steps", [2, 5, 10])
@pytest.mark.parametrize("number_of_corrector_steps", [0, 1, 2])
class TestPredictorCorrectorPositionGenerator(BaseTestGenerator):
    @pytest.fixture(scope="class", autouse=True)
    def set_random_seed(self):
        torch.manual_seed(1234567)

    @pytest.fixture
    def initial_sample(
        self, number_of_samples, number_of_atoms, spatial_dimension, num_atom_types
    ):
        return AXL(
            A=torch.randint(
                0, num_atom_types + 1, (number_of_samples, number_of_atoms)
            ),
            X=torch.rand(number_of_samples, number_of_atoms, spatial_dimension),
            L=torch.randn(
                number_of_samples, int(spatial_dimension * (spatial_dimension + 1) / 2)
            ),
        )

    @pytest.fixture()
    def path_to_starting_configuration_data_pickle(self, initial_sample, number_of_discretization_steps, tmp_path):
        path = str(tmp_path / "starting_configurations.pickle")
        data = {NOISY_AXL_COMPOSITION: initial_sample,
                'start_time_step_index': number_of_discretization_steps}

        torch.save(data, path)
        return path

    @pytest.fixture()
    def trajectory_initializer(self,
                               spatial_dimension,
                               num_atom_types,
                               number_of_atoms,
                               path_to_starting_configuration_data_pickle):
        params = TrajectoryInitializerParameters(
            spatial_dimension=spatial_dimension,
            num_atom_types=num_atom_types,
            number_of_atoms=number_of_atoms,
            path_to_starting_configuration_data_pickle=path_to_starting_configuration_data_pickle)

        trajectory_initializer = StartFromGivenConfigurationTrajectoryInitializer(params)
        return trajectory_initializer

    @pytest.fixture
    def generator(
        self,
        number_of_discretization_steps,
        number_of_corrector_steps,
        spatial_dimension,
        num_atom_types,
        number_of_atoms,
        trajectory_initializer,
    ):
        generator = FakePCGenerator(
            number_of_discretization_steps,
            number_of_corrector_steps,
            spatial_dimension,
            num_atom_types,
            number_of_atoms,
            trajectory_initializer,
        )
        return generator

    @pytest.fixture
    def all_generated_compositions(
        self,
        generator,
        initial_sample,
        number_of_discretization_steps,
        number_of_corrector_steps,
    ):
        list_i = list(range(number_of_discretization_steps))
        list_i.reverse()
        list_j = list(range(number_of_corrector_steps))

        noisy_sample = map_axl_composition_to_unit_cell(
            initial_sample, torch.device("cpu")
        )
        composition_ip1 = noisy_sample

        list_compositions = []
        for i in list_i:
            composition_i = map_axl_composition_to_unit_cell(
                generator.predictor_step(
                    composition_ip1,
                    i + 1,
                    torch.zeros_like(composition_ip1.X),
                ),
                torch.device("cpu"),
            )
            for _ in list_j:
                composition_i = map_axl_composition_to_unit_cell(
                    generator.corrector_step(
                        composition_i,
                        i,
                        torch.zeros_like(composition_i.X),
                    ),
                    torch.device("cpu"),
                )
            composition_ip1 = composition_i
            list_compositions.append(composition_i)

        return list_compositions

    def test_sample(
        self,
        generator,
        number_of_samples,
        all_generated_compositions,
    ):

        expected_samples = all_generated_compositions[-1]
        computed_samples = generator.sample(
            number_of_samples,
            torch.device("cpu"),
        )

        torch.testing.assert_close(expected_samples, computed_samples)

    def test_sample_from_noisy_composition(
        self,
        generator,
        initial_sample,
        number_of_discretization_steps,
        all_generated_compositions,
        unit_cell_sample,
    ):

        starting_noisy_composition = initial_sample

        for idx, starting_step_index in enumerate(
            range(number_of_discretization_steps, 1, -1)
        ):
            ending_step_index = starting_step_index - 1
            generated_sample = generator.sample_from_noisy_composition(
                starting_noisy_composition,
                starting_step_index,
                ending_step_index,
            )

            expected_sample = all_generated_compositions[idx]
            torch.testing.assert_close(expected_sample, generated_sample)
            starting_noisy_composition = generated_sample
