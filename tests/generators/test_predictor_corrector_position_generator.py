import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import \
    PredictorCorrectorAXLGenerator
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
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
        initial_sample: torch.Tensor,
    ):
        super().__init__(
            number_of_discretization_steps,
            number_of_corrector_steps,
            spatial_dimension,
            num_atom_types,
        )
        self.initial_sample = initial_sample

    def initialize(
        self, number_of_samples: int, device: torch.device = torch.device("cpu")
    ):
        return self.initial_sample

    def predictor_step(
        self,
        axl_ip1: AXL,
        ip1: int,
        unit_cell: torch.Tensor,
        forces: torch.Tensor,
    ) -> torch.Tensor:
        updated_axl = AXL(
            A=axl_ip1.A,
            X=map_relative_coordinates_to_unit_cell(
                1.2 * axl_ip1.X + 3.4 + ip1 / 111.0
            ),
            L=axl_ip1.L,
        )
        return updated_axl

    def corrector_step(
        self, axl_i: torch.Tensor, i: int, unit_cell: torch.Tensor, forces: torch.Tensor
    ) -> torch.Tensor:
        updated_axl = AXL(
            A=axl_i.A,
            X=map_relative_coordinates_to_unit_cell(0.56 * axl_i.X + 7.89 + i / 117.0),
            L=axl_i.L,
        )
        return updated_axl


@pytest.mark.parametrize("number_of_discretization_steps", [1, 5, 10])
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
            L=torch.rand(
                number_of_samples, spatial_dimension * (spatial_dimension - 1)
            ),  # TODO placeholder
        )

    @pytest.fixture
    def generator(
        self,
        number_of_discretization_steps,
        number_of_corrector_steps,
        spatial_dimension,
        num_atom_types,
        initial_sample,
    ):
        generator = FakePCGenerator(
            number_of_discretization_steps,
            number_of_corrector_steps,
            spatial_dimension,
            num_atom_types,
            initial_sample,
        )
        return generator

    @pytest.fixture
    def expected_samples(
        self,
        generator,
        initial_sample,
        number_of_discretization_steps,
        number_of_corrector_steps,
        unit_cell_sample,
    ):
        list_i = list(range(number_of_discretization_steps))
        list_i.reverse()
        list_j = list(range(number_of_corrector_steps))

        noisy_sample = map_axl_composition_to_unit_cell(
            initial_sample, torch.device("cpu")
        )
        composition_ip1 = noisy_sample
        for i in list_i:
            composition_i = map_axl_composition_to_unit_cell(
                generator.predictor_step(
                    composition_ip1,
                    i + 1,
                    unit_cell_sample,
                    torch.zeros_like(composition_ip1.X),
                ),
                torch.device("cpu"),
            )
            for _ in list_j:
                composition_i = map_axl_composition_to_unit_cell(
                    generator.corrector_step(
                        composition_i,
                        i,
                        unit_cell_sample,
                        torch.zeros_like(composition_i.X),
                    ),
                    torch.device("cpu"),
                )
            composition_ip1 = composition_i
        return composition_i

    def test_sample(
        self, generator, number_of_samples, expected_samples, unit_cell_sample
    ):
        computed_samples = generator.sample(
            number_of_samples, torch.device("cpu"), unit_cell_sample
        )

        torch.testing.assert_close(expected_samples, computed_samples)
