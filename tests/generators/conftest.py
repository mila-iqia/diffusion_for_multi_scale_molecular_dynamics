from typing import AnyStr, Dict

import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.models.score_networks import (
    ScoreNetwork, ScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, NOISY_AXL_COMPOSITION)


class FakeAXLNetwork(ScoreNetwork):
    """A fake, smooth score network for the ODE solver."""

    def _forward_unchecked(
        self, batch: Dict[AnyStr, torch.Tensor], conditional: bool = False
    ) -> AXL:
        return AXL(A=None, X=batch[NOISY_AXL_COMPOSITION].X, L=None)


class BaseTestGenerator:
    """A base class that contains common test fixtures useful for testing generators."""

    @pytest.fixture()
    def unit_cell_size(self):
        return 10

    @pytest.fixture()
    def number_of_atoms(self):
        return 8

    @pytest.fixture()
    def number_of_samples(self):
        return 5

    @pytest.fixture(params=[2, 3])
    def spatial_dimension(self, request):
        return request.param

    @pytest.fixture()
    def num_atom_types(self):
        return 6

    @pytest.fixture()
    def unit_cell_sample(self, unit_cell_size, spatial_dimension, number_of_samples):
        return torch.diag(torch.Tensor([unit_cell_size] * spatial_dimension)).repeat(
            number_of_samples, 1, 1
        )

    @pytest.fixture()
    def cell_dimensions(self, unit_cell_size, spatial_dimension):
        return spatial_dimension * [unit_cell_size]

    @pytest.fixture()
    def axl_network(self, spatial_dimension, num_atom_types):
        return FakeAXLNetwork(
            ScoreNetworkParameters(
                architecture="dummy",
                spatial_dimension=spatial_dimension,
                num_atom_types=num_atom_types,
            )
        )
