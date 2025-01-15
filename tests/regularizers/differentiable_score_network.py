from dataclasses import dataclass
from typing import AnyStr, Dict

import einops
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.models.score_networks import (
    ScoreNetwork, ScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, CARTESIAN_FORCES, NOISE, NOISY_AXL_COMPOSITION, TIME, UNIT_CELL)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.exploding_variance import \
    SigmaCalculator


@dataclass(kw_only=True)
class DifferentiableScoreNetworkParameters(ScoreNetworkParameters):
    """Hyper-parameters for a differentiable score network."""

    architecture: str = "differentiable_score_network"
    number_of_atoms: int
    sigma_min: float
    sigma_max: float


class DifferentiableScoreNetwork(ScoreNetwork):
    """Differentiable score network.

    A simple score network where it is straightforward to extract the analytical derivatives.
    """

    def __init__(self, hyper_params: DifferentiableScoreNetworkParameters):
        super().__init__(hyper_params)

        self.sigma_calculator = SigmaCalculator(
            sigma_min=hyper_params.sigma_min, sigma_max=hyper_params.sigma_max
        )

        self.natoms = hyper_params.number_of_atoms
        self.nd = self.natoms * self.spatial_dimension

        self.scrambling_matrix = torch.rand(self.nd, self.nd)

    def _get_flat_xyt(self, relative_coordinates: torch.Tensor, times: torch.Tensor):
        x = einops.rearrange(
            relative_coordinates, "batch natoms space -> batch (natoms space)"
        )
        y = einops.einsum(self.scrambling_matrix, x, "nd nd1, batch nd1 -> batch nd")
        t = einops.repeat(times, "batch 1 -> batch nd", nd=self.nd)
        return x, y, t

    def _score_function(
        self, relative_coordinates: torch.Tensor, times: torch.Tensor
    ) -> torch.Tensor:
        # Create a function that is non-trivial, but still easy to differentiate analytically.
        x, y, t = self._get_flat_xyt(relative_coordinates, times)

        f = torch.cos((t * y) ** 2)
        scores = einops.rearrange(
            f,
            "batch (natoms space) -> batch natoms space",
            natoms=self.natoms,
            space=self.spatial_dimension,
        )
        return scores

    def _time_derivative_function(
        self, relative_coordinates: torch.Tensor, times: torch.Tensor
    ) -> torch.Tensor:
        x, y, t = self._get_flat_xyt(relative_coordinates, times)

        df_dt = -2.0 * y**2 * t * torch.sin((t * y) ** 2)
        d_scores_dt = einops.rearrange(
            df_dt,
            "batch (natoms space) -> batch natoms space",
            natoms=self.natoms,
            space=self.spatial_dimension,
        )
        return d_scores_dt

    def _space_jacobian_function(
        self, relative_coordinates: torch.Tensor, times: torch.Tensor
    ) -> torch.Tensor:
        batch_size = len(relative_coordinates)
        x, y, t = self._get_flat_xyt(relative_coordinates, times)

        df_dy = -2.0 * y * t**2 * torch.sin((t * y) ** 2)

        repeated_df_dy = einops.repeat(df_dy, "batch i -> batch i j", j=self.nd)
        repeated_scrambling_matrix = einops.repeat(
            self.scrambling_matrix, "i j -> batch i j", batch=batch_size
        )
        flat_jacobian = repeated_df_dy * repeated_scrambling_matrix

        jacobian = einops.rearrange(
            flat_jacobian,
            "batch (natoms1 space1) (natoms2 space2) -> batch natoms1 space1 natoms2 space2",
            natoms1=self.natoms,
            natoms2=self.natoms,
            space1=self.spatial_dimension,
            space2=self.spatial_dimension,
        )
        return jacobian

    def _space_hessian_function(
        self, relative_coordinates: torch.Tensor, times: torch.Tensor
    ) -> torch.Tensor:
        batch_size = len(relative_coordinates)
        x, y, t = self._get_flat_xyt(relative_coordinates, times)

        yt2 = (y * t) ** 2
        d2f_dy2 = -2.0 * t**2 * (torch.sin(yt2) + 2.0 * yt2 * torch.cos(yt2))

        repeated_d2f_dy2 = einops.repeat(
            d2f_dy2, "batch i -> batch i j k", j=self.nd, k=self.nd
        )
        m_ij = einops.repeat(
            self.scrambling_matrix, "i j -> batch i j k", batch=batch_size, k=self.nd
        )

        m_ik = einops.repeat(
            self.scrambling_matrix, "i k -> batch i j k", batch=batch_size, j=self.nd
        )

        flat_hessian = repeated_d2f_dy2 * m_ij * m_ik

        hessian = einops.rearrange(
            flat_hessian,
            "batch (natoms1 space1) (natoms2 space2) (natoms3 space3) "
            "-> batch natoms1 space1 natoms2 space2 natoms3 space3",
            natoms1=self.natoms,
            natoms2=self.natoms,
            natoms3=self.natoms,
            space1=self.spatial_dimension,
            space2=self.spatial_dimension,
            space3=self.spatial_dimension,
        )
        return hessian

    def _forward_unchecked(
        self, batch: Dict[AnyStr, torch.Tensor], conditional: bool = False
    ) -> AXL:
        relative_coordinates = batch[NOISY_AXL_COMPOSITION].X
        times = batch[TIME]

        sigmas_t = einops.repeat(
            self.sigma_calculator.get_sigma(times),
            "batch 1 -> batch natoms space",
            natoms=self.natoms,
            space=self.spatial_dimension,
        )

        sigma_normalized_scores = sigmas_t * self._score_function(
            relative_coordinates, times
        )

        batch_size = relative_coordinates.shape[0]
        dummy_atom_predictions = torch.zeros(
            batch_size, self.natoms, self.num_atom_types + 1
        )

        return AXL(
            A=dummy_atom_predictions, X=sigma_normalized_scores, L=torch.tensor(0)
        )


class TestDifferentiableScoreNetwork:
    """Yes, testing the test code."""

    @pytest.fixture(scope="class", autouse=True)
    def set_default_type_to_float64(self):
        torch.set_default_dtype(torch.float64)
        yield
        # this returns the default type to float32 at the end of all tests in this class in order
        # to not affect other tests.
        torch.set_default_dtype(torch.float32)

    @pytest.fixture(scope="class", autouse=True)
    def set_seed(self):
        """Set the random seed."""
        torch.manual_seed(2342342)

    @pytest.fixture()
    def sigma_min(self):
        return 0.001

    @pytest.fixture()
    def sigma_max(self):
        return 0.2

    @pytest.fixture()
    def number_of_atoms(self):
        return 4

    @pytest.fixture()
    def spatial_dimension(self):
        return 3

    @pytest.fixture()
    def batch_size(self):
        return 16

    @pytest.fixture()
    def relative_coordinates(self, batch_size, number_of_atoms, spatial_dimension):
        return torch.rand(batch_size, number_of_atoms, spatial_dimension)

    @pytest.fixture()
    def times(self, batch_size):
        return torch.rand(batch_size, 1)

    @pytest.fixture()
    def batch(
        self,
        batch_size,
        number_of_atoms,
        spatial_dimension,
        relative_coordinates,
        times,
        sigma_min,
        sigma_max,
    ):

        atom_types = torch.zeros(batch_size, number_of_atoms, dtype=torch.long)
        unit_cells = torch.diag(5 * torch.ones(spatial_dimension)).repeat(
            batch_size, 1, 1
        )

        sigmas_t = SigmaCalculator(sigma_min=sigma_min, sigma_max=sigma_max).get_sigma(
            times
        )

        forces = torch.zeros_like(relative_coordinates)

        composition = AXL(A=atom_types, X=relative_coordinates, L=unit_cells)

        batch = {
            NOISY_AXL_COMPOSITION: composition,
            NOISE: sigmas_t,
            TIME: times,
            UNIT_CELL: unit_cells,
            CARTESIAN_FORCES: forces,
        }
        return batch

    @pytest.fixture()
    def score_parameters(
        self, number_of_atoms, spatial_dimension, sigma_min, sigma_max
    ):
        score_parameters = DifferentiableScoreNetworkParameters(
            number_of_atoms=number_of_atoms,
            spatial_dimension=spatial_dimension,
            num_atom_types=1,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
        return score_parameters

    @pytest.fixture()
    def score_network(self, score_parameters):
        return DifferentiableScoreNetwork(score_parameters)

    def test_score_network(
        self,
        score_network,
        relative_coordinates,
        times,
        batch,
        number_of_atoms,
        spatial_dimension,
    ):
        sigma_normalized_scores = score_network(batch).X

        scores = score_network._score_function(relative_coordinates, times)
        assert scores.shape == relative_coordinates.shape

        sigmas_t = einops.repeat(
            batch[NOISE],
            "batch 1 -> batch natoms space",
            natoms=number_of_atoms,
            space=spatial_dimension,
        )

        torch.testing.assert_close(sigmas_t * scores, sigma_normalized_scores)

    def test_time_derivative(
        self, score_network, relative_coordinates, times, batch_size
    ):

        computed_time_derivative = score_network._time_derivative_function(
            relative_coordinates, times
        )
        assert computed_time_derivative.shape == relative_coordinates.shape

        time_jacobian = torch.func.jacrev(score_network._score_function, argnums=1)(
            relative_coordinates, times
        )
        # only keep the batch diagonal terms
        batch_idx = torch.arange(batch_size)
        expected_time_derivative = time_jacobian[batch_idx, :, :, batch_idx, 0]

        torch.testing.assert_close(computed_time_derivative, expected_time_derivative)

    def test_jacobian(self, score_network, relative_coordinates, times, batch_size):
        computed_jacobian = score_network._space_jacobian_function(
            relative_coordinates, times
        )

        space_jacobian = torch.func.jacrev(score_network._score_function, argnums=0)(
            relative_coordinates, times
        )
        # only keep the batch diagonal terms
        batch_idx = torch.arange(batch_size)
        expected_jacobian = space_jacobian[batch_idx, :, :, batch_idx, :, :]

        torch.testing.assert_close(computed_jacobian, expected_jacobian)

    def test_hessian(self, score_network, relative_coordinates, times, batch_size):
        computed_hessian = score_network._space_hessian_function(
            relative_coordinates, times
        )

        space_hessian = torch.func.hessian(score_network._score_function, argnums=0)(
            relative_coordinates, times
        )
        # only keep the batch diagonal terms
        batch_idx = torch.arange(batch_size)
        expected_hessian = space_hessian[
            batch_idx, :, :, batch_idx, :, :, batch_idx, :, :
        ]

        torch.testing.assert_close(computed_hessian, expected_hessian)
