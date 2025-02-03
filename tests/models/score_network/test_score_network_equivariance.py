import einops
import pytest
import torch
from e3nn import o3

from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.diffusion_mace_score_network import (
    DiffusionMACEScoreNetwork, DiffusionMACEScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.egnn_score_network import (
    EGNNScoreNetwork, EGNNScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.equivariant_analytical_score_network import (
    EquivariantAnalyticalScoreNetwork,
    EquivariantAnalyticalScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.mace_score_network import (
    MACEScoreNetwork, MACEScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_prediction_head import \
    MaceEquivariantScorePredictionHeadParameters
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, CARTESIAN_FORCES, NOISE, NOISY_AXL_COMPOSITION,
    NOISY_CARTESIAN_POSITIONS, TIME, UNIT_CELL)
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import (
    get_positions_from_coordinates, get_reciprocal_basis_vectors,
    get_relative_coordinates_from_cartesian_positions,
    map_relative_coordinates_to_unit_cell)
from diffusion_for_multi_scale_molecular_dynamics.utils.geometric_utils import \
    get_cubic_point_group_symmetries
from tests.models.score_network.base_test_score_network import \
    BaseTestScoreNetwork


class BaseTestScoreEquivariance(BaseTestScoreNetwork):

    @staticmethod
    def apply_rotation_to_configuration(batch_rotation_matrices, batch_configuration):
        """Apply rotations to configuration.

        Args:
            batch_rotation_matrices : Dimension [batch_size, spatial_dimension, spatial_dimension]
            batch_configuration : Dimension [batch_size, number_of_atoms, spatial_dimension]

        Returns:
            rotated_batch_configuration : Dimension [batch_size, number_of_atoms, spatial_dimension]
        """
        return einops.einsum(
            batch_rotation_matrices,
            batch_configuration,
            "batch alpha beta, batch natoms beta -> batch natoms alpha",
        ).contiguous()

    @staticmethod
    def get_rotated_basis_vectors(batch_rotation_matrices, basis_vectors):
        """Get rotated basis vectors.

        Basis vectors are assumed to be in ROW format,

        basis_vectors = [ --- a1 ---]
                        [---- a2 ---]
                        [---- a3 ---]

        Args:
            batch_rotation_matrices : Dimension [batch_size, spatial_dimension, spatial_dimension]
            basis_vectors : Dimension [batch_size, spatial_dimension, spatial_dimension]

        Returns:
            rotated_basis_vectors : Dimension [batch_size, spatial_dimension, spatial_dimension]
        """
        new_basis_vectors = einops.einsum(
            batch_rotation_matrices,
            basis_vectors,
            "batch alpha beta, batch i beta -> batch i alpha",
        ).contiguous()
        return new_basis_vectors

    @staticmethod
    def create_batch(
        relative_coordinates,
        cartesian_positions,
        atom_types,
        basis_vectors,
        times,
        noises,
        forces,
    ):
        batch = {
            NOISY_AXL_COMPOSITION: AXL(
                A=atom_types,
                X=relative_coordinates,
                L=torch.zeros_like(atom_types),  # TODO
            ),
            NOISY_CARTESIAN_POSITIONS: cartesian_positions,
            TIME: times,
            NOISE: noises,
            UNIT_CELL: basis_vectors,
            CARTESIAN_FORCES: forces,
        }
        return batch

    @pytest.fixture(scope="class", autouse=True)
    def set_default_type_to_float64(self):
        torch.set_default_dtype(torch.float64)
        yield
        # this returns the default type to float32 at the end of all tests in this class in order
        # to not affect other tests.
        torch.set_default_dtype(torch.float32)

    @pytest.fixture()
    def output(self, batch, score_network):
        with torch.no_grad():
            return score_network(batch)

    @pytest.fixture()
    def translated_output(self, translated_batch, score_network):
        with torch.no_grad():
            return score_network(translated_batch)

    @pytest.fixture()
    def rotated_output(self, rotated_batch, score_network):
        with torch.no_grad():
            return score_network(rotated_batch)

    @pytest.fixture()
    def permuted_output(self, permuted_batch, score_network):
        with torch.no_grad():
            return score_network(permuted_batch)

    @pytest.fixture(params=[True, False])
    def are_basis_vectors_rotated(self, request):
        # Should the basis vectors be rotated according to the point group operation?
        return request.param

    @pytest.fixture(params=[True, False])
    def is_cell_cubic(self, request):
        # Should the basis vectors form a cube?
        return request.param

    @pytest.fixture(params=[True, False])
    def is_rotations_cubic_point_group(self, request):
        # Should the rotations be the symmetries of a cube?
        return request.param

    @pytest.fixture()
    def batch_size(self, is_rotations_cubic_point_group):
        if is_rotations_cubic_point_group:
            return len(get_cubic_point_group_symmetries())
        else:
            return 16

    @pytest.fixture()
    def basis_vectors(self, batch_size, spatial_dimension, is_cell_cubic):
        if is_cell_cubic:
            # Cubic unit cells.
            basis_vectors = (5.0 + 5.0 * torch.rand(1)) * torch.eye(
                spatial_dimension
            ).repeat(batch_size, 1, 1)
        else:
            # orthogonal boxes with dimensions between 5 and 10.
            orthogonal_boxes = torch.stack(
                [
                    torch.diag(5.0 + 5.0 * torch.rand(spatial_dimension))
                    for _ in range(batch_size)
                ]
            )
            # add a bit of noise to make the vectors not quite orthogonal
            basis_vectors = orthogonal_boxes + 0.1 * torch.randn(
                batch_size, spatial_dimension, spatial_dimension
            )

        return basis_vectors

    @pytest.fixture()
    def rotated_basis_vectors(
        self, cartesian_rotations, basis_vectors, are_basis_vectors_rotated
    ):
        # The basis vectors are defined as ROWS.
        if are_basis_vectors_rotated:
            return self.get_rotated_basis_vectors(cartesian_rotations, basis_vectors)
        else:
            return basis_vectors

    @pytest.fixture()
    def relative_coordinates(self, batch_size, number_of_atoms, spatial_dimension):
        relative_coordinates = torch.rand(
            batch_size, number_of_atoms, spatial_dimension
        )
        return relative_coordinates

    @pytest.fixture()
    def cartesian_positions(self, relative_coordinates, basis_vectors):
        return get_positions_from_coordinates(relative_coordinates, basis_vectors)

    @pytest.fixture()
    def times(self, batch_size):
        return torch.rand(batch_size, 1)

    @pytest.fixture()
    def noises(self, batch_size):
        return 0.5 * torch.rand(batch_size, 1)

    @pytest.fixture()
    def forces(self, batch_size, spatial_dimension):
        return 0.5 * torch.rand(batch_size, spatial_dimension)

    @pytest.fixture()
    def permutations(self, batch_size, number_of_atoms):
        return torch.stack([torch.randperm(number_of_atoms) for _ in range(batch_size)])

    @pytest.fixture()
    def cartesian_rotations(self, batch_size, is_rotations_cubic_point_group):
        if is_rotations_cubic_point_group:
            return get_cubic_point_group_symmetries()
        else:
            return o3.rand_matrix(batch_size)

    @pytest.fixture()
    def cartesian_translations(
        self, batch_size, number_of_atoms, spatial_dimension, basis_vectors
    ):
        batch_relative_coordinates_translations = torch.rand(
            batch_size, spatial_dimension
        )

        batch_cartesian_translations = []
        for t, cell in zip(batch_relative_coordinates_translations, basis_vectors):
            batch_cartesian_translations.append(t @ cell)

        batch_cartesian_translations = torch.stack(batch_cartesian_translations)

        cartesian_translations = torch.repeat_interleave(
            batch_cartesian_translations.unsqueeze(1), number_of_atoms, dim=1
        )
        return cartesian_translations

    @pytest.fixture()
    def batch(
        self,
        relative_coordinates,
        cartesian_positions,
        atom_types,
        basis_vectors,
        times,
        noises,
        forces,
    ):
        return self.create_batch(
            relative_coordinates,
            cartesian_positions,
            atom_types,
            basis_vectors,
            times,
            noises,
            forces,
        )

    @pytest.fixture()
    def translated_batch(
        self,
        cartesian_translations,
        relative_coordinates,
        cartesian_positions,
        atom_types,
        basis_vectors,
        times,
        noises,
        forces,
    ):
        translated_cartesian_positions = cartesian_positions + cartesian_translations
        reciprocal_basis_vectors = get_reciprocal_basis_vectors(basis_vectors)

        new_relative_coordinates = map_relative_coordinates_to_unit_cell(
            get_relative_coordinates_from_cartesian_positions(
                translated_cartesian_positions, reciprocal_basis_vectors
            )
        )
        new_cartesian_positions = get_positions_from_coordinates(
            new_relative_coordinates, basis_vectors
        )
        return self.create_batch(
            new_relative_coordinates,
            new_cartesian_positions,
            atom_types,
            basis_vectors,
            times,
            noises,
            forces,
        )

    @pytest.fixture()
    def rotated_batch(
        self,
        rotated_basis_vectors,
        cartesian_rotations,
        relative_coordinates,
        cartesian_positions,
        atom_types,
        basis_vectors,
        times,
        noises,
        forces,
    ):
        rotated_cartesian_positions = self.apply_rotation_to_configuration(
            cartesian_rotations, cartesian_positions
        )

        rotated_reciprocal_basis_vectors = get_reciprocal_basis_vectors(
            rotated_basis_vectors
        )

        rel_coords = get_relative_coordinates_from_cartesian_positions(
            rotated_cartesian_positions, rotated_reciprocal_basis_vectors
        )
        new_relative_coordinates = map_relative_coordinates_to_unit_cell(rel_coords)
        new_cartesian_positions = get_positions_from_coordinates(
            new_relative_coordinates, rotated_reciprocal_basis_vectors
        )
        return self.create_batch(
            new_relative_coordinates,
            new_cartesian_positions,
            atom_types,
            rotated_basis_vectors,
            times,
            noises,
            forces,
        )

    @pytest.fixture()
    def permuted_batch(
        self,
        permutations,
        relative_coordinates,
        cartesian_positions,
        atom_types,
        basis_vectors,
        times,
        noises,
        forces,
    ):
        batch_size = relative_coordinates.shape[0]

        new_cartesian_positions = torch.stack(
            [
                cartesian_positions[batch_idx, permutations[batch_idx], :]
                for batch_idx in range(batch_size)
            ]
        )

        new_relative_coordinates = torch.stack(
            [
                relative_coordinates[batch_idx, permutations[batch_idx], :]
                for batch_idx in range(batch_size)
            ]
        )

        new_atom_types = torch.stack(
            [
                atom_types[batch_idx, permutations[batch_idx]]
                for batch_idx in range(batch_size)
            ]
        )
        return self.create_batch(
            new_relative_coordinates,
            new_cartesian_positions,
            new_atom_types,
            basis_vectors,
            times,
            noises,
            forces,
        )

    def test_translation_invariance(self, output, translated_output):
        torch.testing.assert_close(output, translated_output)

    @pytest.fixture()
    def rotated_scores_should_match(
        self, is_rotations_cubic_point_group, is_cell_cubic, are_basis_vectors_rotated
    ):
        # The rotated scores should match the original scores if the basis vectors are rotated.
        # If the basis vectors are NOT rotated, only a cubic unit cell (and cubic symmetries) should match.
        should_match = are_basis_vectors_rotated or (
            is_cell_cubic and is_rotations_cubic_point_group
        )
        return should_match

    @pytest.fixture()
    def atom_output_should_be_tested_for_rotational_equivariance(self):
        return True

    def test_rotation_equivariance(
        self,
        output,
        rotated_output,
        basis_vectors,
        rotated_basis_vectors,
        cartesian_rotations,
        rotated_scores_should_match,
        atom_output_should_be_tested_for_rotational_equivariance,
    ):

        # The score is ~ nabla_x ln P. There must a be a basis change to turn it into a cartesian score of the
        # form ~ nabla_r ln P.
        reciprocal_basis_vectors = get_reciprocal_basis_vectors(basis_vectors)
        cartesian_scores = einops.einsum(
            reciprocal_basis_vectors,
            output.X,
            "batch alpha i, batch natoms i -> batch natoms alpha",
        ).contiguous()

        reciprocal_rotated_basis_vectors = get_reciprocal_basis_vectors(
            rotated_basis_vectors
        )
        rotated_cartesian_scores = einops.einsum(
            reciprocal_rotated_basis_vectors,
            rotated_output.X,
            "batch alpha i, batch natoms i -> batch natoms alpha",
        ).contiguous()

        expected_rotated_cartesian_scores = self.apply_rotation_to_configuration(
            cartesian_rotations, cartesian_scores
        )

        if rotated_scores_should_match:
            torch.testing.assert_close(
                expected_rotated_cartesian_scores, rotated_cartesian_scores
            )
            torch.testing.assert_close(output.L, rotated_output.L)

            if atom_output_should_be_tested_for_rotational_equivariance:
                torch.testing.assert_close(output.A, rotated_output.A)
        else:
            with pytest.raises(AssertionError):
                torch.testing.assert_close(
                    expected_rotated_cartesian_scores, rotated_cartesian_scores
                )
            # TODO: it's not clear what the expectation should be for A and L in this case...

    def test_permutation_equivariance(
        self, output, permuted_output, batch_size, permutations
    ):

        expected_output_x = torch.stack(
            [
                output.X[batch_idx, permutations[batch_idx], :]
                for batch_idx in range(batch_size)
            ]
        )

        expected_output_a = torch.stack(
            [
                output.A[batch_idx, permutations[batch_idx]]
                for batch_idx in range(batch_size)
            ]
        )

        expected_permuted_output = AXL(
            A=expected_output_a, X=expected_output_x, L=output.L
        )

        torch.testing.assert_close(expected_permuted_output, permuted_output)


class TestEquivarianceDiffusionMACE(BaseTestScoreEquivariance):

    @pytest.fixture()
    def score_network_parameters(
        self, number_of_atoms, num_atom_types, spatial_dimension
    ):
        return DiffusionMACEScoreNetworkParameters(
            spatial_dimension=spatial_dimension,
            number_of_atoms=number_of_atoms,
            num_atom_types=num_atom_types,
            r_max=3.0,
            num_bessel=4,
            num_polynomial_cutoff=3,
            hidden_irreps="8x0e + 8x1o",
            mlp_irreps="8x0e",
            number_of_mlp_layers=1,
            correlation=2,
            radial_MLP=[8, 8, 8],
        )

    @pytest.fixture()
    def score_network(self, score_network_parameters):
        return DiffusionMACEScoreNetwork(score_network_parameters)


# TODO: This model has not yet been adapted to multiple atom types, and so is not ready for atom_type related tests.
#  This test should be updated if the model is adapted to multiple atom types.
class TestEquivarianceMaceWithEquivariantScorePredictionHead(BaseTestScoreEquivariance):

    @pytest.fixture()
    def atom_output_should_be_tested_for_rotational_equivariance(self):
        return False

    @pytest.fixture()
    def score_network_parameters(
        self,
        spatial_dimension,
        number_of_atoms,
        num_atom_types,
    ):
        prediction_head_parameters = MaceEquivariantScorePredictionHeadParameters(
            spatial_dimension=spatial_dimension,
            number_of_layers=2,
        )

        return MACEScoreNetworkParameters(
            spatial_dimension=spatial_dimension,
            number_of_atoms=number_of_atoms,
            num_atom_types=num_atom_types,
            r_max=3.0,
            prediction_head_parameters=prediction_head_parameters,
        )

    @pytest.fixture()
    def score_network(self, score_network_parameters):
        return MACEScoreNetwork(score_network_parameters)


class TestEquivarianceEGNN(BaseTestScoreEquivariance):

    @pytest.fixture(params=[True, False])
    def normalize(self, request):
        return request.param

    @pytest.fixture(params=[1, 2, 3])
    def nbloch(self, request):
        return request.param

    @pytest.fixture(params=[("fully_connected", None), ("radial_cutoff", 3.0)])
    def score_network_parameters(self, request, num_atom_types, normalize, nbloch):
        edges, radial_cutoff = request.param
        return EGNNScoreNetworkParameters(
            number_of_bloch_wave_shells=nbloch,
            edges=edges,
            radial_cutoff=radial_cutoff,
            num_atom_types=num_atom_types,
            normalize=normalize,
        )

    @pytest.fixture()
    def score_network(self, score_network_parameters):
        score_network = EGNNScoreNetwork(score_network_parameters)
        return score_network


# Some of the tests below FAIL because the optimal transport is imperfect. It is still
# interesting to keep these around if we ever find a better OT heuristic.
@pytest.mark.skip()
@pytest.mark.parametrize("num_atom_types", [0])
@pytest.mark.parametrize(
    "is_cell_cubic, is_rotations_cubic_point_group", [(True, True)]
)
class TestEquivarianceEquivariantAnalyticalScoreNetwork(BaseTestScoreEquivariance):

    @pytest.fixture()
    def score_network_parameters(
        self, number_of_atoms, num_atom_types, spatial_dimension
    ):
        random = torch.rand(number_of_atoms, spatial_dimension).numpy()
        equilibrium_relative_coordinates = list(list(x) for x in random)

        params = EquivariantAnalyticalScoreNetworkParameters(
            number_of_atoms=number_of_atoms,
            num_atom_types=num_atom_types,
            spatial_dimension=spatial_dimension,
            kmax=5,
            use_point_group_symmetries=True,
            equilibrium_relative_coordinates=equilibrium_relative_coordinates,
            sigma_d=0.01,
        )
        return params

    @pytest.fixture()
    def score_network(self, score_network_parameters):
        score_network = EquivariantAnalyticalScoreNetwork(score_network_parameters)
        return score_network
