import pytest
import torch
from e3nn import o3
from mace.modules import gate_dict, interaction_classes

from diffusion_for_multi_scale_molecular_dynamics.models.diffusion_mace import (
    DiffusionMACE, LinearVectorReadoutBlock, input_to_diffusion_mace)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    CARTESIAN_FORCES, NOISE, NOISY_CARTESIAN_POSITIONS,
    NOISY_RELATIVE_COORDINATES, TIME, UNIT_CELL)
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import (
    get_positions_from_coordinates, get_reciprocal_basis_vectors,
    get_relative_coordinates_from_cartesian_positions,
    map_relative_coordinates_to_unit_cell)


def test_linear_vector_readout_block():

    batch_size = 10
    vector_output_dimension = 3
    irreps_in = o3.Irreps("16x0e + 12x1o + 14x2e")

    vector_readout = LinearVectorReadoutBlock(irreps_in)

    input_features = irreps_in.randn(batch_size, -1)

    output_features = vector_readout(input_features)

    assert output_features.shape == (batch_size, vector_output_dimension)


class TestDiffusionMace:
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
        torch.manual_seed(234233)

    @pytest.fixture(scope="class")
    def batch_size(self):
        return 4

    @pytest.fixture(scope="class")
    def number_of_atoms(self):
        return 8

    @pytest.fixture(scope="class")
    def spatial_dimension(self):
        return 3

    @pytest.fixture(scope="class")
    def basis_vectors(self, batch_size, spatial_dimension):
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

    @pytest.fixture(scope="class")
    def reciprocal_basis_vectors(self, basis_vectors):
        return get_reciprocal_basis_vectors(basis_vectors)

    @pytest.fixture(scope="class")
    def relative_coordinates(self, batch_size, number_of_atoms, spatial_dimension):
        relative_coordinates = torch.rand(
            batch_size, number_of_atoms, spatial_dimension
        )
        return relative_coordinates

    @pytest.fixture(scope="class")
    def cartesian_positions(self, relative_coordinates, basis_vectors):
        return get_positions_from_coordinates(relative_coordinates, basis_vectors)

    @pytest.fixture(scope="class")
    def times(self, batch_size):
        return torch.rand(batch_size, 1)

    @pytest.fixture(scope="class")
    def noises(self, batch_size):
        return 0.5 * torch.rand(batch_size, 1)

    @pytest.fixture(scope="class")
    def forces(self, batch_size, spatial_dimension):
        return 0.5 * torch.rand(batch_size, spatial_dimension)

    @pytest.fixture(scope="class")
    def batch(
        self,
        relative_coordinates,
        cartesian_positions,
        basis_vectors,
        times,
        noises,
        forces,
    ):
        batch = {
            NOISY_RELATIVE_COORDINATES: relative_coordinates,
            NOISY_CARTESIAN_POSITIONS: cartesian_positions,
            TIME: times,
            NOISE: noises,
            UNIT_CELL: basis_vectors,
            CARTESIAN_FORCES: forces,
        }
        return batch

    @pytest.fixture(scope="class")
    def cartesian_rotations(self, batch_size):
        return o3.rand_matrix(batch_size)

    @pytest.fixture(scope="class")
    def permutations(self, batch_size, number_of_atoms):
        return torch.stack([torch.randperm(number_of_atoms) for _ in range(batch_size)])

    @pytest.fixture(scope="class")
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
    def r_max(self):
        return 3.0

    @pytest.fixture()
    def hyperparameters(self, r_max):

        hps = dict(
            r_max=r_max,
            num_bessel=8,
            num_polynomial_cutoff=5,
            num_edge_hidden_layers=0,
            edge_hidden_irreps=o3.Irreps("8x0e"),
            max_ell=2,
            num_elements=1,
            atomic_numbers=[14],
            interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
            interaction_cls_first=interaction_classes["RealAgnosticInteractionBlock"],
            num_interactions=2,
            hidden_irreps=o3.Irreps("8x0e + 8x1o + 8x2e"),
            mlp_irreps=o3.Irreps("8x0e"),
            number_of_mlp_layers=2,
            avg_num_neighbors=1,
            correlation=2,
            gate=gate_dict["silu"],
            radial_MLP=[8, 8, 8],
            radial_type="bessel",
        )
        return hps

    @pytest.fixture()
    def diffusion_mace(self, hyperparameters):
        diffusion_mace = DiffusionMACE(**hyperparameters)
        diffusion_mace.eval()
        return diffusion_mace

    @pytest.fixture()
    def graph_input(self, batch, r_max):
        return input_to_diffusion_mace(batch, radial_cutoff=r_max)

    @pytest.fixture()
    def cartesian_scores(
        self,
        graph_input,
        diffusion_mace,
        batch_size,
        number_of_atoms,
        spatial_dimension,
    ):
        flat_cartesian_scores = diffusion_mace(graph_input)
        return flat_cartesian_scores.reshape(
            batch_size, number_of_atoms, spatial_dimension
        )

    @pytest.fixture()
    def translated_graph_input(
        self,
        batch,
        r_max,
        basis_vectors,
        reciprocal_basis_vectors,
        cartesian_translations,
    ):

        translated_batch = dict(batch)

        original_cartesian_positions = translated_batch[NOISY_CARTESIAN_POSITIONS]
        translated_cartesian_positions = (
            original_cartesian_positions + cartesian_translations
        )

        rel_coords = get_relative_coordinates_from_cartesian_positions(
            translated_cartesian_positions, reciprocal_basis_vectors
        )
        new_relative_coordinates = map_relative_coordinates_to_unit_cell(rel_coords)

        new_cartesian_positions = get_positions_from_coordinates(
            new_relative_coordinates, basis_vectors
        )

        translated_batch[NOISY_CARTESIAN_POSITIONS] = new_cartesian_positions
        translated_batch[NOISY_RELATIVE_COORDINATES] = new_relative_coordinates

        return input_to_diffusion_mace(translated_batch, radial_cutoff=r_max)

    @pytest.fixture()
    def translated_cartesian_scores(
        self,
        diffusion_mace,
        batch_size,
        number_of_atoms,
        spatial_dimension,
        basis_vectors,
        translated_graph_input,
    ):
        flat_translated_cartesian_scores = diffusion_mace(translated_graph_input)
        return flat_translated_cartesian_scores.reshape(
            batch_size, number_of_atoms, spatial_dimension
        )

    @pytest.fixture()
    def rotated_graph_input(
        self, batch, r_max, basis_vectors, reciprocal_basis_vectors, cartesian_rotations
    ):
        rotated_batch = dict(batch)

        original_cartesian_positions = rotated_batch[NOISY_CARTESIAN_POSITIONS]
        original_basis_vectors = rotated_batch[UNIT_CELL]

        rotated_cartesian_positions = torch.matmul(
            original_cartesian_positions, cartesian_rotations.transpose(2, 1)
        )

        rotated_basis_vectors = torch.matmul(
            original_basis_vectors, cartesian_rotations.transpose(2, 1)
        )
        rotated_reciprocal_basis_vectors = get_reciprocal_basis_vectors(
            rotated_basis_vectors
        )

        rel_coords = get_relative_coordinates_from_cartesian_positions(
            rotated_cartesian_positions, rotated_reciprocal_basis_vectors
        )
        new_relative_coordinates = map_relative_coordinates_to_unit_cell(rel_coords)
        new_cartesian_positions = get_positions_from_coordinates(
            new_relative_coordinates, rotated_basis_vectors
        )

        rotated_batch[NOISY_CARTESIAN_POSITIONS] = new_cartesian_positions
        rotated_batch[NOISY_RELATIVE_COORDINATES] = new_relative_coordinates
        rotated_batch[UNIT_CELL] = rotated_basis_vectors

        return input_to_diffusion_mace(rotated_batch, radial_cutoff=r_max)

    @pytest.fixture()
    def rotated_cartesian_scores(
        self,
        diffusion_mace,
        batch_size,
        number_of_atoms,
        spatial_dimension,
        rotated_graph_input,
    ):
        flat_rotated_cartesian_scores = diffusion_mace(rotated_graph_input)
        return flat_rotated_cartesian_scores.reshape(
            batch_size, number_of_atoms, spatial_dimension
        )

    @pytest.fixture()
    def permuted_graph_input(self, batch_size, batch, r_max, permutations):
        permuted_batch = dict(batch)

        for position_key in [NOISY_CARTESIAN_POSITIONS, NOISY_RELATIVE_COORDINATES]:
            pos = permuted_batch[position_key]
            permuted_pos = torch.stack(
                [
                    pos[batch_idx, permutations[batch_idx], :]
                    for batch_idx in range(batch_size)
                ]
            )
            permuted_batch[position_key] = permuted_pos

        return input_to_diffusion_mace(permuted_batch, radial_cutoff=r_max)

    @pytest.fixture()
    def permuted_cartesian_scores(
        self,
        diffusion_mace,
        batch_size,
        number_of_atoms,
        spatial_dimension,
        permuted_graph_input,
    ):
        flat_permuted_cartesian_scores = diffusion_mace(permuted_graph_input)
        return flat_permuted_cartesian_scores.reshape(
            batch_size, number_of_atoms, spatial_dimension
        )

    def test_translation_invariance(
        self, cartesian_scores, translated_cartesian_scores
    ):
        torch.testing.assert_close(translated_cartesian_scores, cartesian_scores)

    def test_rotation_equivariance(
        self, cartesian_scores, rotated_cartesian_scores, cartesian_rotations
    ):
        vector_irreps = o3.Irreps("1o")
        d_matrices = vector_irreps.D_from_matrix(cartesian_rotations)

        expected_rotated_cartesian_scores = torch.matmul(
            cartesian_scores, d_matrices.transpose(2, 1)
        )
        torch.testing.assert_close(
            expected_rotated_cartesian_scores, rotated_cartesian_scores
        )

    def test_permutation_equivariance(
        self, cartesian_scores, permuted_cartesian_scores, batch_size, permutations
    ):

        expected_permuted_cartesian_scores = torch.stack(
            [
                cartesian_scores[batch_idx, permutations[batch_idx], :]
                for batch_idx in range(batch_size)
            ]
        )

        torch.testing.assert_close(
            expected_permuted_cartesian_scores, permuted_cartesian_scores
        )

    def test_time_dependence(self, batch, r_max, diffusion_mace):

        graph_input = input_to_diffusion_mace(batch, radial_cutoff=r_max)
        flat_cartesian_scores1 = diffusion_mace(graph_input)
        flat_cartesian_scores2 = diffusion_mace(graph_input)

        # apply twice on the same input, get the same answer?
        torch.testing.assert_close(flat_cartesian_scores1, flat_cartesian_scores2)

        new_time_batch = dict(batch)
        new_time_batch[TIME] = torch.rand(batch[TIME].shape)
        new_time_batch[NOISE] = torch.rand(batch[NOISE].shape)
        new_graph_input = input_to_diffusion_mace(new_time_batch, radial_cutoff=r_max)
        new_flat_cartesian_scores = diffusion_mace(new_graph_input)

        # Different times, different results?
        with pytest.raises(AssertionError):
            torch.testing.assert_close(
                new_flat_cartesian_scores, flat_cartesian_scores1
            )
