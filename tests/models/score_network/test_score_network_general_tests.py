from dataclasses import asdict, dataclass, fields

import einops
import numpy as np
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.diffusion_mace_score_network import (
    DiffusionMACEScoreNetwork, DiffusionMACEScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.egnn_score_network import (
    EGNNScoreNetwork, EGNNScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.mace_score_network import (
    MACEScoreNetwork, MACEScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.mlp_score_network import (
    MLPScoreNetwork, MLPScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network_factory import \
    create_score_network_parameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_prediction_head import (
    MaceEquivariantScorePredictionHeadParameters,
    MaceMLPScorePredictionHeadParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, CARTESIAN_FORCES, NOISE, NOISY_AXL_COMPOSITION, TIME, UNIT_CELL)
from tests.fake_data_utils import generate_random_string
from tests.models.score_network.base_test_score_network import \
    BaseTestScoreNetwork


class BaseScoreNetworkGeneralTests(BaseTestScoreNetwork):
    """Base score network general tests.

    Base class to run a battery of tests on a score network. To test a specific score network class, this base class
    should be extended by implementing a 'score_network' fixture that instantiates the score network class of interest.
    """

    @staticmethod
    def assert_parameters_are_the_same(parameters1: dataclass, parameters2: dataclass):
        """Compare dataclasses explicitly as a workaround for the potential presence of numpy arrays."""
        assert type(parameters1) is type(parameters2)

        for field in fields(parameters1):
            value1 = getattr(parameters1, field.name)
            value2 = getattr(parameters2, field.name)

            assert type(value1) is type(value2)

            if type(value1) is np.ndarray:
                np.testing.assert_array_equal(value1, value2)
            else:
                assert value1 == value2

    @pytest.fixture(params=[2, 3, 16])
    def num_atom_types(self, request):
        return request.param

    @pytest.fixture
    def unique_elements(self, num_atom_types):
        return [generate_random_string(size=3) for _ in range(num_atom_types)]

    @pytest.fixture()
    def score_network_parameters(self, *args):
        raise NotImplementedError(
            "This fixture must be implemented in the derived class."
        )

    @pytest.fixture()
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

    @pytest.fixture
    def relative_coordinates(
        self, batch_size, number_of_atoms, spatial_dimension, basis_vectors
    ):
        relative_coordinates = torch.rand(
            batch_size, number_of_atoms, spatial_dimension
        )
        return relative_coordinates

    @pytest.fixture
    def atom_types(self, batch_size, number_of_atoms, num_atom_types):
        atom_types = torch.randint(0, num_atom_types + 1, (batch_size, number_of_atoms))
        return atom_types

    @pytest.fixture()
    def expected_score_shape(
        self, batch_size, number_of_atoms, spatial_dimension, num_atom_types
    ):
        first_dims = (
            batch_size,
            number_of_atoms,
        )
        return {
            "X": first_dims + (spatial_dimension,),
            "A": first_dims + (num_atom_types + 1,),
        }

    @pytest.fixture
    def cartesian_forces(
        self, batch_size, number_of_atoms, spatial_dimension, basis_vectors
    ):
        cartesian_forces = torch.rand(batch_size, number_of_atoms, spatial_dimension)
        return cartesian_forces

    @pytest.fixture
    def times(self, batch_size):
        times = torch.rand(batch_size, 1)
        return times

    @pytest.fixture
    def noises(self, batch_size):
        return torch.rand(batch_size, 1)

    @pytest.fixture
    def lattice_parameters(self, basis_vectors, spatial_dimension):
        lattice_dim = int(spatial_dimension * (spatial_dimension + 1) / 2)
        lattice_params = torch.zeros(basis_vectors.shape[0], lattice_dim)
        lattice_params[:, :spatial_dimension] = torch.diagonal(basis_vectors, dim1=-2, dim2=-1)
        return lattice_params

    @pytest.fixture()
    def batch(
        self,
        relative_coordinates,
        cartesian_forces,
        times,
        noises,
        basis_vectors,
        lattice_parameters,
        atom_types,
    ):
        return {
            NOISY_AXL_COMPOSITION: AXL(
                A=atom_types,
                X=relative_coordinates,
                L=lattice_parameters,
            ),
            TIME: times,
            UNIT_CELL: basis_vectors,  # TODO remove this
            NOISE: noises,
            CARTESIAN_FORCES: cartesian_forces,
        }

    @pytest.fixture()
    def global_parameters_dictionary(self, spatial_dimension, unique_elements):
        return dict(spatial_dimension=spatial_dimension, irrelevant=123, elements=unique_elements)

    @pytest.fixture()
    def score_network_dictionary(
        self, score_network_parameters, global_parameters_dictionary
    ):
        dictionary = asdict(score_network_parameters)
        for key in global_parameters_dictionary.keys():
            if key in dictionary:
                dictionary.pop(key)
        return dictionary

    def test_output_shape(self, score_network, batch, expected_score_shape):
        scores = score_network(batch)
        assert scores.X.shape == expected_score_shape["X"]
        assert scores.A.shape == expected_score_shape["A"]

    def test_create_score_network_parameters(
        self,
        score_network_parameters,
        score_network_dictionary,
        global_parameters_dictionary,
    ):
        computed_score_network_parameters = create_score_network_parameters(
            score_network_dictionary, global_parameters_dictionary
        )
        self.assert_parameters_are_the_same(
            computed_score_network_parameters, score_network_parameters
        )

    def test_consistent_output(self, batch, score_network):
        # apply twice on the same input, get the same answer?
        with torch.no_grad():
            output1 = score_network(batch)
            output2 = score_network(batch)

        torch.testing.assert_close(output1, output2)

    def test_time_dependence(self, batch, score_network):
        # Different times, different results?
        new_time_batch = dict(batch)
        new_time_batch[TIME] = torch.rand(batch[TIME].shape)
        new_time_batch[NOISE] = torch.rand(batch[NOISE].shape)
        with torch.no_grad():
            output1 = score_network(batch)
            output2 = score_network(new_time_batch)

        with pytest.raises(AssertionError):
            torch.testing.assert_close(output1, output2)


@pytest.mark.parametrize("spatial_dimension", [2, 3])
@pytest.mark.parametrize("n_hidden_dimensions", [1, 2, 3])
@pytest.mark.parametrize("hidden_dimensions_size", [8, 16])
@pytest.mark.parametrize("embedding_dimensions_size", [4, 12])
class TestMLPScoreNetwork(BaseScoreNetworkGeneralTests):

    @pytest.fixture()
    def score_network_parameters(
        self,
        number_of_atoms,
        spatial_dimension,
        num_atom_types,
        embedding_dimensions_size,
        n_hidden_dimensions,
        hidden_dimensions_size,
    ):
        return MLPScoreNetworkParameters(
            spatial_dimension=spatial_dimension,
            number_of_atoms=number_of_atoms,
            num_atom_types=num_atom_types,
            noise_embedding_dimensions_size=embedding_dimensions_size,
            time_embedding_dimensions_size=embedding_dimensions_size,
            atom_type_embedding_dimensions_size=embedding_dimensions_size,
            n_hidden_dimensions=n_hidden_dimensions,
            hidden_dimensions_size=hidden_dimensions_size,
        )

    @pytest.fixture()
    def score_network(self, score_network_parameters):
        return MLPScoreNetwork(score_network_parameters)


@pytest.mark.parametrize("n_hidden_dimensions", [2])
@pytest.mark.parametrize("hidden_dimensions_size", [8])
class TestMACEScoreNetworkMLPHead(BaseScoreNetworkGeneralTests):

    @pytest.fixture()
    def prediction_head_parameters(
        self, spatial_dimension, n_hidden_dimensions, hidden_dimensions_size
    ):
        prediction_head_parameters = MaceMLPScorePredictionHeadParameters(
            spatial_dimension=spatial_dimension,
            hidden_dimensions_size=hidden_dimensions_size,
            n_hidden_dimensions=n_hidden_dimensions,
        )
        return prediction_head_parameters

    @pytest.fixture()
    def score_network_parameters(
        self,
        number_of_atoms,
        spatial_dimension,
        num_atom_types,
        prediction_head_parameters,
    ):
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


@pytest.mark.parametrize("spatial_dimension", [3])
class TestMACEScoreNetworkEquivariantHead(BaseScoreNetworkGeneralTests):
    @pytest.fixture()
    def prediction_head_parameters(self, spatial_dimension):
        prediction_head_parameters = MaceEquivariantScorePredictionHeadParameters(
            spatial_dimension=spatial_dimension, number_of_layers=2
        )
        return prediction_head_parameters

    @pytest.fixture()
    def score_network_parameters(
        self,
        number_of_atoms,
        spatial_dimension,
        num_atom_types,
        prediction_head_parameters,
    ):
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


class TestDiffusionMACEScoreNetwork(BaseScoreNetworkGeneralTests):
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


class TestEGNNScoreNetwork(BaseScoreNetworkGeneralTests):

    @pytest.fixture(params=[("fully_connected", None), ("radial_cutoff", 3.0)])
    def score_network_parameters(self, request, num_atom_types):
        edges, radial_cutoff = request.param
        return EGNNScoreNetworkParameters(
            edges=edges, radial_cutoff=radial_cutoff, num_atom_types=num_atom_types
        )

    @pytest.fixture()
    def score_network(self, score_network_parameters):
        score_network = EGNNScoreNetwork(score_network_parameters)
        return score_network

    @pytest.mark.parametrize(
        "edges, radial_cutoff", [("fully_connected", 3.0), ("radial_cutoff", None)]
    )
    def test_score_network_parameters(self, edges, radial_cutoff, num_atom_types):
        score_network_parameters = EGNNScoreNetworkParameters(
            edges=edges, radial_cutoff=radial_cutoff, num_atom_types=num_atom_types
        )
        with pytest.raises(AssertionError):
            # Check that the code crashes when inconsistent parameters are fed in.
            EGNNScoreNetwork(score_network_parameters)

    def test_create_block_diagonal_projection_matrices(
        self, score_network, spatial_dimension
    ):
        expected_matrices = []
        for space_idx in range(spatial_dimension):
            matrix = torch.zeros(2 * spatial_dimension, 2 * spatial_dimension)
            matrix[2 * space_idx + 1, 2 * space_idx] = 1.0
            matrix[2 * space_idx, 2 * space_idx + 1] = -1.0
            expected_matrices.append(matrix)

        expected_matrices = torch.stack(expected_matrices)

        computed_matrices = score_network._create_block_diagonal_projection_matrices(
            spatial_dimension
        )

        torch.testing.assert_close(computed_matrices, expected_matrices)

    @pytest.fixture()
    def flat_relative_coordinates(self, batch):
        relative_coordinates = batch[NOISY_AXL_COMPOSITION].X
        flat_relative_coordinates = einops.rearrange(
            relative_coordinates, "batch natom space -> (batch natom) space"
        )
        return flat_relative_coordinates

    @pytest.fixture()
    def expected_euclidean_positions(self, flat_relative_coordinates):
        expected_euclidean_positions = []
        for relative_coordinates in flat_relative_coordinates:
            euclidean_position = []
            for f in relative_coordinates:
                cos = torch.cos(2 * torch.pi * f)
                sin = torch.sin(2 * torch.pi * f)
                euclidean_position.append(cos)
                euclidean_position.append(sin)

            expected_euclidean_positions.append(torch.tensor(euclidean_position))

        expected_euclidean_positions = torch.stack(expected_euclidean_positions)
        return expected_euclidean_positions

    def test_get_euclidean_positions(
        self, score_network, flat_relative_coordinates, expected_euclidean_positions
    ):
        computed_euclidean_positions = score_network._get_euclidean_positions(
            flat_relative_coordinates
        )
        torch.testing.assert_close(
            expected_euclidean_positions, computed_euclidean_positions
        )
