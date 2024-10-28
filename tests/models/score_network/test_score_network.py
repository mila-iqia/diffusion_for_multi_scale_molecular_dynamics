import itertools
from copy import deepcopy
from dataclasses import asdict, dataclass, fields

import einops
import numpy as np
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.diffusion_mace_score_network import (
    DiffusionMACEScoreNetwork,
    DiffusionMACEScoreNetworkParameters,
)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.egnn_score_network import (
    EGNNScoreNetwork,
    EGNNScoreNetworkParameters,
)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.mace_score_network import (
    MACEScoreNetwork,
    MACEScoreNetworkParameters,
)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.mlp_score_network import (
    MLPScoreNetwork,
    MLPScoreNetworkParameters,
)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network import (
    ScoreNetwork,
    ScoreNetworkParameters,
)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network_factory import (
    create_score_network_parameters,
)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_prediction_head import (
    MaceEquivariantScorePredictionHeadParameters,
    MaceMLPScorePredictionHeadParameters,
)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL,
    CARTESIAN_FORCES,
    NOISE,
    NOISY_AXL,
    TIME,
    UNIT_CELL,
)
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import (
    map_relative_coordinates_to_unit_cell,
)


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


@pytest.mark.parametrize("spatial_dimension", [2, 3])
@pytest.mark.parametrize("num_atom_types", [3])
class TestScoreNetworkCheck:

    @pytest.fixture(scope="class", autouse=True)
    def set_random_seed(self):
        torch.manual_seed(123)

    @pytest.fixture()
    def base_score_network(self, spatial_dimension, num_atom_types):
        return ScoreNetwork(
            ScoreNetworkParameters(
                architecture="dummy",
                spatial_dimension=spatial_dimension,
                num_atom_types=num_atom_types,
            )
        )

    @pytest.fixture()
    def good_batch(self, spatial_dimension, num_atom_types):
        batch_size = 16
        relative_coordinates = torch.rand(batch_size, 8, spatial_dimension)
        times = torch.rand(batch_size, 1)
        noises = torch.rand(batch_size, 1)
        unit_cell = torch.rand(batch_size, spatial_dimension, spatial_dimension)
        atom_types = torch.randint(0, num_atom_types + 1, (batch_size, 8))
        return {
            NOISY_AXL: AXL(
                A=atom_types, X=relative_coordinates, L=torch.zeros_like(atom_types)
            ),
            TIME: times,
            NOISE: noises,
            UNIT_CELL: unit_cell,
        }

    @pytest.fixture()
    def bad_batch(self, good_batch, problem):

        bad_batch_dict = dict(good_batch)

        match problem:
            case "position_name":
                bad_batch_dict["bad_position_name"] = bad_batch_dict[NOISY_AXL]
                del bad_batch_dict[NOISY_AXL]

            case "position_shape":
                shape = bad_batch_dict[NOISY_AXL].X.shape
                bad_batch_dict[NOISY_AXL] = AXL(
                    A=bad_batch_dict[NOISY_AXL].A,
                    X=bad_batch_dict[NOISY_AXL].X.reshape(
                        shape[0], shape[1] // 2, shape[2] * 2
                    ),
                    L=bad_batch_dict[NOISY_AXL].L,
                )

            case "position_range1":
                bad_positions = bad_batch_dict[NOISY_AXL].X
                bad_positions[0, 0, 0] = 1.01
                bad_batch_dict[NOISY_AXL] = AXL(
                    A=bad_batch_dict[NOISY_AXL].A,
                    X=bad_positions,
                    L=bad_batch_dict[NOISY_AXL].L,
                )

            case "position_range2":
                bad_positions = bad_batch_dict[NOISY_AXL].X
                bad_positions[1, 0, 0] = -0.01
                bad_batch_dict[NOISY_AXL] = AXL(
                    A=bad_batch_dict[NOISY_AXL].A,
                    X=bad_positions,
                    L=bad_batch_dict[NOISY_AXL].L,
                )

            case "time_name":
                bad_batch_dict["bad_time_name"] = bad_batch_dict[TIME]
                del bad_batch_dict[TIME]

            case "time_shape":
                shape = bad_batch_dict[TIME].shape
                bad_batch_dict[TIME] = bad_batch_dict[TIME].reshape(
                    shape[0] // 2, shape[1] * 2
                )

            case "noise_name":
                bad_batch_dict["bad_noise_name"] = bad_batch_dict[NOISE]
                del bad_batch_dict[NOISE]

            case "noise_shape":
                shape = bad_batch_dict[NOISE].shape
                bad_batch_dict[NOISE] = bad_batch_dict[NOISE].reshape(
                    shape[0] // 2, shape[1] * 2
                )

            case "time_range1":
                bad_batch_dict[TIME][5, 0] = 2.00
            case "time_range2":
                bad_batch_dict[TIME][0, 0] = -0.05

            case "cell_name":
                bad_batch_dict["bad_unit_cell_key"] = bad_batch_dict[UNIT_CELL]
                del bad_batch_dict[UNIT_CELL]

            case "cell_shape":
                shape = bad_batch_dict[UNIT_CELL].shape
                bad_batch_dict[UNIT_CELL] = bad_batch_dict[UNIT_CELL].reshape(
                    shape[0] // 2, shape[1] * 2, shape[2]
                )
            # TODO errors with atom types

        return bad_batch_dict

    def test_check_batch_good(self, base_score_network, good_batch):
        base_score_network._check_batch(good_batch)

    @pytest.mark.parametrize(
        "problem",
        [
            "position_name",
            "time_name",
            "position_shape",
            "time_shape",
            "noise_name",
            "noise_shape",
            "position_range1",
            "position_range2",
            "time_range1",
            "time_range2",
            "cell_name",
            "cell_shape",
        ],
    )
    def test_check_batch_bad(self, base_score_network, bad_batch):
        with pytest.raises(AssertionError):
            base_score_network._check_batch(bad_batch)


class BaseTestScoreNetwork:
    """Base Test Score Network.

    Base class to run a battery of tests on a score network. To test a specific score network class, this base class
    should be extended by implementing a 'score_network' fixture that instantiates the score network class of interest.
    """

    @pytest.fixture()
    def score_network_parameters(self, *args):
        raise NotImplementedError(
            "This fixture must be implemented in the derived class."
        )

    @pytest.fixture()
    def score_network(self, *args):
        raise NotImplementedError(
            "This fixture must be implemented in the derived class."
        )

    @pytest.fixture(scope="class", autouse=True)
    def set_random_seed(self):
        torch.manual_seed(23423423)

    @pytest.fixture()
    def batch_size(self):
        return 16

    @pytest.fixture()
    def number_of_atoms(self):
        return 8

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

    @pytest.fixture()
    def expected_score_shape(self, batch_size, number_of_atoms, spatial_dimension):
        return batch_size, number_of_atoms, spatial_dimension

    @pytest.fixture()
    def batch(
        self,
        relative_coordinates,
        cartesian_forces,
        times,
        noises,
        basis_vectors,
        atom_types,
    ):
        return {
            NOISY_AXL: AXL(
                A=atom_types,
                X=relative_coordinates,
                L=torch.zeros_like(atom_types),  # TODO
            ),
            TIME: times,
            UNIT_CELL: basis_vectors,
            NOISE: noises,
            CARTESIAN_FORCES: cartesian_forces,
        }

    @pytest.fixture()
    def global_parameters_dictionary(self, spatial_dimension):
        return dict(spatial_dimension=spatial_dimension, irrelevant=123)

    @pytest.fixture()
    def score_network_dictionary(
        self, score_network_parameters, global_parameters_dictionary
    ):
        dictionary = asdict(score_network_parameters)
        for key in global_parameters_dictionary.keys():
            if key in dictionary:
                dictionary.pop(key)
        return dictionary

    def test_coordinates_output_shape(self, score_network, batch, expected_score_shape):
        scores = score_network(batch)
        assert scores.X.shape == expected_score_shape

    def test_create_score_network_parameters(
        self,
        score_network_parameters,
        score_network_dictionary,
        global_parameters_dictionary,
    ):
        computed_score_network_parameters = create_score_network_parameters(
            score_network_dictionary, global_parameters_dictionary
        )
        assert_parameters_are_the_same(
            computed_score_network_parameters, score_network_parameters
        )


@pytest.mark.parametrize("spatial_dimension", [2, 3])
@pytest.mark.parametrize("num_atom_types", [2, 3, 16])
@pytest.mark.parametrize("n_hidden_dimensions", [1, 2, 3])
@pytest.mark.parametrize("hidden_dimensions_size", [8, 16])
@pytest.mark.parametrize("embedding_dimensions_size", [4, 12])
class TestMLPScoreNetwork(BaseTestScoreNetwork):

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
            atom_type_embedding_dimensions_size=embedding_dimensions_size,
            n_hidden_dimensions=n_hidden_dimensions,
            hidden_dimensions_size=hidden_dimensions_size,
        )

    @pytest.fixture()
    def score_network(self, score_network_parameters):
        return MLPScoreNetwork(score_network_parameters)


@pytest.mark.parametrize("spatial_dimension", [3])
@pytest.mark.parametrize("num_atom_types", [2, 3, 16])
@pytest.mark.parametrize("n_hidden_dimensions", [1, 2, 3])
@pytest.mark.parametrize("hidden_dimensions_size", [8, 16])
class TestMACEScoreNetworkMLPHead(BaseTestScoreNetwork):

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
@pytest.mark.parametrize("num_atom_types", [2])
class TestMACEScoreNetworkEquivariantHead(BaseTestScoreNetwork):
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


@pytest.mark.parametrize("spatial_dimension", [3])
@pytest.mark.parametrize("num_atom_types", [2, 3, 16])
class TestDiffusionMACEScoreNetwork(BaseTestScoreNetwork):
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


class TestEGNNScoreNetwork(BaseTestScoreNetwork):

    @pytest.fixture(scope="class", autouse=True)
    def set_default_type_to_float64(self):
        # Set the default type to float64 to make sure the tests are stringent.
        torch.set_default_dtype(torch.float64)
        yield
        # this returns the default type to float32 at the end of all tests in this class in order
        # to not affect other tests.
        torch.set_default_dtype(torch.float32)

    @pytest.fixture()
    def spatial_dimension(self):
        return 3

    @pytest.fixture()
    def num_atom_types(self):
        return 4

    @pytest.fixture()
    def basis_vectors(self, batch_size, spatial_dimension):
        # The basis vectors should form a cube in order to test the equivariance of the current implementation
        # of the EGNN model. The octaheral point group only applies in this case!
        acell = 5.5
        cubes = torch.stack(
            [
                torch.diag(acell * torch.ones(spatial_dimension))
                for _ in range(batch_size)
            ]
        )
        return cubes

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

    @pytest.fixture()
    def octahedral_point_group_symmetries(self):
        permutations = [
            torch.diag(torch.ones(3))[[idx]]
            for idx in itertools.permutations([0, 1, 2])
        ]
        sign_changes = [
            torch.diag(torch.tensor(diag))
            for diag in itertools.product([-1.0, 1.0], repeat=3)
        ]

        symmetries = []
        for permutation in permutations:
            for sign_change in sign_changes:
                symmetries.append(permutation @ sign_change)

        return symmetries

    @pytest.mark.parametrize(
        "edges, radial_cutoff", [("fully_connected", 3.0), ("radial_cutoff", None)]
    )
    def test_score_network_parameters(self, edges, radial_cutoff):
        score_network_parameters = EGNNScoreNetworkParameters(
            edges=edges, radial_cutoff=radial_cutoff
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
        relative_coordinates = batch[NOISY_AXL].X
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

    @pytest.fixture()
    def global_translations(self, batch_size, number_of_atoms, spatial_dimension):
        translations = einops.repeat(
            torch.rand(batch_size, spatial_dimension),
            "batch spatial_dimension -> batch natoms spatial_dimension",
            natoms=number_of_atoms,
        )
        return translations

    def test_equivariance(
        self,
        score_network,
        batch,
        octahedral_point_group_symmetries,
        global_translations,
    ):
        with torch.no_grad():
            normalized_scores = score_network(batch)

        for point_group_symmetry in octahedral_point_group_symmetries:
            op = point_group_symmetry.transpose(1, 0)
            modified_batch = deepcopy(batch)
            relative_coordinates = modified_batch[NOISY_AXL].X

            op_relative_coordinates = relative_coordinates @ op + global_translations
            op_relative_coordinates = map_relative_coordinates_to_unit_cell(
                op_relative_coordinates
            )

            modified_batch[NOISY_AXL] = AXL(
                A=modified_batch[NOISY_AXL].A,
                X=op_relative_coordinates,
                L=modified_batch[NOISY_AXL].L,
            )
            with torch.no_grad():
                modified_normalized_scores = score_network(modified_batch)

            expected_modified_normalized_scores = normalized_scores.X @ op

            torch.testing.assert_close(
                expected_modified_normalized_scores, modified_normalized_scores.X
            )
