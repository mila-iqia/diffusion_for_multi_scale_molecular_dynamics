import pytest
import torch

from crystal_diffusion.models.score_networks.diffusion_mace_score_network import (
    DiffusionMACEScoreNetwork, DiffusionMACEScoreNetworkParameters)
from crystal_diffusion.models.score_networks.mace_score_network import (
    MACEScoreNetwork, MACEScoreNetworkParameters)
from crystal_diffusion.models.score_networks.mlp_score_network import (
    MLPScoreNetwork, MLPScoreNetworkParameters)
from crystal_diffusion.models.score_networks.score_network import (
    ScoreNetwork, ScoreNetworkParameters)
from crystal_diffusion.models.score_networks.score_prediction_head import (
    MaceEquivariantScorePredictionHeadParameters,
    MaceMLPScorePredictionHeadParameters)
from crystal_diffusion.namespace import (CARTESIAN_FORCES, NOISE,
                                         NOISY_RELATIVE_COORDINATES, TIME,
                                         UNIT_CELL)


@pytest.mark.parametrize("spatial_dimension", [2, 3])
class TestScoreNetworkCheck:

    @pytest.fixture(scope="class", autouse=True)
    def set_random_seed(self):
        torch.manual_seed(123)

    @pytest.fixture()
    def base_score_network(self, spatial_dimension):
        return ScoreNetwork(ScoreNetworkParameters(architecture='dummy',
                                                   spatial_dimension=spatial_dimension))

    @pytest.fixture()
    def good_batch(self, spatial_dimension):
        batch_size = 16
        relative_coordinates = torch.rand(batch_size, 8, spatial_dimension)
        times = torch.rand(batch_size, 1)
        noises = torch.rand(batch_size, 1)
        unit_cell = torch.rand(batch_size, spatial_dimension, spatial_dimension)
        return {NOISY_RELATIVE_COORDINATES: relative_coordinates, TIME: times, NOISE: noises, UNIT_CELL: unit_cell}

    @pytest.fixture()
    def bad_batch(self, good_batch, problem):

        bad_batch = dict(good_batch)

        match problem:
            case "position_name":
                bad_batch['bad_position_name'] = bad_batch[NOISY_RELATIVE_COORDINATES]
                del bad_batch[NOISY_RELATIVE_COORDINATES]

            case "position_shape":
                shape = bad_batch[NOISY_RELATIVE_COORDINATES].shape
                bad_batch[NOISY_RELATIVE_COORDINATES] = \
                    bad_batch[NOISY_RELATIVE_COORDINATES].reshape(shape[0], shape[1] // 2, shape[2] * 2)

            case "position_range1":
                bad_batch[NOISY_RELATIVE_COORDINATES][0, 0, 0] = 1.01

            case "position_range2":
                bad_batch[NOISY_RELATIVE_COORDINATES][1, 0, 0] = -0.01

            case "time_name":
                bad_batch['bad_time_name'] = bad_batch[TIME]
                del bad_batch[TIME]

            case "time_shape":
                shape = bad_batch[TIME].shape
                bad_batch[TIME] = bad_batch[TIME].reshape(shape[0] // 2, shape[1] * 2)

            case "noise_name":
                bad_batch['bad_noise_name'] = bad_batch[NOISE]
                del bad_batch[NOISE]

            case "noise_shape":
                shape = bad_batch[NOISE].shape
                bad_batch[NOISE] = bad_batch[NOISE].reshape(shape[0] // 2, shape[1] * 2)

            case "time_range1":
                bad_batch[TIME][5, 0] = 2.00
            case "time_range2":
                bad_batch[TIME][0, 0] = -0.05

            case "cell_name":
                bad_batch['bad_unit_cell_key'] = bad_batch[UNIT_CELL]
                del bad_batch[UNIT_CELL]

            case "cell_shape":
                shape = bad_batch[UNIT_CELL].shape
                bad_batch[UNIT_CELL] = bad_batch[UNIT_CELL].reshape(shape[0] // 2, shape[1] * 2, shape[2])

        return bad_batch

    def test_check_batch_good(self, base_score_network, good_batch):
        base_score_network._check_batch(good_batch)

    @pytest.mark.parametrize('problem', ['position_name', 'time_name', 'position_shape',
                                         'time_shape', 'noise_name', 'noise_shape', 'position_range1',
                                         'position_range2', 'time_range1', 'time_range2', 'cell_name', 'cell_shape'])
    def test_check_batch_bad(self, base_score_network, bad_batch):
        with pytest.raises(AssertionError):
            base_score_network._check_batch(bad_batch)


class BaseTestScoreNetwork:
    """Base Test Score Network.

    Base class to run a battery of tests on a score network. To test a specific score network class, this base class
    should be extended by implementing a 'score_network' fixture that instantiates the score network class of interest.
    """
    @pytest.fixture()
    def score_network(self, *args):
        raise NotImplementedError("This fixture must be implemented in the derived class.")

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
        orthogonal_boxes = torch.stack([torch.diag(5. + 5. * torch.rand(spatial_dimension)) for _ in range(batch_size)])
        # add a bit of noise to make the vectors not quite orthogonal
        basis_vectors = orthogonal_boxes + 0.1 * torch.randn(batch_size, spatial_dimension, spatial_dimension)
        return basis_vectors

    @pytest.fixture
    def relative_coordinates(self, batch_size, number_of_atoms, spatial_dimension, basis_vectors):
        relative_coordinates = torch.rand(batch_size, number_of_atoms, spatial_dimension)
        return relative_coordinates

    @pytest.fixture
    def cartesian_forces(self, batch_size, number_of_atoms, spatial_dimension, basis_vectors):
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
    def batch(self, relative_coordinates, cartesian_forces, times, noises, basis_vectors):
        return {NOISY_RELATIVE_COORDINATES: relative_coordinates, TIME: times, UNIT_CELL: basis_vectors, NOISE: noises,
                CARTESIAN_FORCES: cartesian_forces}

    def test_output_shape(self, score_network, batch, expected_score_shape):
        scores = score_network(batch)
        assert scores.shape == expected_score_shape


@pytest.mark.parametrize("spatial_dimension", [2, 3])
@pytest.mark.parametrize("n_hidden_dimensions", [1, 2, 3])
@pytest.mark.parametrize("hidden_dimensions_size", [8, 16])
class TestMLPScoreNetwork(BaseTestScoreNetwork):

    @pytest.fixture()
    def score_network(self, number_of_atoms, spatial_dimension, n_hidden_dimensions, hidden_dimensions_size):
        hyper_params = MLPScoreNetworkParameters(spatial_dimension=spatial_dimension,
                                                 number_of_atoms=number_of_atoms,
                                                 n_hidden_dimensions=n_hidden_dimensions,
                                                 hidden_dimensions_size=hidden_dimensions_size)
        return MLPScoreNetwork(hyper_params)


@pytest.mark.parametrize("spatial_dimension", [3])
@pytest.mark.parametrize("n_hidden_dimensions", [1, 2, 3])
@pytest.mark.parametrize("hidden_dimensions_size", [8, 16])
class TestMACEScoreNetworkMLPHead(BaseTestScoreNetwork):

    @pytest.fixture()
    def prediction_head_parameters(self, spatial_dimension, n_hidden_dimensions, hidden_dimensions_size):
        prediction_head_parameters = MaceMLPScorePredictionHeadParameters(spatial_dimension=spatial_dimension,
                                                                          hidden_dimensions_size=hidden_dimensions_size,
                                                                          n_hidden_dimensions=n_hidden_dimensions)
        return prediction_head_parameters

    @pytest.fixture()
    def score_network(self, number_of_atoms, spatial_dimension, prediction_head_parameters):
        hyper_params = MACEScoreNetworkParameters(spatial_dimension=spatial_dimension,
                                                  number_of_atoms=number_of_atoms,
                                                  r_max=3.0,
                                                  prediction_head_parameters=prediction_head_parameters)
        return MACEScoreNetwork(hyper_params)


@pytest.mark.parametrize("spatial_dimension", [3])
class TestMACEScoreNetworkEquivariantHead(BaseTestScoreNetwork):
    @pytest.fixture()
    def prediction_head_parameters(self, spatial_dimension):
        prediction_head_parameters = MaceEquivariantScorePredictionHeadParameters(spatial_dimension=spatial_dimension,
                                                                                  number_of_layers=2)
        return prediction_head_parameters

    @pytest.fixture()
    def score_network(self, number_of_atoms, spatial_dimension, prediction_head_parameters):
        hyper_params = MACEScoreNetworkParameters(spatial_dimension=spatial_dimension,
                                                  number_of_atoms=number_of_atoms,
                                                  r_max=3.0,
                                                  prediction_head_parameters=prediction_head_parameters)
        return MACEScoreNetwork(hyper_params)


@pytest.mark.parametrize("spatial_dimension", [3])
class TestDiffusionMACEScoreNetwork(BaseTestScoreNetwork):
    @pytest.fixture()
    def score_network(self, number_of_atoms, spatial_dimension):
        hyper_params = DiffusionMACEScoreNetworkParameters(spatial_dimension=spatial_dimension,
                                                           number_of_atoms=number_of_atoms,
                                                           r_max=3.0,
                                                           num_bessel=4,
                                                           num_polynomial_cutoff=3,
                                                           hidden_irreps="8x0e + 8x1o",
                                                           MLP_irreps="8x0e",
                                                           correlation=2,
                                                           radial_MLP=[8, 8, 8],
                                                           )

        return DiffusionMACEScoreNetwork(hyper_params)
