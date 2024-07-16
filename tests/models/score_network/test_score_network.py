from dataclasses import asdict, dataclass, fields

import numpy as np
import pytest
import torch

from crystal_diffusion.models.score_networks.diffusion_mace_score_network import (
    DiffusionMACEScoreNetwork, DiffusionMACEScoreNetworkParameters)
from crystal_diffusion.models.score_networks.egnn_score_network import (
    EGNNScoreNetwork, EGNNScoreNetworkParameters)
from crystal_diffusion.models.score_networks.mace_score_network import (
    MACEScoreNetwork, MACEScoreNetworkParameters)
from crystal_diffusion.models.score_networks.mlp_score_network import (
    MLPScoreNetwork, MLPScoreNetworkParameters)
from crystal_diffusion.models.score_networks.score_network import (
    ScoreNetwork, ScoreNetworkParameters)
from crystal_diffusion.models.score_networks.score_network_factory import \
    create_score_network_parameters
from crystal_diffusion.models.score_networks.score_prediction_head import (
    MaceEquivariantScorePredictionHeadParameters,
    MaceMLPScorePredictionHeadParameters)
from crystal_diffusion.namespace import (CARTESIAN_FORCES, NOISE,
                                         NOISY_RELATIVE_COORDINATES, TIME,
                                         UNIT_CELL)


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

        bad_batch_dict = dict(good_batch)

        match problem:
            case "position_name":
                bad_batch_dict['bad_position_name'] = bad_batch_dict[NOISY_RELATIVE_COORDINATES]
                del bad_batch_dict[NOISY_RELATIVE_COORDINATES]

            case "position_shape":
                shape = bad_batch_dict[NOISY_RELATIVE_COORDINATES].shape
                bad_batch_dict[NOISY_RELATIVE_COORDINATES] = \
                    bad_batch_dict[NOISY_RELATIVE_COORDINATES].reshape(shape[0], shape[1] // 2, shape[2] * 2)

            case "position_range1":
                bad_batch_dict[NOISY_RELATIVE_COORDINATES][0, 0, 0] = 1.01

            case "position_range2":
                bad_batch_dict[NOISY_RELATIVE_COORDINATES][1, 0, 0] = -0.01

            case "time_name":
                bad_batch_dict['bad_time_name'] = bad_batch_dict[TIME]
                del bad_batch_dict[TIME]

            case "time_shape":
                shape = bad_batch_dict[TIME].shape
                bad_batch_dict[TIME] = bad_batch_dict[TIME].reshape(shape[0] // 2, shape[1] * 2)

            case "noise_name":
                bad_batch_dict['bad_noise_name'] = bad_batch_dict[NOISE]
                del bad_batch_dict[NOISE]

            case "noise_shape":
                shape = bad_batch_dict[NOISE].shape
                bad_batch_dict[NOISE] = bad_batch_dict[NOISE].reshape(shape[0] // 2, shape[1] * 2)

            case "time_range1":
                bad_batch_dict[TIME][5, 0] = 2.00
            case "time_range2":
                bad_batch_dict[TIME][0, 0] = -0.05

            case "cell_name":
                bad_batch_dict['bad_unit_cell_key'] = bad_batch_dict[UNIT_CELL]
                del bad_batch_dict[UNIT_CELL]

            case "cell_shape":
                shape = bad_batch_dict[UNIT_CELL].shape
                bad_batch_dict[UNIT_CELL] = bad_batch_dict[UNIT_CELL].reshape(shape[0] // 2, shape[1] * 2, shape[2])

        return bad_batch_dict

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
    def score_network_parameters(self, *args):
        raise NotImplementedError("This fixture must be implemented in the derived class.")

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

    @pytest.fixture()
    def global_parameters_dictionary(self, spatial_dimension):
        return dict(spatial_dimension=spatial_dimension, irrelevant=123)

    @pytest.fixture()
    def score_network_dictionary(self, score_network_parameters, global_parameters_dictionary):
        dictionary = asdict(score_network_parameters)
        for key in global_parameters_dictionary.keys():
            if key in dictionary:
                dictionary.pop(key)
        return dictionary

    def test_output_shape(self, score_network, batch, expected_score_shape):
        scores = score_network(batch)
        assert scores.shape == expected_score_shape

    def test_create_score_network_parameters(self, score_network_parameters,
                                             score_network_dictionary,
                                             global_parameters_dictionary):
        computed_score_network_parameters = create_score_network_parameters(score_network_dictionary,
                                                                            global_parameters_dictionary)
        assert_parameters_are_the_same(computed_score_network_parameters, score_network_parameters)


@pytest.mark.parametrize("spatial_dimension", [2, 3])
@pytest.mark.parametrize("n_hidden_dimensions", [1, 2, 3])
@pytest.mark.parametrize("hidden_dimensions_size", [8, 16])
class TestMLPScoreNetwork(BaseTestScoreNetwork):

    @pytest.fixture()
    def score_network_parameters(self, number_of_atoms, spatial_dimension, n_hidden_dimensions, hidden_dimensions_size):
        return MLPScoreNetworkParameters(spatial_dimension=spatial_dimension,
                                         number_of_atoms=number_of_atoms,
                                         n_hidden_dimensions=n_hidden_dimensions,
                                         hidden_dimensions_size=hidden_dimensions_size)

    @pytest.fixture()
    def score_network(self, score_network_parameters):
        return MLPScoreNetwork(score_network_parameters)


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
    def score_network_parameters(self, number_of_atoms, spatial_dimension, prediction_head_parameters):
        return MACEScoreNetworkParameters(spatial_dimension=spatial_dimension,
                                          number_of_atoms=number_of_atoms,
                                          r_max=3.0,
                                          prediction_head_parameters=prediction_head_parameters)

    @pytest.fixture()
    def score_network(self, score_network_parameters):
        return MACEScoreNetwork(score_network_parameters)


@pytest.mark.parametrize("spatial_dimension", [3])
class TestMACEScoreNetworkEquivariantHead(BaseTestScoreNetwork):
    @pytest.fixture()
    def prediction_head_parameters(self, spatial_dimension):
        prediction_head_parameters = MaceEquivariantScorePredictionHeadParameters(spatial_dimension=spatial_dimension,
                                                                                  number_of_layers=2)
        return prediction_head_parameters

    @pytest.fixture()
    def score_network_parameters(self, number_of_atoms, spatial_dimension, prediction_head_parameters):
        return MACEScoreNetworkParameters(spatial_dimension=spatial_dimension,
                                          number_of_atoms=number_of_atoms,
                                          r_max=3.0,
                                          prediction_head_parameters=prediction_head_parameters)

    @pytest.fixture()
    def score_network(self, score_network_parameters):
        return MACEScoreNetwork(score_network_parameters)


@pytest.mark.parametrize("spatial_dimension", [3])
class TestDiffusionMACEScoreNetwork(BaseTestScoreNetwork):
    @pytest.fixture()
    def score_network_parameters(self, number_of_atoms, spatial_dimension):
        return DiffusionMACEScoreNetworkParameters(spatial_dimension=spatial_dimension,
                                                   number_of_atoms=number_of_atoms,
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


@pytest.mark.parametrize("spatial_dimension", [3])
class TestEGNNScoreNetwork(BaseTestScoreNetwork):
    @pytest.fixture()
    def score_network_parameters(self):
        return EGNNScoreNetworkParameters(hidden_dim=32)

    @pytest.fixture()
    def score_network(self, score_network_parameters):
        return EGNNScoreNetwork(score_network_parameters)
