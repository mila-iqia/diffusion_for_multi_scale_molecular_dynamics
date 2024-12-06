import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.models.score_networks import (
    ScoreNetwork, ScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, NOISE, NOISY_AXL_COMPOSITION, TIME, UNIT_CELL)
from tests.models.score_network.base_test_score_network import \
    BaseTestScoreNetwork


@pytest.mark.parametrize("spatial_dimension", [2, 3])
class TestScoreNetworkBasicCheck(BaseTestScoreNetwork):

    @pytest.fixture()
    def score_network(self, spatial_dimension, num_atom_types):
        score_parameters = ScoreNetworkParameters(
            architecture="dummy",
            spatial_dimension=spatial_dimension,
            num_atom_types=num_atom_types,
        )

        return ScoreNetwork(score_parameters)

    @pytest.fixture()
    def good_batch(self, spatial_dimension, num_atom_types, number_of_atoms):
        batch_size = 16
        relative_coordinates = torch.rand(
            batch_size, number_of_atoms, spatial_dimension
        )
        times = torch.rand(batch_size, 1)
        noises = torch.rand(batch_size, 1)
        unit_cell = torch.rand(batch_size, spatial_dimension, spatial_dimension)
        lattice_params = torch.zeros(batch_size, int(spatial_dimension * (spatial_dimension + 1) / 2))
        lattice_params[:, :spatial_dimension] = torch.diagonal(unit_cell, dim1=-2, dim2=-1)
        atom_types = torch.randint(0, num_atom_types + 1, (batch_size, number_of_atoms))
        return {
            NOISY_AXL_COMPOSITION: AXL(
                A=atom_types, X=relative_coordinates, L=lattice_params
            ),
            TIME: times,
            NOISE: noises,
            UNIT_CELL: unit_cell,
        }

    @pytest.fixture()
    def bad_batch(self, good_batch, problem, num_atom_types):

        bad_batch_dict = dict(good_batch)

        match problem:
            case "position_name":
                bad_batch_dict["bad_position_name"] = bad_batch_dict[
                    NOISY_AXL_COMPOSITION
                ]
                del bad_batch_dict[NOISY_AXL_COMPOSITION]

            case "position_shape":
                shape = bad_batch_dict[NOISY_AXL_COMPOSITION].X.shape
                bad_batch_dict[NOISY_AXL_COMPOSITION] = AXL(
                    A=bad_batch_dict[NOISY_AXL_COMPOSITION].A,
                    X=bad_batch_dict[NOISY_AXL_COMPOSITION].X.reshape(
                        shape[0], shape[1] // 2, shape[2] * 2
                    ),
                    L=bad_batch_dict[NOISY_AXL_COMPOSITION].L,
                )

            case "position_range1":
                bad_positions = bad_batch_dict[NOISY_AXL_COMPOSITION].X
                bad_positions[0, 0, 0] = 1.01
                bad_batch_dict[NOISY_AXL_COMPOSITION] = AXL(
                    A=bad_batch_dict[NOISY_AXL_COMPOSITION].A,
                    X=bad_positions,
                    L=bad_batch_dict[NOISY_AXL_COMPOSITION].L,
                )

            case "position_range2":
                bad_positions = bad_batch_dict[NOISY_AXL_COMPOSITION].X
                bad_positions[1, 0, 0] = -0.01
                bad_batch_dict[NOISY_AXL_COMPOSITION] = AXL(
                    A=bad_batch_dict[NOISY_AXL_COMPOSITION].A,
                    X=bad_positions,
                    L=bad_batch_dict[NOISY_AXL_COMPOSITION].L,
                )

            case "atom_types_shape":
                shape = bad_batch_dict[NOISY_AXL_COMPOSITION].A.shape
                bad_batch_dict[NOISY_AXL_COMPOSITION] = AXL(
                    A=bad_batch_dict[NOISY_AXL_COMPOSITION].A.reshape(
                        shape[0] * 2, shape[1] // 2
                    ),
                    X=bad_batch_dict[NOISY_AXL_COMPOSITION].X,
                    L=bad_batch_dict[NOISY_AXL_COMPOSITION].L,
                )

            case "atom_types_range1":
                bad_types = bad_batch_dict[NOISY_AXL_COMPOSITION].A
                bad_types[0, 0] = num_atom_types + 2
                bad_batch_dict[NOISY_AXL_COMPOSITION] = AXL(
                    A=bad_types,
                    X=bad_batch_dict[NOISY_AXL_COMPOSITION].X,
                    L=bad_batch_dict[NOISY_AXL_COMPOSITION].L,
                )

            case "atom_types_range2":
                bad_types = bad_batch_dict[NOISY_AXL_COMPOSITION].A
                bad_types[1, 0] = -1
                bad_batch_dict[NOISY_AXL_COMPOSITION] = AXL(
                    A=bad_types,
                    X=bad_batch_dict[NOISY_AXL_COMPOSITION].X,
                    L=bad_batch_dict[NOISY_AXL_COMPOSITION].L,
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

            case "lattice_shape":
                shape = bad_batch_dict[NOISY_AXL_COMPOSITION].L.shape
                bad_shaped_lattice = bad_batch_dict[NOISY_AXL_COMPOSITION].L.reshape(
                    shape[0] // 2, shape[1] * 2
                )
                bad_batch_dict[NOISY_AXL_COMPOSITION] = AXL(
                    A=bad_batch_dict[NOISY_AXL_COMPOSITION].A,
                    X=bad_batch_dict[NOISY_AXL_COMPOSITION].X,
                    L=bad_shaped_lattice,
                )

        return bad_batch_dict

    def test_check_batch_good(self, score_network, good_batch):
        score_network._check_batch(good_batch)

    @pytest.mark.parametrize(
        "problem",
        [
            "position_name",
            "time_name",
            "position_shape",
            "atom_types_shape",
            "time_shape",
            "noise_name",
            "noise_shape",
            "position_range1",
            "position_range2",
            "atom_types_range1",
            "atom_types_range2",
            "time_range1",
            "time_range2",
            "lattice_shape",
        ],
    )
    def test_check_batch_bad(self, score_network, bad_batch):
        with pytest.raises(AssertionError):
            score_network._check_batch(bad_batch)
