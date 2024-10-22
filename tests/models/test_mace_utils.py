import unittest.mock
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from crystal_diffusion.models.mace_utils import (
    get_normalized_irreps_permutation_indices, get_pretrained_mace,
    input_to_mace, reshape_from_e3nn_to_mace, reshape_from_mace_to_e3nn)
from crystal_diffusion.namespace import NOISY_CARTESIAN_POSITIONS, UNIT_CELL
from crystal_diffusion.utils.basis_transformations import \
    get_positions_from_coordinates
from e3nn import o3
from mace.data import AtomicData, Configuration
from mace.tools import get_atomic_number_table_from_zs
from mace.tools.torch_geometric.dataloader import Collater

from tests.fake_data_utils import find_aligning_permutation


class TestInputToMaceChain:

    @pytest.fixture()
    def spatial_dim(self):
        return 3

    @pytest.fixture()
    def batch_size(self):
        return 8

    @pytest.fixture()
    def n_atoms(self):
        return 4

    @pytest.fixture()
    def atomic_positions(self, n_atoms, spatial_dim):
        # chain of atoms at x=0,1,2,3, y=z=0
        pos = torch.zeros(n_atoms, spatial_dim)
        pos[:, 0] += torch.arange(n_atoms)
        return pos

    @pytest.fixture()
    def cell_size(self, n_atoms):
        # get a cell much larger than spacing between atoms to not deal with periodicity
        return 10 * n_atoms

    @pytest.fixture()
    def radial_cutoff(self):
        # set the cutoff at 1.1 so the atoms form a simple chain 0 - 1 - 2 - 3
        return 1.1

    @pytest.fixture()
    def mace_graph(self, cell_size, spatial_dim, atomic_positions, n_atoms, batch_size, radial_cutoff):
        unit_cell = np.eye(spatial_dim) * cell_size  # box as a spatial_dim x spatial_dim array
        atom_type = np.ones(n_atoms) * 14
        pbc = np.array([True] * spatial_dim)  # periodic boundary conditions
        graph_config = Configuration(atomic_numbers=atom_type,
                                     positions=atomic_positions.numpy(),
                                     cell=unit_cell,
                                     pbc=pbc)
        z_table = get_atomic_number_table_from_zs(list(range(89)))
        graph_data = AtomicData.from_config(graph_config, z_table=z_table, cutoff=radial_cutoff)
        collate_fn = Collater(follow_batch=[None], exclude_keys=[None])
        mace_batch = collate_fn([graph_data] * batch_size)
        return mace_batch

    @pytest.fixture()
    def score_network_input(self, batch_size, spatial_dim, cell_size, atomic_positions):
        score_network_input = {NOISY_CARTESIAN_POSITIONS: atomic_positions.unsqueeze(0).repeat(batch_size, 1, 1),
                               UNIT_CELL: torch.eye(spatial_dim).repeat(batch_size, 1, 1) * cell_size}

        return score_network_input

    def test_input_to_mace(self, score_network_input, radial_cutoff, mace_graph):
        crystal_diffusion_graph = input_to_mace(score_network_input,
                                                radial_cutoff=radial_cutoff)

        # check the features used in MACE as the same in both our homemade graph and the one native to mace library
        for k in ['edge_index', 'node_attrs', 'positions', 'ptr', 'batch', 'cell']:
            assert k in crystal_diffusion_graph.keys()
            assert crystal_diffusion_graph[k].size() == mace_graph[k].size()
            assert torch.equal(crystal_diffusion_graph[k], mace_graph[k])


class TestInputToMaceRandom(TestInputToMaceChain):
    @pytest.fixture(scope="class", autouse=True)
    def set_seed(self):
        """Set the random seed."""
        torch.manual_seed(2345234234)

    @pytest.fixture()
    def n_atoms(self):
        return 16

    @pytest.fixture
    def basis_vectors(self, batch_size):
        # orthogonal boxes with dimensions between 5 and 10.
        orthogonal_boxes = torch.stack([torch.diag(5. + 5. * torch.rand(3)) for _ in range(batch_size)])
        # add a bit of noise to make the vectors not quite orthogonal
        basis_vectors = orthogonal_boxes + 0.1 * torch.randn(batch_size, 3, 3)
        return basis_vectors

    @pytest.fixture
    def cartesian_positions(self, batch_size, n_atoms, spatial_dim, basis_vectors):
        relative_coordinates = torch.rand(batch_size, n_atoms, spatial_dim)
        positions = get_positions_from_coordinates(relative_coordinates, basis_vectors)
        return positions

    @pytest.fixture()
    def mace_graph(self, basis_vectors, cartesian_positions, spatial_dim, n_atoms, radial_cutoff):
        pbc = np.array([True] * spatial_dim)  # periodic boundary conditions
        atom_type = np.ones(n_atoms) * 14

        list_graphs = []
        for unit_cell, atomic_positions in zip(basis_vectors, cartesian_positions):
            graph_config = Configuration(atomic_numbers=atom_type,
                                         positions=atomic_positions.numpy(),
                                         cell=unit_cell,
                                         pbc=pbc)
            z_table = get_atomic_number_table_from_zs(list(range(89)))
            graph_data = AtomicData.from_config(graph_config, z_table=z_table, cutoff=radial_cutoff)
            list_graphs.append(graph_data)

        collate_fn = Collater(follow_batch=[None], exclude_keys=[None])
        mace_batch = collate_fn(list_graphs)
        return mace_batch

    @pytest.fixture()
    def score_network_input(self, cartesian_positions, basis_vectors):
        score_network_input = {NOISY_CARTESIAN_POSITIONS: cartesian_positions, UNIT_CELL: basis_vectors}
        return score_network_input

    @pytest.mark.parametrize("radial_cutoff", [1.1, 2.2, 4.4])
    def test_input_to_mace(self, score_network_input, radial_cutoff, mace_graph):
        computed_mace_graph = input_to_mace(score_network_input,
                                            radial_cutoff=radial_cutoff)

        for feature_name in ['node_attrs', 'positions', 'ptr', 'batch', 'cell']:
            torch.testing.assert_close(mace_graph[feature_name], computed_mace_graph[feature_name])

        # The EDGES might not be in the same order. Compute permutations of edges

        expected_src_idx = mace_graph['edge_index'][0, :]
        expected_dst_idx = mace_graph['edge_index'][1, :]

        expected_displacements = (mace_graph['positions'][expected_dst_idx] + mace_graph['shifts']
                                  - mace_graph['positions'][expected_src_idx])

        computed_src_idx = computed_mace_graph['edge_index'][0, :]
        computed_dst_idx = computed_mace_graph['edge_index'][1, :]
        computed_displacements = (computed_mace_graph['positions'][computed_dst_idx] + computed_mace_graph['shifts']
                                  - computed_mace_graph['positions'][computed_src_idx])

        # The edge order can be different between the two graphs
        edge_permutation_indices = find_aligning_permutation(expected_displacements, computed_displacements)

        torch.testing.assert_close(computed_mace_graph['shifts'][edge_permutation_indices], mace_graph['shifts'])
        torch.testing.assert_close(computed_mace_graph['edge_index'][:, edge_permutation_indices],
                                   mace_graph['edge_index'])


@pytest.fixture()
def scrampled_and_expected_irreps(seed):

    torch.manual_seed(seed)

    list_irrep_strings = []
    for ell in range(4):
        for parity in ['o', 'e']:
            list_irrep_strings.append(f'{ell}{parity}')

    n_irreps = len(list_irrep_strings)
    list_repetitions = torch.randint(0, 4, (n_irreps,))

    list_irreps = []
    expected_irreps = o3.Irreps('')
    for irrep_string, repetitions in zip(list_irrep_strings, list_repetitions):
        if repetitions == 0:
            continue
        multiplicities = torch.randint(1, 8, (repetitions,))
        list_irreps.extend([o3.Irreps(f'{multiplicity}x{irrep_string}') for multiplicity in multiplicities])

        total_multiplicity = int(multiplicities.sum())

        expected_irreps += o3.Irreps(f'{total_multiplicity}x{irrep_string}')

    scrambled_indices = torch.randperm(len(list_irreps))
    scrambled_irreps = o3.Irreps('')
    for index in scrambled_indices:
        scrambled_irreps += list_irreps[index]

    return scrambled_irreps, expected_irreps


@pytest.mark.parametrize("seed", [123, 999, 1234234234])
def test_get_normalized_irreps_permutation_indices(scrampled_and_expected_irreps):
    scrambled_irreps, expected_irreps = scrampled_and_expected_irreps

    normalized_irreps, column_permutation_indices = get_normalized_irreps_permutation_indices(scrambled_irreps)

    assert expected_irreps == normalized_irreps

    # It is very difficult to determine the column_permutation_indices because there is more than
    # one valid possibility. Let's check instead that data transforms as it should.

    batch_size = 16
    scrambled_data = scrambled_irreps.randn(batch_size, -1)

    rot = o3.rand_matrix()

    scrambled_d_matrix = scrambled_irreps.D_from_matrix(rot)
    scrambled_rotated_data = scrambled_data @ scrambled_d_matrix.T
    rotate_then_normalize_data = scrambled_rotated_data[:, column_permutation_indices]

    normalized_d_matrix = normalized_irreps.D_from_matrix(rot)
    normalized_data = scrambled_data[:, column_permutation_indices]
    normalize_then_rotate_data = normalized_data @ normalized_d_matrix.T

    torch.testing.assert_close(rotate_then_normalize_data, normalize_then_rotate_data)


class TestPretrainedMace:
    @pytest.fixture
    def mock_model_savedir(self, tmp_path):
        return str(tmp_path)

    # Test correctly downloading a small model
    @patch("os.makedirs")
    @patch("os.path.isfile", return_value=False)
    @patch("urllib.request.urlretrieve", return_value=(None, 'abc'))
    @patch("torch.load")
    def test_download_pretrained_mace_small(self, mock_load, mock_urlretrieve, mock_isfile, mock_makedirs,
                                            mock_model_savedir):
        mock_model = MagicMock(spec=torch.nn.Module)
        mock_model.float.return_value = mock_model
        mock_load.return_value = mock_model
        model, node_feats_output_size = get_pretrained_mace("small", mock_model_savedir)

        mock_urlretrieve.assert_called()
        mock_load.assert_called_with(f=unittest.mock.ANY)
        assert model is mock_model
        assert node_feats_output_size == 256  # assuming 256 is the small model's size

    # Test handling invalid model name
    def test_download_pretrained_mace_invalid_model_name(self, mock_model_savedir):
        with pytest.raises(AssertionError) as e:
            get_pretrained_mace("invalid_name", mock_model_savedir)
        assert "Model name should be small, medium or large. Got invalid_name" in str(e.value)


class TestReshapes:
    @pytest.fixture
    def num_nodes(self):
        return 5

    @pytest.fixture
    def num_channels(self):
        return 3

    @pytest.fixture
    def ell_max(self):
        return 2

    @pytest.fixture
    def irrep(self, num_channels, ell_max):
        irrep_str = f"{num_channels}x0e"
        for i in range(1, ell_max + 1):
            parity = "e" if i % 2 == 0 else "o"
            irrep_str += f"+ {num_channels}x{i}{parity}"
        return o3.Irreps(irrep_str)

    @pytest.fixture
    def mace_format_tensor(self, num_nodes, num_channels, ell_max):
        return torch.rand(num_nodes, num_channels, (ell_max + 1) ** 2)

    @pytest.fixture
    def e3nn_format_tensor(self, num_nodes, num_channels, ell_max):
        return torch.rand(num_nodes, num_channels * (ell_max + 1) ** 2)

    def test_reshape_from_mace_to_e3nn(self, mace_format_tensor, irrep, ell_max, num_channels):
        converted_tensor = reshape_from_mace_to_e3nn(mace_format_tensor, irrep)
        # mace_format_tensor: (node, channel, (lmax + 1) ** 2)
        # converted: (node, channel * (lmax +1)**2)
        # check for each ell that the values match
        for ell in range(ell_max + 1):
            start_idx = ell ** 2
            end_idx = (ell + 1) ** 2
            expected_values = mace_format_tensor[:, :, start_idx:end_idx].reshape(-1, num_channels * (2 * ell + 1))
            assert torch.allclose(expected_values,
                                  converted_tensor[:, num_channels * start_idx: num_channels * end_idx])

    def test_reshape_from_e3nn_to_mace(self, e3nn_format_tensor, irrep, ell_max, num_channels):
        converted_tensor = reshape_from_e3nn_to_mace(e3nn_format_tensor, irrep)
        # e3nn_format_tensor: (node, channel * (lmax +1)**2)
        # converted: (node, channel, (lmax + 1) ** 2)
        for ell in range(ell_max + 1):
            start_idx = num_channels * (ell ** 2)
            end_idx = num_channels * ((ell + 1) ** 2)
            expected_values = e3nn_format_tensor[:, start_idx:end_idx].reshape(-1, num_channels, 2 * ell + 1)
            assert torch.allclose(expected_values,
                                  converted_tensor[:, :, ell ** 2:(ell + 1) ** 2])
