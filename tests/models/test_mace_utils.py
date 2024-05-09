import numpy as np
import pytest
import torch
from mace.data import AtomicData, Configuration
from mace.tools import get_atomic_number_table_from_zs
from mace.tools.torch_geometric.dataloader import Collater

from crystal_diffusion.models.mace_utils import input_to_mace
from crystal_diffusion.utils.neighbors import _get_positions_from_coordinates
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
        score_network_input = dict(abs_positions=atomic_positions.unsqueeze(0).repeat(batch_size, 1, 1),
                                   cell=torch.eye(spatial_dim).repeat(batch_size, 1, 1) * cell_size)
        return score_network_input

    def test_input_to_mace(self, score_network_input, radial_cutoff, mace_graph):
        crystal_diffusion_graph = input_to_mace(score_network_input,
                                                unit_cell_key='cell',
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
    def positions(self, batch_size, n_atoms, spatial_dim, basis_vectors):
        relative_coordinates = torch.rand(batch_size, n_atoms, spatial_dim)
        positions = _get_positions_from_coordinates(relative_coordinates, basis_vectors)
        return positions

    @pytest.fixture()
    def mace_graph(self, basis_vectors, positions, spatial_dim, n_atoms, radial_cutoff):
        pbc = np.array([True] * spatial_dim)  # periodic boundary conditions
        atom_type = np.ones(n_atoms) * 14

        list_graphs = []
        for unit_cell, atomic_positions in zip(basis_vectors, positions):
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
    def score_network_input(self, positions, basis_vectors):
        score_network_input = dict(abs_positions=positions, cell=basis_vectors)
        return score_network_input

    @pytest.mark.parametrize("radial_cutoff", [1.1, 2.2, 4.4])
    def test_input_to_mace(self, score_network_input, radial_cutoff, mace_graph):
        computed_mace_graph = input_to_mace(score_network_input,
                                            unit_cell_key='cell',
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
