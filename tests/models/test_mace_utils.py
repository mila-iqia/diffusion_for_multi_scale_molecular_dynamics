import numpy as np
import pytest
import torch
from mace.data import AtomicData, Configuration
from mace.tools import get_atomic_number_table_from_zs
from mace.tools.torch_geometric.dataloader import Collater

from crystal_diffusion.models.mace_utils import input_to_mace


@pytest.fixture()
def n_atoms():
    return 4


@pytest.fixture()
def spatial_dim():
    return 3


@pytest.fixture()
def atomic_positions(n_atoms, spatial_dim):
    # chain of atoms at x=0,1,2,3, y=z=0
    pos = torch.zeros(n_atoms, spatial_dim)
    pos[:, 0] += torch.arange(n_atoms)
    return pos


@pytest.fixture()
def cell_size(n_atoms):
    # get a cell much larger than spacing between atoms to not deal with periodicity
    return 10 * n_atoms


@pytest.fixture()
def batchsize():
    return 8


@pytest.fixture()
def mace_graph(cell_size, spatial_dim, atomic_positions, n_atoms, batchsize):
    unit_cell = np.eye(spatial_dim) * cell_size  # box as a spatial_dim x spatial_dim array
    atom_type = np.ones(n_atoms)
    pbc = np.array([True] * spatial_dim)  # periodic boundary conditions
    graph_config = Configuration(atomic_numbers=atom_type,
                                 positions=atomic_positions.numpy(),
                                 cell=unit_cell,
                                 pbc=pbc)
    z_table = get_atomic_number_table_from_zs([1])
    # set the cutoff at 1.1 so the atoms form a simple chain 0 - 1 - 2 - 3
    graph_data = AtomicData.from_config(graph_config, z_table=z_table, cutoff=1.1)
    collate_fn = Collater(follow_batch=[None], exclude_keys=[None])
    mace_batch = collate_fn([graph_data] * batchsize)
    return mace_batch


@pytest.fixture()
def mocked_adjacency_matrix(n_atoms, batchsize):
    # adjacency should be a (2, n_edge)
    # we have a simple chain 0-1-2-3... in 1D, we can ignore complexities from periodicity and 3D
    adj = torch.zeros(n_atoms, n_atoms)
    for i in range(n_atoms):
        if i < n_atoms - 1:
            adj[i, i + 1] = 1  # 0 to 1, ... n-1 to n ; but not n to n+1
        if i > 0:
            adj[i, i - 1] = 1  # 1 to 0, 2 to 1... but not 0 to -1
    adj = adj.unsqueeze(0).repeat(batchsize, 1, 1)
    adj = adj.to_sparse().indices()
    # adj contains (batch index, node index 1, node index 2)
    # the batch index will create problems, so we need to remove it by reindexing the node
    # for example, node 0 in the second graph should be considered as node index 0 + num_node
    adj = adj[1:, :] + adj[0, :] * n_atoms  # now a 2 * num_edge tensor
    return adj


@pytest.fixture()
def mocked_shift_matrix(n_atoms, batchsize, spatial_dim):
    # chain structure: there are n_atoms - 1 links - which means 2 * (n_atoms - 1) edges from bidirectionality
    # times batchsize to account for batching
    return torch.zeros(2 * (n_atoms - 1) * batchsize, spatial_dim)


def test_input_to_mace(mocker, batchsize, cell_size, spatial_dim, atomic_positions, mace_graph,
                       mocked_adjacency_matrix, mocked_shift_matrix):
    score_network_input = {}
    score_network_input['abs_positions'] = atomic_positions.unsqueeze(0).repeat(batchsize, 1, 1)
    mocker.patch("crystal_diffusion.models.mace_utils.get_adj_matrix",
                 return_value=(mocked_adjacency_matrix, mocked_shift_matrix))
    repeat_size = [batchsize] + [1] * spatial_dim
    score_network_input['cell'] = torch.eye(spatial_dim).repeat(*repeat_size) * cell_size
    crystal_diffusion_graph = input_to_mace(score_network_input, unit_cell_key='cell')

    # check the features used in MACE as the same in both our homemade graph and the one native to mace library
    for k in ['edge_index', 'node_attrs', 'positions', 'ptr', 'batch', 'cell']:
        assert k in crystal_diffusion_graph.keys()
        assert crystal_diffusion_graph[k].size() == mace_graph[k].size()
        assert torch.equal(crystal_diffusion_graph[k], mace_graph[k])
