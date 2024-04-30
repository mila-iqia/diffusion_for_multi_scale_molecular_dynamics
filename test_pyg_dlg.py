import time

import dgl
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import ToUndirected

NUM_NODES = 64
NUM_NODE_FEAT = 32
NUM_EDGE_FEAT = 16
BATCHSIZE = 128
NRUN = 10000


def edge_to_sparse(e: torch.Tensor, a: torch.Tensor):
    # convert edge features to a 2 * num_edge * num_edge_features tensor
    # and adjacency to a 2 * num_edge tensor
    # this might not play well on gpu on pytorch using the sparse operators, but we can find workarounds without it
    a_sparse = a.to_sparse().indices()  # 2 or 3 x num_edge

    e_flat = e[*a_sparse, :]  # from (batch - opt) * n * n * num_feat to num_edge * num_feat
    if a.dim() == 3:  # batch, num_node, num_node
        # a_sparse contains (batch index, node index 1, node index 2)
        # the batch index will create problems, so we need to remove it by reindexing the node
        # for example, node 0 in the second graph should be considered as node index 0 + num_node
        a_sparse = a_sparse[1:, :] + a_sparse[0, :] * a.size(-1)  # now a 2 * num_edge tensor
    return a_sparse, e_flat



def tensor_to_pyg(x: torch.Tensor, e: torch.Tensor, a: torch.Tensor):
    # convert adjacency and edge features to sparse tensors
    a, e = edge_to_sparse(e, a)
    # build a pytorch-geometric graph
    # if batchsize is not None, this will create a singular partially connected graph
    if x.dim() == 3:
        x = x.view(-1, x.size(-1))  # (batch, num_node, node_feat) -> (batch * num_node, node_feat)
    # data = ToUndirected()(Data(x=x, edge_index=a, edge_attr=e))
    data = Data(x=x, edge_index=a, edge_attr=e)
    return data


def tensor_to_pyg_batch(x: torch.Tensor, e: torch.Tensor, a: torch.Tensor):
    bsize = x.size(0)
    graph_list = [tensor_to_pyg(x[b, :, :], e[b, :, :], a[b, :, :]) for b in range(bsize)]
    return Batch.from_data_list(graph_list)


def tensor_to_dgl(x, e, a):
    a, e = edge_to_sparse(e, a)
    g = dgl.graph((a[0, :], a[1, :]))  # creates the create
    if x.dim() == 3:
        x = x.view(-1, x.size(-1))
    g.ndata['node_features'] = x  # ndata = node features as n_node * feature_size
    g.edata['edge_features'] = e  # edata = edge features as n_edge * feature_size
    # g = dgl.add_reverse_edges(g)  # make the graph undirected
    return g


def tensor_to_dgl_batch(x, e, a):
    bsize = x.size(0)
    graph_list = [tensor_to_dgl(x[b, :, :], e[b, :, :], a[b, :, :]) for b in range(bsize)]
    return dgl.batch(graph_list)


def make_tensors(num_node: int, num_node_feat: int, num_edge_feat: int, seed=1, batchsize=None):
    torch.manual_seed(seed)
    if batchsize is None:
        # node features
        x = torch.rand(num_node, num_node_feat)
        # edge features
        e = torch.rand(num_node, num_node, num_edge_feat)
        # adjacency matrix
        a = torch.rand(num_node, num_node)
    else:
        x = torch.rand(batchsize, num_node, num_node_feat)
        e = torch.rand(batchsize, num_node, num_node, num_edge_feat)
        a = torch.rand(batchsize, num_node, num_node)
    # symmetrize adjacency
    a = 0.5 * (a + a.transpose(-2, -1))
    # remove self-loop
    for n in range(num_node):
        a[:, n, n] = 0
    a = (a > 0.5).int()
    # a = torch.triu(a, diagonal=1)  # keep only top half of adjacency, excluding diagonal
    if torch.cuda.is_available():
        return x.cuda(), e.cuda(), a.cuda()
    else:
        return x, e, a


def main():
    x, e, a = make_tensors(NUM_NODES, NUM_NODE_FEAT, NUM_EDGE_FEAT, batchsize=BATCHSIZE)
    start = time.time()
    tensor_to_pyg(x, e, a)
    for _ in range(NRUN):
        d = Batch(tensor_to_pyg(x, e, a))
    print(f'Time to create {NRUN} graphs with PYG Data: {time.time() - start} s')
    start = time.time()
    for _ in range(NRUN):
        d = tensor_to_dgl(x, e, a)
    print(f'Time to create {NRUN} graphs with DGL Data: {time.time() - start} s')
    start = time.time()
    for _ in range(NRUN):
        b = tensor_to_pyg_batch(x, e, a)
    print(f'Time to create {NRUN} graphs with PYG Batch: {time.time() - start} s')
    start = time.time()
    for _ in range(NRUN):
        b = tensor_to_dgl_batch(x, e, a)
    print(f'Time to create {NRUN} graphs with DGL Batch: {time.time() - start} s')


if __name__ == '__main__':
    main()
