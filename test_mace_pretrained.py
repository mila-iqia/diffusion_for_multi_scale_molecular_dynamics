import os
import urllib.request
from pathlib import Path
from typing import Literal, Union

import mace
import numpy as np
import torch
from mace.data import AtomicData, Configuration
from mace.tools import get_atomic_number_table_from_zs
from mace.tools.torch_geometric.dataloader import Collater


model = 'large'

NUM_NODES = 64
NUM_NODE_FEAT = 1
NUM_EDGE_FEAT = 16
BATCHSIZE = 128

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


urls = dict(
    small="https://tinyurl.com/46jrkm3v",  # 2023-12-10-mace-128-L0_energy_epoch-249.model
    medium="https://tinyurl.com/5yyxdm76",  # 2023-12-03-mace-128-L1_epoch-199.model
    large="https://tinyurl.com/5f5yavf3",  # MACE_MPtrj_2022.9.model
)
checkpoint_url = (
    urls.get(model, urls["medium"])
    if model in (None, "small", "medium", "large")
    else model
)

cache_dir = os.path.expanduser(".cache/mace")
checkpoint_url_name = "".join(
      c for c in os.path.basename(checkpoint_url) if c.isalnum() or c in "_"
 )
cached_model_path = f"{cache_dir}/{checkpoint_url_name}"
if not os.path.isfile(cached_model_path):
    os.makedirs(cache_dir, exist_ok=True)
    # download and save to disk
    print(f"Downloading MACE model from {checkpoint_url!r}")
    _, http_msg = urllib.request.urlretrieve(
        checkpoint_url, cached_model_path
    )
    if "Content-Type: text/html" in http_msg:
        raise RuntimeError(
            f"Model download failed, please check the URL {checkpoint_url}"
        )
    print(f"Cached MACE model to {cached_model_path}")
model = cached_model_path
print(model)

model = torch.load(f=model)
print(model)


class AModel(torch.nn.Module):
    def __init__(self, model_path: str):
        super(AModel, self).__init__()
        self.mace_model = torch.load(f=model_path)

    def forward(self, x):
        return self.mace_model(x)


def mace_graph() -> AtomicData:
    # create a graph using MACE native functions
    unit_cell = np.eye(3) * 10.023456  # box as a 3x3 array
    positions = np.random.random((NUM_NODES, 3)) * 10
    atom_type = 14 * np.ones(NUM_NODES)  # TODO we need a atom_type dict to convert to atomic number for MACE
    pbc = np.array([True] * 3)  # periodic boundary conditions
    graph_config = Configuration(atomic_numbers=atom_type,
                                 positions=positions,
                                 cell=unit_cell,
                                 pbc=pbc)
    z_table = get_atomic_number_table_from_zs(list(range(89)))
    graph_data = AtomicData.from_config(graph_config, z_table=z_table, cutoff=5.0)
    return graph_data


my_model = AModel(model_path=cached_model_path).float()

input = mace_graph()
collate_fn = Collater(follow_batch=[None], exclude_keys=[None])
mace_batch = collate_fn([input] * BATCHSIZE)
output = my_model(mace_batch)
print(output['node_feats'].size())