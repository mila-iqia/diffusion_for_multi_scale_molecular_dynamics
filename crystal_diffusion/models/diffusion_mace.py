from typing import AnyStr, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
from e3nn import o3
from mace.modules import (AtomicEnergiesBlock, EquivariantProductBasisBlock,
                          InteractionBlock, LinearNodeEmbeddingBlock,
                          LinearReadoutBlock, NonLinearReadoutBlock,
                          RadialEmbeddingBlock)
from mace.modules.utils import (get_edge_vectors_and_lengths, get_outputs,
                                get_symmetric_displacement)
from mace.tools.scatter import scatter_sum
from torch_geometric.data import Data

from crystal_diffusion.models.mace_utils import get_adj_matrix
from crystal_diffusion.namespace import (NOISE, NOISY_CARTESIAN_POSITIONS,
                                         UNIT_CELL)


def input_to_diffusion_mace(batch: Dict[AnyStr, torch.Tensor], radial_cutoff: float) -> Data:
    """Convert score network input to Diffusion MACE input.

    Args:
        batch: score network input dictionary
        radial_cutoff : largest distance between neighbors.

    Returns:
        pytorch-geometric graph data compatible with MACE forward
    """
    cartesian_positions = batch[NOISY_CARTESIAN_POSITIONS]
    batch_size, n_atom_per_graph, spatial_dimension = cartesian_positions.shape
    device = cartesian_positions.device

    basis_vectors = batch[UNIT_CELL]  # batch, spatial_dimension, spatial_dimension

    adj_matrix, shift_matrix, batch_tensor = get_adj_matrix(positions=cartesian_positions,
                                                            basis_vectors=basis_vectors,
                                                            radial_cutoff=radial_cutoff)

    # The node attributes will be the diffusion noise sigma, which is constant for each structure in the batch.
    noises = batch[NOISE]  # [batch_size, 1]
    node_attrs = noises.repeat_interleave(n_atom_per_graph, dim=0)  # [flat_batch_size, 1]

    # [batchsize * natoms, spatial dimension]
    flat_cartesian_positions = cartesian_positions.view(-1, spatial_dimension)

    # pointer tensor that yields the first node index for each batch - this is a fixed tensor in our case
    ptr = torch.arange(0, n_atom_per_graph * batch_size + 1, step=n_atom_per_graph)  # 0, natoms, 2 * natoms, ...

    flat_basis_vectors = basis_vectors.view(-1, spatial_dimension)  # batch * spatial_dimension, spatial_dimension
    # create the pytorch-geometric graph
    graph_data = Data(edge_index=adj_matrix,
                      node_attrs=node_attrs.to(device),
                      positions=flat_cartesian_positions,
                      ptr=ptr.to(device),
                      batch=batch_tensor.to(device),
                      shifts=shift_matrix,
                      cell=flat_basis_vectors
                      )
    return graph_data


class DiffusionMACE(torch.nn.Module):
    """This architecture is inspired by MACE, but modified to take in diffusion time as a scalar."""
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        avg_num_neighbors: float,
        atomic_numbers: List[int],
        correlation: Union[int, List[int]],
        gate: Optional[Callable],
        radial_MLP: List[int],
        radial_type: Optional[str] = "bessel",
    ):
        """Init method."""
        assert num_elements == 1, "only a single element can be used at this time. Set 'num_elements' to 1."
        assert len(atomic_numbers) == 1, \
            "only a single element can be used at this time. Set 'atomic_numbers' to length 1."
        super().__init__()
        self.register_buffer(
            "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        )
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions
        # Embedding

        # define the "0e" representation as a constant to avoid "magic numbers" below.
        scalar_irrep = o3.Irrep(0, 1)

        # we assume a single 'scalar' diffusion time-like  input.
        number_of_node_scalar_dimensions = 1

        node_attr_irreps = o3.Irreps([(number_of_node_scalar_dimensions, scalar_irrep)])
        number_of_hidden_scalar_dimensions = hidden_irreps.count(scalar_irrep)

        # The node features will be only '0e' scalars, with a number of channels identical with the hidden irreps
        node_feats_irreps = o3.Irreps([(number_of_hidden_scalar_dimensions, scalar_irrep)])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
        )
        edge_feats_irreps = o3.Irreps([(self.radial_embedding.out_dim, scalar_irrep)])

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        interaction_irreps = (sh_irreps * number_of_hidden_scalar_dimensions).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearReadoutBlock(hidden_irreps))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = str(
                    hidden_irreps[0]
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(hidden_irreps_out, MLP_irreps, gate)
                )
            else:
                self.readouts.append(LinearReadoutBlock(hidden_irreps))

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Forward method."""
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        energies = [e0]
        node_energies_list = [node_e0]
        node_feats_list = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_feats_list.append(node_feats)
            node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            energy = scatter_sum(
                src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs
            )  # [n_graphs,]
            energies.append(energy)
            node_energies_list.append(node_energies)

        # Concatenate node features
        node_feats_out = torch.cat(node_feats_list, dim=-1)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]

        # Outputs
        forces, virials, stress = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "node_feats": node_feats_out,
        }
