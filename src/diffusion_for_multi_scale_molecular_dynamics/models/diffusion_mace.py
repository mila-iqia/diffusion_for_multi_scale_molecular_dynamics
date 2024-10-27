from typing import AnyStr, Callable, Dict, List, Optional, Type, Union

import torch
from e3nn import o3
from e3nn.nn import Activation, BatchNorm, NormActivation
from mace.modules import (
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearNodeEmbeddingBlock,
    RadialEmbeddingBlock,
)
from mace.modules.utils import get_edge_vectors_and_lengths
from torch_geometric.data import Data

from diffusion_for_multi_scale_molecular_dynamics.models.mace_utils import (
    get_adj_matrix,
    reshape_from_e3nn_to_mace,
    reshape_from_mace_to_e3nn,
)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL,
    CARTESIAN_FORCES,
    NOISE,
    NOISY_AXL,
    UNIT_CELL,
)


class LinearVectorReadoutBlock(torch.nn.Module):
    """Linear readout block for vector representation."""

    def __init__(self, irreps_in: o3.Irreps):
        """Init method."""
        super().__init__()
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=o3.Irreps("1o"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return self.linear(x)


class LinearClassificationReadoutBlock(torch.nn.Module):
    """Linear readout for scalar representation."""

    def __init__(self, irreps_in: o3.Irreps, num_classes: int):
        """Init method."""
        super().__init__()
        self.linear = o3.Linear(
            irreps_in=irreps_in, irreps_out=o3.Irreps(f"{num_classes}x0e")
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return self.linear(x)


def input_to_diffusion_mace(
    batch: Dict[AnyStr, torch.Tensor],
    radial_cutoff: float,
    num_atom_types: int = 1,
) -> Data:
    """Convert score network input to Diffusion MACE input.

    Args:
        batch: score network input dictionary
        radial_cutoff : largest distance between neighbors.
        num_atom_types: number of atomic species, including the MASK class

    Returns:
        pytorch-geometric graph data compatible with MACE forward
    """
    cartesian_positions = batch[NOISY_AXL].X
    batch_size, n_atom_per_graph, spatial_dimension = cartesian_positions.shape
    device = cartesian_positions.device

    # TODO replace with AXL L
    basis_vectors = batch[UNIT_CELL]  # batch, spatial_dimension, spatial_dimension

    adj_matrix, shift_matrix, batch_tensor, num_edges = get_adj_matrix(
        positions=cartesian_positions,
        basis_vectors=basis_vectors,
        radial_cutoff=radial_cutoff,
    )

    # node features are int corresponding to atom type
    # TODO handle different atom types
    atom_types = batch[NOISY_AXL].A
    node_attrs = torch.nn.functional.one_hot(
        atom_types.long(), num_classes=num_atom_types
    ).to(atom_types)
    # The node diffusion scalars will be the diffusion noise sigma, which is constant for each structure in the batch.
    # We broadcast to each node to avoid complex broadcasting logic within the model itself.
    # TODO: it might be better to define the noise as a 'global' graph attribute, and find 'the right way' of
    #   mixing it with bona fide node features within the model.
    noises = (
        batch[NOISE] + 1
    )  # [batch_size, 1]  - add 1 to avoid getting a zero at sigma=0 (initialization issues)
    node_diffusion_scalars = noises.repeat_interleave(
        n_atom_per_graph, dim=0
    )  # [flat_batch_size, 1]
    edge_diffusion_scalars = noises.repeat_interleave(
        num_edges.long().to(device=noises.device), dim=0
    )

    # [batchsize * natoms, spatial dimension]
    flat_cartesian_positions = cartesian_positions.view(-1, spatial_dimension)

    # pointer tensor that yields the first node index for each batch - this is a fixed tensor in our case
    ptr = torch.arange(
        0, n_atom_per_graph * batch_size + 1, step=n_atom_per_graph
    )  # 0, natoms, 2 * natoms, ...

    flat_basis_vectors = basis_vectors.view(
        -1, spatial_dimension
    )  # batch * spatial_dimension, spatial_dimension
    # create the pytorch-geometric graph

    forces = batch[CARTESIAN_FORCES].view(
        -1, spatial_dimension
    )  # batch * n_atom_per_graph, spatial dimension

    graph_data = Data(
        edge_index=adj_matrix,
        node_attrs=node_attrs.to(device),
        node_diffusion_scalars=node_diffusion_scalars.to(device),
        edge_diffusion_scalars=edge_diffusion_scalars.to(device),
        positions=flat_cartesian_positions,
        ptr=ptr.to(device),
        batch=batch_tensor.to(device),
        shifts=shift_matrix,
        cell=flat_basis_vectors,
        forces=forces,
    )
    return graph_data


class DiffusionMACE(torch.nn.Module):
    """This architecture is inspired by MACE, but modified to take in diffusion sigma as a scalar.

    the original article
        "MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields"
    will be referenced in the comments as the PAPER.
    """

    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        num_edge_hidden_layers: int,
        edge_hidden_irreps: o3.Irreps,
        max_ell: int,
        interaction_cls: Type[InteractionBlock],
        interaction_cls_first: Type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        mlp_irreps: o3.Irreps,
        number_of_mlp_layers: int,
        avg_num_neighbors: float,
        correlation: Union[int, List[int]],
        gate: Optional[Callable],
        radial_MLP: List[int],
        radial_type: Optional[str] = "bessel",
        condition_embedding_size: int = 64,  # dimension of the conditional variable embedding - assumed to be l=1 (odd)
        use_batchnorm: bool = False,
        tanh_after_interaction: bool = True,
    ):
        """Init method."""
        super().__init__()
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )

        # The 'num_interactions' corresponds to T, the number of layers in the GNN in eq. 13 of the PAPER.
        self.register_buffer(
            "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        )

        # "correlation" corresponds to the "nu" index that appears in the definition of B^{(t)} in eq. 10 of the PAPER.
        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions

        # Define the spatial dimension to avoid magic numbers later
        self.spatial_dimension = 3

        # define the "0e" representation as a constant to avoid "magic numbers" below.
        scalar_irrep = o3.Irrep(0, 1)

        # Apply an MLP with a bias on the scalar diffusion time-like  input.
        number_of_node_scalar_dimensions = 1
        number_of_hidden_diffusion_scalar_dimensions = mlp_irreps.count(scalar_irrep)

        diffusion_scalar_irreps_in = o3.Irreps(
            [(number_of_node_scalar_dimensions, scalar_irrep)]
        )
        diffusion_scalar_irreps_out = o3.Irreps(
            [(number_of_hidden_diffusion_scalar_dimensions, scalar_irrep)]
        )

        self.diffusion_scalar_embedding = torch.nn.Sequential()
        linear = o3.Linear(
            irreps_in=diffusion_scalar_irreps_in,
            irreps_out=diffusion_scalar_irreps_out,
            biases=True,
        )
        self.diffusion_scalar_embedding.append(linear)
        non_linearity = Activation(irreps_in=diffusion_scalar_irreps_out, acts=[gate])
        for _ in range(number_of_mlp_layers):
            self.diffusion_scalar_embedding.append(non_linearity)

            linear = o3.Linear(
                irreps_in=diffusion_scalar_irreps_out,
                irreps_out=diffusion_scalar_irreps_out,
                biases=True,
            )
            self.diffusion_scalar_embedding.append(linear)

        # The node_attr is the one-hot version of the atom types.
        node_attr_irreps = o3.Irreps([(num_elements, scalar_irrep)])

        # Perform a tensor product to mix the diffusion scalar and node attributes
        self.attribute_mixing = o3.FullyConnectedTensorProduct(
            irreps_in1=diffusion_scalar_irreps_out,
            irreps_in2=node_attr_irreps,
            irreps_out=node_attr_irreps,
            irrep_normalization="norm",
        )

        number_of_hidden_scalar_dimensions = hidden_irreps.count(scalar_irrep)

        node_feats_irreps = o3.Irreps(
            [(number_of_hidden_scalar_dimensions, scalar_irrep)]
        )
        # The "node embedding" corresponds to W^{(1)} in the definition of A^{(1)}, eq. 9 of the PAPER.
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )

        # The "radial embedding" corresponds to the "R^(1)" object in the definition of A^{(1)}, eq. 9 of the PAPER.
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            radial_type=radial_type,
        )
        edge_feats_irreps = o3.Irreps([(self.radial_embedding.out_dim, scalar_irrep)])

        if num_edge_hidden_layers > 0:
            self.edge_attribute_mixing = o3.FullyConnectedTensorProduct(
                irreps_in1=diffusion_scalar_irreps_out,
                irreps_in2=edge_feats_irreps,
                irreps_out=edge_hidden_irreps,
                irrep_normalization="norm",
            )
            self.edge_hidden_layers = torch.nn.Sequential()
            edge_non_linearity = Activation(irreps_in=edge_hidden_irreps, acts=[gate])
            for i in range(num_edge_hidden_layers):
                if i != 0:
                    self.edge_hidden_layers.append(edge_non_linearity)
                edge_hidden_layer = o3.Linear(
                    irreps_in=edge_hidden_irreps,
                    irreps_out=edge_hidden_irreps,
                    biases=False,
                )
                self.edge_hidden_layers.append(edge_hidden_layer)
        else:
            self.edge_attribute_mixing, self.edge_hidden_layers = None, None
            edge_hidden_irreps = edge_feats_irreps

        # The "spherical harmonics" correspond to Y_{lm} in the definition of A^{(1)}, eq. 9 of the PAPER.
        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        interaction_irreps = (
            (sh_irreps * number_of_hidden_scalar_dimensions).sort()[0].simplify()
        )
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        # Interactions and node operations

        # The first 'interaction block' takes care of (1) the tensor product between W^{(1)}, Y_{lm} and R^{(1)}, and
        # (2) takes care of the sum on neighbors "j", namely 'message passing'. The output of this interaction should
        # thus be A^{(1)}. This takes the form of 'node features', namely a tensor with multiple channels along
        # different (lm) angular momenta for each node.
        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_hidden_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        if tanh_after_interaction:
            self.interactions_tanh = torch.nn.ModuleList(
                [NormActivation(inter.target_irreps, torch.tanh)]
            )
        else:
            self.interactions_tanh = None

        # 'sc' means 'self-connection', namely a 'residual-like' connection, h^{t+1} = m^{t} + (sc) x h^{(t)}
        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        # The 'product' operation corresponds to computing the tensor product B^{(1)} from the various channels
        # of A^{(1)}, as described in eq. 10 of the PAPER, and then mixing the various channels to
        # get the 'message' that will update the node features, as in eq. 11 of the PAPER. Finally,
        # the representations h are updated with a residual block, as in eq. 12 of the PAPER.
        # A key difference with the 'interaction' block is that all of these operations are 'on node', ie, there
        # is no internode communication. The 'sum on neighbors' already occurred in the interaction block.
        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.batch_norms = torch.nn.ModuleList([BatchNorm(node_feats_irreps)])

        hidden_irreps_out = hidden_irreps

        for i in range(num_interactions - 1):
            # Inter computes A^{(t)} from h^{(t)}, doing a sum-on-neighbors operation.
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_hidden_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)

            if self.interactions_tanh is not None:
                self.interactions_tanh.append(
                    NormActivation(interaction_irreps, torch.tanh)
                )

            # prod compute h^{(t+1)} from A^{(t)} and h^{(t)}, computing B and the messages internally.
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)

            if self.use_batchnorm:
                bn = BatchNorm(hidden_irreps)
                self.batch_norms.append(bn)

        # the output is a single vector.
        self.vector_readout = LinearVectorReadoutBlock(irreps_in=hidden_irreps_out)

        # and an output for atom classification
        self.classification_readout = LinearClassificationReadoutBlock(
            irreps_in=hidden_irreps_out, num_classes=num_elements
        )

        # Apply a MLP with a bias on the forces as a conditional feature. This would be a 1o irrep
        forces_irreps_in = o3.Irreps("1x1o")
        forces_irreps_embedding = o3.Irreps(f"{condition_embedding_size}x1o")
        self.condition_embedding_layer = o3.Linear(
            irreps_in=forces_irreps_in, irreps_out=forces_irreps_embedding, biases=False
        )  # can't have biases with 1o irreps

        # conditional layers for the forces as a conditional feature to guide the diffusion
        self.conditional_layers = torch.nn.ModuleList([])
        for _ in range(num_interactions):
            cond_layer = o3.Linear(
                irreps_in=forces_irreps_embedding,
                irreps_out=hidden_irreps_out,
                biases=False,
            )
            self.conditional_layers.append(cond_layer)

    def forward(self, data: Dict[str, torch.Tensor], conditional: bool = False) -> AXL:
        """Forward method."""
        # Setup

        # Augment the node attributes with information from the diffusion scalar.
        diffusion_scalar_embeddings = self.diffusion_scalar_embedding(
            data["node_diffusion_scalars"]
        )
        raw_node_attributes = data["node_attrs"]
        augmented_node_attributes = self.attribute_mixing(
            diffusion_scalar_embeddings, raw_node_attributes
        )

        # Embeddings
        node_feats = self.node_embedding(augmented_node_attributes)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)
        if self.edge_attribute_mixing is not None:
            edge_diffusion_scalar_embeddings = self.diffusion_scalar_embedding(
                data["edge_diffusion_scalars"]
            )
            augmented_edge_attributes = self.edge_attribute_mixing(
                edge_diffusion_scalar_embeddings, edge_feats
            )
            edge_feats = self.edge_hidden_layers(augmented_edge_attributes)

        forces_embedding = self.condition_embedding_layer(
            data["forces"]
        )  # 0e + 1o embedding

        for i, (interaction, product, cond_layer) in enumerate(
            zip(self.interactions, self.products, self.conditional_layers)
        ):
            if self.use_batchnorm:
                node_feats = self.batch_norms[i](node_feats)

            node_feats, sc = interaction(
                node_attrs=augmented_node_attributes,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )

            if self.interactions_tanh is not None:
                # reshape from (node, channels, (l_max + 1)**2) to a (node, -1) tensor compatible with e3nn
                node_feats = reshape_from_mace_to_e3nn(
                    node_feats, self.interactions_tanh[i].irreps_in
                )
                # apply non-linearity
                node_feats = self.interactions_tanh[i](node_feats)
                # reshape from e3nn shape to mace format (node, channels, (l_max+1)**2)
                node_feats = reshape_from_e3nn_to_mace(
                    node_feats, self.interactions_tanh[i].irreps_out
                )

            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=augmented_node_attributes,
            )
            if (
                conditional
            ):  # modify the node features to account for the conditional features i.e. forces
                force_embed = cond_layer(forces_embedding)
                node_feats += force_embed

        # Outputs
        vectors_output = self.vector_readout(node_feats)
        classification_output = self.classification_readout(node_feats)
        axl_output = AXL(
            A=classification_output,
            X=vectors_output,
            L=torch.zeros_like(classification_output),
        )
        return axl_output
