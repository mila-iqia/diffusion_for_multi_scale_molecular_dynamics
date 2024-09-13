from typing import Optional, Union

import torch
import torch.nn as nn
from faenet.model import EmbeddingBlock, FAENet
from torch.nn import Linear


class EmbeddingBlockWithSigma(EmbeddingBlock):
    def __init__(self,
                 num_gaussians,
                 num_filters,
                 hidden_channels,
                 tag_hidden_channels,
                 pg_hidden_channels,
                 phys_hidden_channels,
                 phys_embeds,
                 act,
                 second_layer_MLP,
                 sigma_hidden_channels: int = 0,
                 ):
        super(EmbeddingBlockWithSigma, self).__init__(
            num_gaussians,
            num_filters,
            hidden_channels,
            tag_hidden_channels,
            pg_hidden_channels,
            phys_hidden_channels,
            phys_embeds,
            act,
            second_layer_MLP,
        )
        self.use_sigma = sigma_hidden_channels > 0
        if self.use_sigma:
            self.sigma_embedding = Linear(1, sigma_hidden_channels)  # sigma as node feature
            # override the base class definition
            self.lin = Linear(hidden_channels + sigma_hidden_channels, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize layers weights."""
        super(EmbeddingBlockWithSigma, self).reset_parameters()
        nn.init.xavier_uniform_(self.sigma_embedding.weight)  # following FAENet original implementation
        self.sigma_embedding.bias.fill_(0)

    def forward(self, z, rel_pos, edge_attr, tag=None, sigma=None):
        """Forward pass of the Embedding block using sigma.

        Called in FAENet to generate initial atom and edge representations.

        Args:
            z (tensor): atomic numbers. (num_atoms, )
            rel_pos (tensor): relative atomic positions. (num_edges, 3)
            edge_attr (tensor): RBF of pairwise distances. (num_edges, num_gaussians)
            tag (tensor, optional): atom information specific to OCP. Defaults to None.
            sigma (tensor, optional): diffusion sigma (num_atoms, 1)

        Returns:
            (tensor, tensor): atom embeddings, edge embeddings
        """
        # --- Edge embedding --
        rel_pos = self.lin_e1(rel_pos)  # r_ij
        edge_attr = self.lin_e12(edge_attr)  # d_ij
        e = torch.cat((rel_pos, edge_attr), dim=1)
        e = self.act(e)  # can comment out

        if self.second_layer_MLP:
            e = self.act(self.lin_e2(e))

        # --- Node embedding --

        # Create atom embeddings based on its characteristic number
        h = self.emb(z)

        if self.phys_emb.device != h.device:
            self.phys_emb = self.phys_emb.to(h.device)

        # Concat tag embedding
        if self.use_tag:
            h_tag = self.tag_embedding(tag)
            h = torch.cat((h, h_tag), dim=1)

        # Concat physics embeddings
        if self.phys_emb.n_properties > 0:
            h_phys = self.phys_emb.properties[z]
            if self.use_mlp_phys:
                h_phys = self.phys_lin(h_phys)
            h = torch.cat((h, h_phys), dim=1)

        # Concat period & group embedding
        if self.use_pg:
            h_period = self.period_embedding(self.phys_emb.period[z])
            h_group = self.group_embedding(self.phys_emb.group[z])
            h = torch.cat((h, h_period, h_group), dim=1)

        if self.use_sigma:
            h_sigma = self.sigma_embedding(sigma)
            h = torch.cat((h, h_sigma), dim=1)

        # MLP
        h = self.act(self.lin(h))
        if self.second_layer_MLP:
            h = self.act(self.lin_2(h))

        return h, e


class FAENetWithSigma(FAENet):
    def __init__(
            self,
            cutoff: float = 6.0,
            preprocess: Union[str, callable] = "pbc_preprocess",
            act: str = "swish",
            max_num_neighbors: int = 40,
            hidden_channels: int = 128,
            tag_hidden_channels: int = 32,
            pg_hidden_channels: int = 32,
            phys_embeds: bool = True,
            phys_hidden_channels: int = 0,
            num_interactions: int = 4,
            num_gaussians: int = 50,
            num_filters: int = 128,
            second_layer_MLP: bool = True,
            skip_co: str = "concat",
            mp_type: str = "updownscale_base",
            graph_norm: bool = True,
            complex_mp: bool = False,
            energy_head: Optional[str] = None,  # {False, weighted-av-initial-embeds, weighted-av-final-embeds}
            out_dim: int = 1,
            pred_as_dict: bool = True,
            regress_forces: Optional[str] = None,
            force_decoder_type: Optional[str] = "mlp",
            force_decoder_model_config: Optional[dict] = {"hidden_channels": 128},
            sigma_hidden_channels: int = 0,
    ):
        super(FAENetWithSigma, self).__init__(
            cutoff=cutoff,
            preprocess=preprocess,
            act=act,
            max_num_neighbors=max_num_neighbors,
            hidden_channels=hidden_channels,
            tag_hidden_channels=tag_hidden_channels,
            pg_hidden_channels=pg_hidden_channels,
            phys_embeds=phys_embeds,
            phys_hidden_channels=phys_hidden_channels,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            num_filters=num_filters,
            second_layer_MLP=second_layer_MLP,
            skip_co=skip_co,
            mp_type=mp_type,
            graph_norm=graph_norm,
            complex_mp=complex_mp,
            energy_head=energy_head,
            out_dim=out_dim,
            pred_as_dict=pred_as_dict,
            regress_forces=regress_forces,
            force_decoder_type=force_decoder_type,
            force_decoder_model_config=force_decoder_model_config,
        )
        # override the embedding block
        self.embed_block = EmbeddingBlockWithSigma(
            self.num_gaussians,
            self.num_filters,
            self.hidden_channels,
            self.tag_hidden_channels,
            self.pg_hidden_channels,
            self.phys_hidden_channels,
            self.phys_embeds,
            self.act,
            self.second_layer_MLP,
            sigma_hidden_channels=sigma_hidden_channels
        )

    def energy_forward(self, data, preproc=True):
        """Predicts any graph-level property (e.g. energy) for 3D atomic systems.

        Override the implementation to pass sigma in the embedding block

        Args:
            data (data.Batch): Batch of graphs data objects.
            preproc (bool): Whether to apply (any given) preprocessing to the graph.
                Default to True.

        Returns:
            (dict): predicted properties for each graph (key: "energy")
                and final atomic representations (key: "hidden_state")
        """
        # Pre-process data (e.g. pbc, cutoff graph, etc.)
        # Should output all necessary attributes, in correct format.
        if preproc:
            z, batch, edge_index, rel_pos, edge_weight = self.preprocess(
                data, self.cutoff, self.max_num_neighbors
            )
        else:
            rel_pos = data.pos[data.edge_index[0]] - data.pos[data.edge_index[1]]
            z, batch, edge_index, rel_pos, edge_weight = (
                data.atomic_numbers.long(),
                data.batch,
                data.edge_index,
                rel_pos,
                rel_pos.norm(dim=-1),
            )

        edge_attr = self.distance_expansion(edge_weight)  # RBF of pairwise distances
        assert z.dim() == 1 and z.dtype == torch.long

        # Embedding block
        h, e = self.embed_block(
            z, rel_pos, edge_attr, data.tags if hasattr(data, "tags") else None, data.sigma is hasattr(data, "sigma")
        )

        # Compute atom weights for late energy head
        if self.energy_head == "weighted-av-initial-embeds":
            alpha = self.w_lin(h)
        else:
            alpha = None

        # Interaction blocks
        energy_skip_co = []
        for interaction in self.interaction_blocks:
            if self.skip_co == "concat_atom":
                energy_skip_co.append(h)
            elif self.skip_co:
                energy_skip_co.append(
                    self.output_block(h, edge_index, edge_weight, batch, alpha)
                )
            h = h + interaction(h, edge_index, e)

        # Atom skip-co
        if self.skip_co == "concat_atom":
            energy_skip_co.append(h)
            h = self.act(self.mlp_skip_co(torch.cat(energy_skip_co, dim=1)))

        energy = self.output_block(h, edge_index, edge_weight, batch, alpha)

        # Skip-connection
        energy_skip_co.append(energy)
        if self.skip_co == "concat":
            energy = self.mlp_skip_co(torch.cat(energy_skip_co, dim=1))
        elif self.skip_co == "add":
            energy = sum(energy_skip_co)

        preds = {"energy": energy, "hidden_state": h}

        return preds
