import itertools
from typing import AnyStr, Dict, Tuple

import torch

from diffusion_for_multi_scale_molecular_dynamics.models.score_networks import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.mlp_score_network import (
    MLPScoreNetwork, MLPScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, NOISY_AXL_COMPOSITION)


class PermutationMLPScoreNetwork(ScoreNetwork):
    """Permutation MLP score network.

    An MLP score network that is made invariant to permutations of the atoms.
    This is done in a brute force way and will not scale well for large numbers of atoms.

    This class is mostly for exploration and understanding, not for serious production calculations.
    """

    def __init__(self, hyper_params: MLPScoreNetworkParameters):
        """Permutation MLP score network constructor."""
        super().__init__(hyper_params)

        self.mlp_score_network = MLPScoreNetwork(hyper_params)

        self.all_permutation_indices, self.all_inverse_permutation_indices = (
            self._get_all_permutation_indices(self.mlp_score_network.natoms))

    @staticmethod
    def _get_all_permutation_indices(number_of_atoms) -> Tuple[torch.Tensor, torch.Tensor]:
        # Shape : [number of permutations, number of atoms]
        perm_indices = torch.stack(
            [
                torch.tensor(perm)
                for perm in itertools.permutations(range(number_of_atoms))
            ]
        )

        inverse_perm_indices = perm_indices.argsort(dim=1)

        return perm_indices, inverse_perm_indices

    def _forward_unchecked(
        self, batch: Dict[AnyStr, torch.Tensor], conditional: bool = False
    ) -> AXL:

        atom_types = batch[NOISY_AXL_COMPOSITION].A
        relative_coordinates = batch[NOISY_AXL_COMPOSITION].X
        lattice = batch[NOISY_AXL_COMPOSITION].L

        list_batch_keys = list(batch.keys())
        list_batch_keys.remove(NOISY_AXL_COMPOSITION)

        list_output = []
        for perm, inv_perm in zip(self.all_permutation_indices, self.all_inverse_permutation_indices):

            permuted_composition = AXL(A=atom_types[:, perm],
                                       X=relative_coordinates[:, perm],
                                       L=lattice)
            permuted_batch = {NOISY_AXL_COMPOSITION: permuted_composition}
            for key in list_batch_keys:
                permuted_batch[key] = batch[key]

            perm_axl_output = self.mlp_score_network._forward_unchecked(permuted_batch, conditional)
            axl_output = AXL(A=perm_axl_output.A[:, inv_perm],
                             X=perm_axl_output.X[:, inv_perm],
                             L=lattice)
            list_output.append(axl_output)

        average_a = torch.stack([axl.A for axl in list_output]).mean(dim=0)
        average_x = torch.stack([axl.X for axl in list_output]).mean(dim=0)
        average_L = torch.stack([axl.L for axl in list_output]).mean(dim=0)

        return AXL(A=average_a, X=average_x, L=average_L)
