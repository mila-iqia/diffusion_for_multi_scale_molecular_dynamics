from typing import Tuple

import torch


class AtomTypesNoiser:
    """Atom types noiser.

    This class provides methods to generate noisy atom types.
    """
    @staticmethod
    def _get_uniform_noise(shape: Tuple[int]) -> torch.Tensor:
        """Get uniform noise.

        Get a sample from U(0, 1) of dimensions shape.

        Args:
            shape : the shape of the sample.

        Returns:
            gaussian_noise: a sample from U(0, 1) of dimensions shape.
        """
        return torch.rand(shape)

    @staticmethod
    def get_noisy_atom_types_sample(
            real_onehot_atom_types: torch.Tensor, q_bar: torch.Tensor
    ) -> torch.Tensor:
        r"""Get noisy atom types sample.

        This method generates a sample using the transition probabilities defined by the q_bar matrices.

        Args:
            real_onehot_atom_types : atom types of the real sample. Assumed to be a one-hot vector. The size is assumed
                to be (..., num_classes + 1) where num_classes is the number of atoms.
            q_bar : cumulative transition matrices i.e. the q_bar in q(a_t | a_0) = a_0 \bar{Q}_t. Assumed to be of size
                (..., num_classes + 1, num_classes + 1)

        Returns:
            noisy_atom_types: a sample of noised atom types as classes, not 1-hot, of the same shape as
            real_onehot_atom_types except for the last dimension that is removed.
        """
        assert real_onehot_atom_types.shape == q_bar.shape[:-1], \
            "q_bar array first dimensions should match real_atom_types array"

        u_scores = AtomTypesNoiser._get_uniform_noise(
            real_onehot_atom_types.shape
        ).to(q_bar)
        # we need to sample from q(x_t | x_0)
        posterior_xt = q_xt_bar_xo(real_onehot_atom_types, q_bar)
        # gumbel trick to sample from a distribution
        noise = -torch.log(-torch.log(u_scores)).to(real_onehot_atom_types.device)
        noisy_atom_types = torch.log(posterior_xt) + noise
        noisy_atom_types = torch.argmax(noisy_atom_types, dim=-1)
        return noisy_atom_types
