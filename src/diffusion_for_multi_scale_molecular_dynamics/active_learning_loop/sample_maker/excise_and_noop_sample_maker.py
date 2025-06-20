from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.base_sample_maker import (
    BaseExciseSampleMaker, BaseExciseSampleMakerArguments)
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL


@dataclass(kw_only=True)
class ExciseAndNoOpSampleMakerArguments(BaseExciseSampleMakerArguments):
    """Parameters for a trivial sample maker method."""
    algorithm: str = "excise_and_noop"


class ExciseAndNoOpSampleMaker(BaseExciseSampleMaker):
    """Trivial sample maker that reproduces the excised environment without modifications."""

    def make_samples_from_constrained_substructure(
        self,
        substructure: AXL,
        active_atom_index: int,
        num_samples: int = 1,
    ) -> Tuple[List[AXL], List[int], List[Dict[str, Any]]]:
        """Create new samples using a constrained structure.

        Since this is a "no op" implementation, the output will simply be the input.

        Args:
            substructure: constrained atoms described as an AXL.
            active_atom_index: index of the "active atom" in the input substructure.
            num_samples: number of samples to make. Defaults to 1.

        Returns:
            list_samples: the list of created samples. The length of the list should match num_samples.
            list_active_atom_indices: for each created sample, the index of the "active atom", ie the
                atom at the center of the excised region.
            list_info: list of samples additional information.
        """
        list_samples = num_samples * [substructure]
        list_active_atom_indices = num_samples * [active_atom_index]
        list_info = num_samples * [{}]
        return list_samples, list_active_atom_indices, list_info

    def filter_made_samples(self, structures: List[AXL]) -> List[AXL]:
        """Return identical structures."""
        return structures
