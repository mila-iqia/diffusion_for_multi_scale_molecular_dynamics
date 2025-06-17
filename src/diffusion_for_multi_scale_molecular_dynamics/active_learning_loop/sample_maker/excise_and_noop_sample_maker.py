from dataclasses import dataclass
from typing import List

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
        num_samples: int = 1,
    ) -> List[AXL]:
        """Create new samples using a constrained structure.

        Args:
            substructure: constrained atoms described as an AXL
            num_samples: number of samples to make. Defaults to 1.

        Returns:
            list of samples created. The length of the list should match num_samples.
            list of additional information on samples created.
        """
        return [substructure] * num_samples, [{}] * num_samples

    def filter_made_samples(self, structures: List[AXL]) -> List[AXL]:
        """Return identical structures."""
        return structures
