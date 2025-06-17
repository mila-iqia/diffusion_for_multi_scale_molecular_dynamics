from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.base_sample_maker import (
    BaseSampleMaker, BaseSampleMakerArguments)
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL


@dataclass(kw_only=True)
class NoOpSampleMakerArguments(BaseSampleMakerArguments):
    """Parameters for a trivial sample maker method."""
    # Note that the fields must be TYPED exactly the same was as in the base class, or else the
    # inheritance breaks.
    algorithm: str = "noop"
    sample_box_strategy: str = "noop"


class NoOpSampleMaker(BaseSampleMaker):
    """Trivial sample maker that reproduces the excised environment without modifications."""

    def make_samples(
        self,
        structure: AXL,
        uncertainty_per_atom: np.array,
    ) -> Tuple[List[AXL], List[Dict[str, Any]]]:
        """Noop make samples."""
        return [structure], [{"uncertainty_per_atom": uncertainty_per_atom}]

    def filter_made_samples(self, structures: List[AXL]) -> List[AXL]:
        """Noop filter samples."""
        return structures
