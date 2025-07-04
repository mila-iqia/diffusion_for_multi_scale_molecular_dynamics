from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.atom_selector.base_atom_selector import \
    BaseAtomSelector
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
    """No Op Sample Maker.

    This is a trivial sample maker that reproduces the input structures without modification.
    An Excisor is still needed in order to identify the active environments.
    """
    def __init__(self, sample_maker_arguments: BaseSampleMakerArguments, atom_selector: BaseAtomSelector):
        """Init method."""
        super().__init__(sample_maker_arguments, atom_selector)

    def make_samples(
        self,
        structure: AXL,
        uncertainty_per_atom: np.array,
    ) -> Tuple[List[AXL], List[np.array], List[Dict[str, Any]]]:
        """Noop make samples."""
        central_atom_indices = self.atom_selector.select_central_atoms(uncertainty_per_atom)
        return [structure], [central_atom_indices], [self._create_sample_info_dictionary(structure)]

    def filter_made_samples(self, structures: List[AXL]) -> List[AXL]:
        """Noop filter samples."""
        return structures
