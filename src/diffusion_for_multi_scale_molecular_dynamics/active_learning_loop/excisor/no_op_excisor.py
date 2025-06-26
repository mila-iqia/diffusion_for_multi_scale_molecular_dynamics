from dataclasses import dataclass
from typing import Tuple

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.base_excisor import (
    BaseEnvironmentExcision, BaseEnvironmentExcisionArguments)
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL


@dataclass(kw_only=True)
class NoOpExcisionArguments(BaseEnvironmentExcisionArguments):
    """Parameters for a trivial excision method."""
    algorithm: str = "noop"


class NoOpExcision(BaseEnvironmentExcision):
    """Trivial environment excision method that returns the full environment without modifications."""
    def _excise_one_environment(
        self,
        structure: AXL,
        central_atom_idx: int,
    ) -> Tuple[AXL, int]:
        return structure, central_atom_idx
