"""The ARTn input parameters written to an 'artn.in' (the &ARTN_PARAMETERS namelist plus the push)."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ArtnInputConfiguration:
    """All parameters written to an ARTn 'artn.in' file (the &ARTN_PARAMETERS namelist plus the push).

    push_ids selects the atom ARTn pushes; push_add_const, when set, is its four-component push constraint
    (omitted for the radial push modes). extra holds any other ARTn variable not exposed as a field.
    """

    push_ids: Optional[int] = None
    push_add_const: Optional[List[float]] = None
    engine_units: str = "lammps/metal"
    verbose: int = 2
    ninit: int = 2
    nevalf_max: int = 50000
    nperp_limitation: List[int] = field(default_factory=lambda: [4, 8, 12, 16, 32])
    lpush_final: bool = True
    nnewchance: int = 10
    nsmooth: int = 5
    forc_thr: float = 0.01
    delr_thr: float = 0.1
    push_step_size: float = 0.25
    push_dist_thr: float = 3.0
    push_mode: str = "rad"
    lanczos_disp: float = 1e-4
    lanczos_max_size: int = 16
    lanczos_min_size: int = 3
    lanczos_eval_conv_thr: float = 1e-2
    eigen_step_size: float = 0.1
    push_over: float = 2.0
    extra: Dict = field(default_factory=dict)

    def to_namelist(self) -> Dict:
        """Return the ordered &ARTN_PARAMETERS name -> value mapping (push_add_const keyed to push_ids)."""
        namelist = {
            "engine_units": self.engine_units,
            "verbose": self.verbose,
            "ninit": self.ninit,
            "nevalf_max": self.nevalf_max,
            "nperp_limitation": self.nperp_limitation,
            "lpush_final": self.lpush_final,
            "nnewchance": self.nnewchance,
            "nsmooth": self.nsmooth,
            "forc_thr": self.forc_thr,
            "delr_thr": self.delr_thr,
            "push_step_size": self.push_step_size,
            "push_dist_thr": self.push_dist_thr,
            "push_mode": self.push_mode,
            "lanczos_disp": self.lanczos_disp,
            "lanczos_max_size": self.lanczos_max_size,
            "lanczos_min_size": self.lanczos_min_size,
            "lanczos_eval_conv_thr": self.lanczos_eval_conv_thr,
            "eigen_step_size": self.eigen_step_size,
            "push_over": self.push_over,
        }
        namelist.update(self.extra)
        namelist["push_ids"] = self.push_ids
        if self.push_add_const is not None:
            namelist[f"push_add_const(:,{self.push_ids})"] = self.push_add_const
        return namelist
