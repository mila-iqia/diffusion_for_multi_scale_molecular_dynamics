"""Configuration dataclass defining a GRACE-FS model."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from pymatgen.core import Element

MODEL_SIZES = ("small", "medium", "large")


@dataclass(kw_only=True)
class GraceConfiguration:
    """A GRACE-FS model, trained with gracemaker and run through the lammps grace/fs pair_style.

    The model architecture is selected by 'size' (small/medium/large) through tensorpotential's preset
    system; an expert can override the resulting 'model_kwargs' afterwards.
    """

    elements: List[str]
    cutoff: float = 5.0  # top-level 'cutoff' (Angstrom); the preset's own 'rcut' is dropped in favor of this.
    seed: int = 1

    # 'potential' block. 'model_kwargs' is derived from (preset, size) when left None.
    preset: str = "FS"
    size: str = "medium"
    model_kwargs: Optional[Dict] = None
    scale: bool = True

    # 'data' block; the trainer fills filename / test_filename at fit time.
    reference_energy: Union[int, str, Dict] = 0

    # 'fit' block.
    loss: Dict = field(
        default_factory=lambda: dict(
            energy=dict(weight=16, type="huber", delta=0.01),
            forces=dict(weight=32, type="huber", delta=0.01),
        )
    )
    optimizer: str = "BFGS"
    opt_params: Dict = field(default_factory=lambda: dict(maxcor=100, maxls=20, gtol=1.0e-8, iprint=-1))
    target_total_updates: int = 500
    batch_size: int = 16
    test_batch_size: int = 64
    jit_compile: bool = True  # JIT speeds up production (long) fits; its one-off compilation dominates short ones.

    def __post_init__(self):
        """Validate the configuration and, if needed, derive the model architecture from the size."""
        if len(self.elements) == 0:
            raise ValueError("The list of elements should not be empty.")
        if len(set(self.elements)) != len(self.elements):
            raise ValueError("The elements are not unique!")
        for element in self.elements:
            try:
                Element(element)
            except Exception:
                raise ValueError(f"Expected real elements; got '{element}'.")

        if self.cutoff <= 0.0:
            raise ValueError("The cutoff should be positive.")
        if self.seed < 0:
            raise ValueError("The gracemaker seed should be non-negative.")
        if self.size not in MODEL_SIZES:
            raise ValueError(f"The size should be one of {MODEL_SIZES}; got '{self.size}'.")
        if self.target_total_updates <= 0:
            raise ValueError("The target_total_updates should be positive.")
        if self.batch_size <= 0 or self.test_batch_size <= 0:
            raise ValueError("The batch sizes should be positive.")

        if self.model_kwargs is None:
            self.model_kwargs = self._preset_model_kwargs(self.preset, self.size)

    @staticmethod
    def _preset_model_kwargs(preset: str, size: str) -> Dict:
        """Return the model kwargs for a (preset, size) from tensorpotential, dropping the preset's rcut."""
        try:
            from tensorpotential.potentials import get_preset_settings
            preset_settings = get_preset_settings(preset)
        except ImportError:
            from tensorpotential.cli.prepare import allowed_preset_complexities
            preset_settings = allowed_preset_complexities[preset]

        model_kwargs = dict(preset_settings[size])
        model_kwargs.pop("rcut", None)
        return model_kwargs
