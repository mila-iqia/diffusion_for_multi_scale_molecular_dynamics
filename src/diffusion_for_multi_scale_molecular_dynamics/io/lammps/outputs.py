"""Read LAMMPS output into ase.Atoms, forces and per-atom uncertainties.

The dumps are LAMMPS *text* dumps (``dump ... custom``); they are read with ``ase.io.read`` so the box
(orthogonal or triclinic), the atomic positions and the species are parsed by ase rather than by hand. The
total energy is not part of a text dump and is read separately by the caller (the single-point calculator
writes it to its own file).
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from ase import Atoms
from ase.io import read as read_ase

from diffusion_for_multi_scale_molecular_dynamics.namespace import \
    UNCERTAINTY_FIELD


def extract_all_fields_from_dump(
    lammps_dump_path: Path,
    uncertainty_field: Union[str, None] = UNCERTAINTY_FIELD,
) -> Tuple[List[Atoms], List[np.ndarray], List[Optional[np.ndarray]]]:
    """Extract configurations, forces and per-atom uncertainties from a LAMMPS text dump.

    Args:
        lammps_dump_path: path to a LAMMPS text dump (``dump ... custom``); may hold several frames.
        uncertainty_field: name of the per-atom uncertainty column to read, or None if there is none.

    Returns:
        list_atoms: the configurations in the dump file.
        list_forces: the forces in the dump file, in the same order as the configurations.
        list_uncertainties: the per-atom uncertainties in the dump file, if present, else None.
    """
    frames = read_ase(str(lammps_dump_path), format="lammps-dump-text", index=":")
    if not isinstance(frames, list):
        frames = [frames]

    list_atoms = []
    list_forces = []
    list_uncertainties = []
    for atoms in frames:
        list_atoms.append(atoms)
        list_forces.append(atoms.get_forces())
        if uncertainty_field is not None and uncertainty_field in atoms.arrays:
            # ase stores a scalar per-atom column as (N, 1); flatten it to the (N,) per-atom contract.
            list_uncertainties.append(np.asarray(atoms.arrays[uncertainty_field], dtype=float).ravel())
        else:
            list_uncertainties.append(None)

    return list_atoms, list_forces, list_uncertainties


def extract_timesteps_from_dump(lammps_dump_path: Path) -> List[int]:
    """Return the LAMMPS timestep of each frame in a text dump."""
    frames = read_ase(str(lammps_dump_path), format="lammps-dump-text", index=":")
    if not isinstance(frames, list):
        frames = [frames]
    return [int(atoms.info["timestep"]) for atoms in frames]


def extract_all_fields_from_cfg(
    configuration_output_path: Path,
    uncertainty_field: Union[str, None] = UNCERTAINTY_FIELD,
) -> Tuple[List[Atoms], List[np.ndarray], List[Optional[np.ndarray]]]:
    """Extract configurations, forces and uncertainties from a MTP '.cfg' output."""
    raise NotImplementedError("Reading a '.cfg' configuration output is not implemented yet.")


def extract_all_fields(
    configuration_output_path: Path,
    uncertainty_field: Union[str, None] = UNCERTAINTY_FIELD,
) -> Tuple[List[Atoms], List[np.ndarray], List[Optional[np.ndarray]]]:
    """Extract all fields from a configuration output, dispatching on the file extension."""
    suffix = Path(configuration_output_path).suffix
    if suffix in (".dump", ".lammpstrj"):
        return extract_all_fields_from_dump(configuration_output_path, uncertainty_field)
    if suffix == ".cfg":
        return extract_all_fields_from_cfg(configuration_output_path, uncertainty_field)
    raise ValueError(f"Unknown configuration output extension: '{suffix}'.")
