"""General input/output helpers, centred on the ASE .traj format used to move structures around."""

from pathlib import Path
from typing import List


def write_atoms_trajectory(atoms_list: List, path: Path) -> Path:
    """Write a list of ase.Atoms to an ASE .traj file.

    Whatever each atoms object carries is preserved: an attached calculator's energy/forces, its
    per-atom arrays (e.g. an 'uncertainty' array) and its info dict (e.g. 'active_environment_indices').

    Args:
        atoms_list: the ase.Atoms objects to write.
        path: destination .traj path.

    Returns:
        The written path.
    """
    from ase.io.trajectory import Trajectory

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with Trajectory(str(path), mode="w") as trajectory:
        for atoms in atoms_list:
            trajectory.write(atoms)
    return path


def read_atoms_trajectory(path: Path) -> List:
    """Read every frame of an ASE .traj file back into a list of ase.Atoms.

    Args:
        path: the .traj file to read.

    Returns:
        The ase.Atoms frames; each labelled frame exposes get_potential_energy()/get_forces() through the
        SinglePointCalculator ASE reconstructs, and keeps its info dict and per-atom arrays.
    """
    from ase.io.trajectory import Trajectory

    with Trajectory(str(path), mode="r") as trajectory:
        return [atoms for atoms in trajectory]
