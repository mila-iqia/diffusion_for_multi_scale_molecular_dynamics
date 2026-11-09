"""Serialization helpers for MLIP training datasets (gracemaker .pkl.gz, MLIP-3 .cfg)."""

from pathlib import Path
from typing import List, Optional


def write_grace_pkl(atoms_list: List, path: Path, origins: Optional[List[str]] = None) -> Path:
    """Serialize labelled frames to the gzip-pickled pandas DataFrame gracemaker reads.

    Each atoms object must carry its energy and forces (e.g. through an attached calculator). gracemaker
    (via input.yaml ``data.filename``) loads the DataFrame with ``pd.read_pickle(path, compression="gzip")``.

    Args:
        atoms_list: labelled ase.Atoms objects, each exposing get_potential_energy()/get_forces().
        path: destination .pkl.gz path.
        origins: optional provenance strings (one per frame), stored in the 'source_file' column.

    Returns:
        The written path.
    """
    import numpy as np
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, atoms in enumerate(atoms_list):
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)
        number_of_atoms = len(atoms)
        rows.append({
            "ase_atoms": atoms,
            "energy": energy,
            "forces": forces,
            "NUMBER_OF_ATOMS": int(number_of_atoms),
            "natoms_tag": f"N{number_of_atoms}",
            "source_file": origins[index] if origins is not None else "",
            "energy_corrected": energy,
        })
    columns = ["ase_atoms", "energy", "forces", "NUMBER_OF_ATOMS", "natoms_tag", "source_file", "energy_corrected"]
    pd.DataFrame(rows, columns=columns).to_pickle(path, compression="gzip")
    return path


def write_mtp_cfg(training_pool: List, elements: List[str], path: Path) -> Path:
    """Write an MLIP-3 .cfg training file from a maml cfg pool and the element ordering.

    Args:
        training_pool: a maml cfg pool (e.g. from maml.utils.pool_from).
        elements: the element symbols, in the order used to index species in the .cfg.
        path: destination .cfg path (interpreted relative to the current working directory by maml).

    Returns:
        The written path.
    """
    from maml.apps.pes import MTPotential

    configuration_writer = MTPotential()
    configuration_writer.elements = list(elements)
    configuration_writer.write_cfg(str(path), cfg_pool=training_pool)
    return Path(path)
