"""Normalize configuration objects between pymatgen Structures and ase.Atoms."""

from typing import Union

from ase import Atoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor


def to_pymatgen_structure(configuration: Union[Structure, Atoms]) -> Structure:
    """Convert a configuration (pymatgen Structure or ase.Atoms) to a pymatgen Structure."""
    if isinstance(configuration, Structure):
        return configuration
    if isinstance(configuration, Atoms):
        return AseAtomsAdaptor.get_structure(configuration)
    raise TypeError(
        f"Cannot convert a configuration of type '{type(configuration).__name__}' to a Structure; "
        "expected a pymatgen Structure or ase.Atoms."
    )


def to_ase_atoms(configuration: Union[Structure, Atoms]) -> Atoms:
    """Convert a configuration (pymatgen Structure or ase.Atoms) to an ase.Atoms."""
    if isinstance(configuration, Atoms):
        return configuration
    if isinstance(configuration, Structure):
        return configuration.to_ase_atoms()
    raise TypeError(
        f"Cannot convert a configuration of type '{type(configuration).__name__}' to an ase.Atoms; "
        "expected a pymatgen Structure or ase.Atoms."
    )
