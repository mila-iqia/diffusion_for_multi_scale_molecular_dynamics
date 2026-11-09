from typing import List, Tuple, Union

from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from pymatgen.core import Structure


def sort_symbols_by_atomic_mass(symbols: List[str]) -> List[Tuple[str, float]]:
    """Return the distinct element symbols as (symbol, atomic mass) pairs, sorted by increasing mass.

    This is the canonical element ordering shared by every LAMMPS input (data file, group/mass blocks).

    Args:
        symbols: element symbols (duplicates allowed).

    Returns:
        the (symbol, atomic_mass) pairs of the distinct elements, sorted by increasing atomic mass.
    """
    elements = [(symbol, atomic_masses[atomic_numbers[symbol]]) for symbol in set(symbols)]
    return sorted(elements, key=lambda symbol_and_mass: symbol_and_mass[1])


def sort_structure_elements_by_atomic_mass(structure: Structure) -> List[Tuple[str, float]]:
    """Return the structure's unique elements as (symbol, atomic mass) pairs, sorted by increasing mass."""
    return sort_symbols_by_atomic_mass([element.symbol for element in structure.elements])


def sort_atoms_elements_by_atomic_mass(atoms: Atoms) -> List[Tuple[str, float]]:
    """Return the configuration's unique elements as (symbol, atomic mass) pairs, sorted by increasing mass."""
    return sort_symbols_by_atomic_mass(atoms.get_chemical_symbols())


def generate_named_elements_blocks(configuration: Union[Structure, Atoms]) -> Tuple[str, str, str]:
    """Generate named elements blocks.

    The LAMMPS input file requires the list of the elements present. This creates consistently sorted text
    blocks to identify the group ids, the masses and the symbols of the elements.

    Args:
        configuration: a pymatgen structure or an ase.Atoms configuration.

    Returns:
        group_block: a multiline string, with the group id and element symbol on each line.
        mass_block:   a multiline string, with the group id mass symbol on each line.
        elements_string: a string with the element symbols.
    """
    if isinstance(configuration, Atoms):
        sorted_elements = sort_atoms_elements_by_atomic_mass(configuration)
    else:
        sorted_elements = sort_structure_elements_by_atomic_mass(configuration)

    elements_string = ""
    group_block = ""
    mass_block = ""

    for group_id, (symbol, atomic_mass) in enumerate(sorted_elements, 1):
        group_block += f"\ngroup {symbol} type {group_id}"
        mass_block += f"\nmass {group_id} {atomic_mass}"
        elements_string += f"{symbol} "

    return group_block, mass_block, elements_string.strip()
