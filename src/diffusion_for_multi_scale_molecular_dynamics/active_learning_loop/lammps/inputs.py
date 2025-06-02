from typing import List, Tuple

import numpy as np
from pymatgen.core import Element, Structure


def sort_elements_by_atomic_mass(list_elements: List[Element]) -> List[Element]:
    """Sort elements by atomic mass.

    This is useful to define a canonical order for the elements.

    Args:
        list_elements: Pymatgen elements, assumed to be real elements.

    Returns:
        list_sorted_elements: same elements, sorted by increasing atomic mass.
    """
    list_masses = [element.atomic_mass.real for element in list_elements]
    indices = np.argsort(list_masses)
    list_sorted_elements = [list_elements[idx] for idx in indices]
    return list_sorted_elements


def generate_named_elements_blocks(structure: Structure) -> Tuple[str, str, str]:
    """Generate named elements blocks.

    The LAMMPS input file requires the list of the elements present.
    This method creates consistently sorted text blocks to identify the
    group ids, the masses and the symbols of the elements.

    Args:
        structure: a pymatgen structure object.

    Returns:
        group_block: a multiline string, with the group id and element symbol on each line.
        mass_block:   a multiline string, with the group id mass symbol on each line.
        elements_string: a string with the element symbols.
    """
    sorted_elements = sort_elements_by_atomic_mass(structure.elements)

    elements_string = ""
    group_block = ""
    mass_block = ""

    for group_id, element in enumerate(sorted_elements, 1):
        symbol = element.symbol
        group_block += f"\ngroup {symbol} type {group_id}"
        mass_block += f"\nmass {group_id} {element.atomic_mass.real}"
        elements_string += f"{symbol} "

    return group_block, mass_block, elements_string.strip()
