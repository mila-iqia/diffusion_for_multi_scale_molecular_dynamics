from typing import List

import numpy as np
import pymatgen.core


def sort_elements_by_atomic_mass(list_symbols: List[str]) -> List[str]:
    """Sort elements by atomic mass.

    This is useful to define a canonical order for the elements.

    Args:
        list_symbols: element symbols, assumed to be real elements.

    Returns:
        list_sorted_symbols: same symbols, sorted by increasing atomic mass.
    """
    list_masses = []
    for symbol in list_symbols:
        specie = pymatgen.core.Element(symbol)
        list_masses.append(specie.atomic_mass.real)
    indices = np.argsort(list_masses)

    list_sorted_symbols = [list_symbols[idx] for idx in indices]

    return list_sorted_symbols
