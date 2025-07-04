from typing import Dict, List, Optional, Union

import numpy as np
from pymatgen.core import Structure


def generate_xyz_text(
    structure: Structure,
    site_properties: Optional[Union[str, List[str]]],
    properties_dim: Optional[Dict[str, int]]
):
    """Generate XYZ text.

    This method transforms a pymatgen structure and attendant
    site properties into text in the xyz format.

    Args:
        structure: pymatgen structure to convert
        site_properties: atomic properties names
        properties_dim: A dictionary defining the dimensionality of each site property.

    Returns:
        xyz_text: a string in the xyz format, ready to be dumped to file.
    """
    lattice = structure.lattice._matrix
    lattice = list(
        map(str, lattice.flatten())
    )  # flatten and convert to string for formatting
    lattice_str = 'Lattice="' + " ".join(lattice) + '" Origin="0 0 0" pbc="T T T"'

    n_atom = len(structure.sites)
    if site_properties is None:
        site_properties = []
        properties_dim = []

    elif site_properties is not None and isinstance(site_properties, str):
        site_properties = [site_properties]
        assert (
            properties_dim is not None
        ), "site properties are defined, but dimensionalities are not."

    if properties_dim is not None:
        properties_dim = [properties_dim[k] for k in site_properties]

    assert len(properties_dim) == len(
        site_properties
    ), "mismatch between number of site properties names and dimensions"

    xyz_txt = f"{n_atom}\n"
    xyz_txt += lattice_str + " Properties=pos:R:3"
    for prop, prop_dim in zip(site_properties, properties_dim):
        xyz_txt += f":{prop}:R:{prop_dim}"
    xyz_txt += "\n"
    for i in range(n_atom):
        positions_values = structure.sites[i].coords
        xyz_txt += " ".join(map(str, positions_values))
        for prop in site_properties:
            prop_value = structure.sites[i].properties.get(prop, 0)
            if not isinstance(prop_value, (list, np.ndarray)):
                prop_value = [prop_value]
            xyz_txt += f" {' '.join(map(str, prop_value))}"
        xyz_txt += "\n"

    return xyz_txt
