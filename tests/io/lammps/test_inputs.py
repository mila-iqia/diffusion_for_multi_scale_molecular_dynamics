import numpy as np
import pytest
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers

from diffusion_for_multi_scale_molecular_dynamics.io.lammps.inputs import (
    generate_named_elements_blocks, sort_atoms_elements_by_atomic_mass,
    sort_structure_elements_by_atomic_mass, sort_symbols_by_atomic_mass)
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.stillinger_weber import \
    StillingerWeberPotential
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.single_point_calc_lammps_input import \
    LammpsInputBuilder
from diffusion_for_multi_scale_molecular_dynamics.oracle import \
    SW_COEFFICIENTS_DIR


@pytest.fixture()
def expected_group_block(expected_sorted_list_element_symbols):
    number_of_elements = len(expected_sorted_list_element_symbols)
    lines = ""
    for group_id in np.arange(1, number_of_elements + 1):
        symbol = expected_sorted_list_element_symbols[group_id - 1]
        lines += f"\ngroup {symbol} type {group_id}"

    return lines


@pytest.fixture()
def expected_mass_block(expected_sorted_list_element_symbols):
    number_of_elements = len(expected_sorted_list_element_symbols)
    lines = ""
    for group_id in np.arange(1, number_of_elements + 1):
        symbol = expected_sorted_list_element_symbols[group_id - 1]
        mass = atomic_masses[atomic_numbers[symbol]]
        lines += f"\nmass {group_id} {mass}"
    return lines


@pytest.fixture()
def expected_elements_string(expected_sorted_list_element_symbols):
    return " ".join(expected_sorted_list_element_symbols)


def test_sort_symbols_by_atomic_mass(list_element_symbols, expected_sorted_list_element_symbols):
    sorted_elements = sort_symbols_by_atomic_mass(list_element_symbols)
    assert [symbol for symbol, _ in sorted_elements] == expected_sorted_list_element_symbols


def test_sort_structure_elements_by_atomic_mass(structure, expected_sorted_list_element_symbols):
    sorted_elements = sort_structure_elements_by_atomic_mass(structure)
    assert [symbol for symbol, _ in sorted_elements] == expected_sorted_list_element_symbols


def test_sort_atoms_elements_by_atomic_mass(list_element_symbols, expected_sorted_list_element_symbols):
    sorted_elements = sort_atoms_elements_by_atomic_mass(Atoms(list_element_symbols))
    assert [symbol for symbol, _ in sorted_elements] == expected_sorted_list_element_symbols


@pytest.fixture(params=["structure", "atoms"])
def configuration(request, structure, list_element_symbols):
    """The same elements as either a pymatgen structure or an ase.Atoms, so both dispatch paths are tested."""
    if request.param == "structure":
        return structure
    return Atoms(list_element_symbols)


def test_generate_named_elements_blocks(
    configuration, expected_group_block, expected_mass_block, expected_elements_string
):
    group_block, mass_block, elements_string = generate_named_elements_blocks(configuration)
    assert group_block == expected_group_block
    assert mass_block == expected_mass_block
    assert elements_string == expected_elements_string


def test_build_looping_single_point(structure):
    """A looping input holds one self-contained block per structure, separated by 'clear'."""
    potential = StillingerWeberPotential(sw_coefficients_file_path=SW_COEFFICIENTS_DIR / "Si.sw")
    structures = [structure, structure, structure]
    configuration_filenames = [f"configuration_{index}.dat" for index in range(len(structures))]
    dump_filenames = [f"dump_{index}.dump" for index in range(len(structures))]
    energy_filenames = [f"energy_{index}.dat" for index in range(len(structures))]

    input_script = LammpsInputBuilder().build_looping_single_point(
        structures, potential, configuration_filenames, dump_filenames, energy_filenames
    )

    # One 'clear' separates each pair of blocks; one read_data / run 0 / dump / energy file per structure.
    assert input_script.count("clear") == len(structures) - 1
    assert input_script.count("read_data") == len(structures)
    assert input_script.count("run 0") == len(structures)
    for configuration_filename, dump_filename, energy_filename in zip(
        configuration_filenames, dump_filenames, energy_filenames
    ):
        assert f"read_data {configuration_filename}" in input_script
        assert dump_filename in input_script
        assert energy_filename in input_script
