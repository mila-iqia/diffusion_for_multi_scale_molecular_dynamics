import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.units import kB

from diffusion_for_multi_scale_molecular_dynamics.utils.interatomic_force_constants import (
    compute_effective_stiffness_by_element, compute_force_constants,
    compute_thermal_width_by_element)

SPRING_CONSTANT = 2.0  # eV/Angstrom^2


class HarmonicPairCalculator(Calculator):
    """Two atoms joined by an isotropic spring around their reference positions."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, reference_positions, spring_constant):
        super().__init__()
        self.reference_positions = np.array(reference_positions)
        self.spring_constant = spring_constant

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        displacements = self.atoms.positions - self.reference_positions
        relative_displacement = displacements[0] - displacements[1]
        forces = np.zeros_like(displacements)
        forces[0] = -self.spring_constant * relative_displacement
        forces[1] = self.spring_constant * relative_displacement
        self.results = dict(energy=0.5 * self.spring_constant * relative_displacement @ relative_displacement,
                            forces=forces)


@pytest.fixture()
def reference_positions():
    return [[0.0, 0.0, 0.0], [2.35, 0.0, 0.0]]


def make_pair(symbols, positions, reference_positions, spring_constant=SPRING_CONSTANT):
    atoms = Atoms(symbols, positions=positions, cell=20.0 * np.eye(3), pbc=False)
    atoms.calc = HarmonicPairCalculator(reference_positions, spring_constant)
    return atoms


def test_force_constants_of_a_spring(reference_positions):
    atoms = make_pair("Si2", reference_positions, reference_positions)
    identity = np.eye(3)
    expected = SPRING_CONSTANT * np.block([[identity, -identity], [-identity, identity]])
    np.testing.assert_allclose(compute_force_constants(atoms), expected, atol=1e-10)


def test_effective_stiffness_of_a_spring(reference_positions):
    """Without the rigid translation, each atom moves half of the relative coordinate: K = 4 k for both atoms."""
    atoms = make_pair("SiGe", reference_positions, reference_positions)
    effective_stiffness_by_element = compute_effective_stiffness_by_element(atoms)
    assert list(effective_stiffness_by_element) == ["Ge", "Si"]
    for effective_stiffness in effective_stiffness_by_element.values():
        np.testing.assert_allclose(effective_stiffness, 4.0 * SPRING_CONSTANT * np.eye(3), atol=1e-8)


def test_thermal_width_of_a_spring(reference_positions):
    """Each atom moves half of the relative coordinate, whose variance is k_B T / k: s^2 = k_B T / (4 k)."""
    atoms = make_pair("SiGe", reference_positions, reference_positions)
    thermal_width_by_element = compute_thermal_width_by_element(atoms, temperature=300.0)
    expected_thermal_width = np.sqrt(kB * 300.0 / (4.0 * SPRING_CONSTANT))
    for thermal_width in thermal_width_by_element.values():
        assert thermal_width == pytest.approx(expected_thermal_width, rel=1e-6)


def test_unrelaxed_structure_raises(reference_positions):
    stretched_positions = [[0.0, 0.0, 0.0], [2.45, 0.0, 0.0]]
    atoms = make_pair("Si2", stretched_positions, reference_positions)
    with pytest.raises(ValueError, match="not relaxed"):
        compute_effective_stiffness_by_element(atoms)


def test_structure_at_a_maximum_raises(reference_positions):
    atoms = make_pair("Si2", reference_positions, reference_positions, spring_constant=-SPRING_CONSTANT)
    with pytest.raises(ValueError, match="not at a minimum"):
        compute_effective_stiffness_by_element(atoms)


def test_list_of_one_structure_gives_the_same_result(reference_positions):
    atoms = make_pair("SiGe", reference_positions, reference_positions)
    effective_stiffness_by_element = compute_effective_stiffness_by_element(atoms)
    effective_stiffness_by_element_from_list = compute_effective_stiffness_by_element([atoms])
    assert effective_stiffness_by_element_from_list.keys() == effective_stiffness_by_element.keys()
    for element, effective_stiffness in effective_stiffness_by_element.items():
        np.testing.assert_allclose(effective_stiffness_by_element_from_list[element], effective_stiffness)


def test_elements_of_different_structures(reference_positions):
    """Each element gets the stiffness of the structure that contains it: K = 4 k for both atoms of a spring."""
    silicon_pair = make_pair("Si2", reference_positions, reference_positions, spring_constant=2.0)
    germanium_pair = make_pair("Ge2", reference_positions, reference_positions, spring_constant=3.0)
    effective_stiffness_by_element = compute_effective_stiffness_by_element([silicon_pair, germanium_pair])
    assert list(effective_stiffness_by_element) == ["Ge", "Si"]
    np.testing.assert_allclose(effective_stiffness_by_element["Si"], 4.0 * 2.0 * np.eye(3), atol=1e-8)
    np.testing.assert_allclose(effective_stiffness_by_element["Ge"], 4.0 * 3.0 * np.eye(3), atol=1e-8)


def test_same_element_in_several_structures_is_averaged(reference_positions):
    """(Phi^+)_ii = 1 / (4 k) is averaged over the atoms of both structures, then inverted."""
    soft_pair = make_pair("Si2", reference_positions, reference_positions, spring_constant=2.0)
    stiff_pair = make_pair("Si2", reference_positions, reference_positions, spring_constant=3.0)
    mean_thermal_displacement = 0.5 * (1.0 / (4.0 * 2.0) + 1.0 / (4.0 * 3.0))

    effective_stiffness_by_element = compute_effective_stiffness_by_element([soft_pair, stiff_pair])
    np.testing.assert_allclose(effective_stiffness_by_element["Si"], np.eye(3) / mean_thermal_displacement, atol=1e-8)

    thermal_width_by_element = compute_thermal_width_by_element([soft_pair, stiff_pair], temperature=300.0)
    assert thermal_width_by_element["Si"] == pytest.approx(np.sqrt(kB * 300.0 * mean_thermal_displacement), rel=1e-6)


def test_error_gives_the_index_of_the_structure(reference_positions):
    relaxed_pair = make_pair("Si2", reference_positions, reference_positions)
    stretched_pair = make_pair("Si2", [[0.0, 0.0, 0.0], [2.45, 0.0, 0.0]], reference_positions)
    with pytest.raises(ValueError, match="Structure 1: The structure is not relaxed"):
        compute_effective_stiffness_by_element([relaxed_pair, stretched_pair])


def test_empty_list_raises():
    with pytest.raises(ValueError, match="No structure"):
        compute_effective_stiffness_by_element([])


@pytest.mark.parametrize("calculator", [None, "single_point_calculator"])
def test_structure_without_a_real_calculator_raises(reference_positions, calculator):
    atoms = Atoms("Si2", positions=reference_positions, cell=20.0 * np.eye(3), pbc=False)
    if calculator == "single_point_calculator":
        atoms.calc = SinglePointCalculator(atoms, energy=0.0, forces=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="SinglePointCalculator"):
        compute_effective_stiffness_by_element(atoms)
