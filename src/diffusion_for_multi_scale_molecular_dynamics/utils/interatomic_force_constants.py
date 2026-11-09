"""Interatomic force constants (IFC) and the effective stiffness of each element in the harmonic approximation.

In a harmonic crystal, the classical thermal displacements of atom i follow <u_i u_i^T> = k_B T (Phi^+)_ii, where Phi
is the IFC matrix and Phi^+ its pseudo-inverse without the rigid translations. The effective stiffness of atom i,
K_i = [(Phi^+)_ii]^-1, is the 3x3 spring that reproduces these displacements, with the motion of every other atom
taken into account. The thermal width of an element at temperature T is then s^2 = k_B T tr(K^-1) / 3.
"""

from typing import Dict, List, Union

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.units import kB


def compute_thermal_width_by_element(
    structures: Union[Atoms, List[Atoms]], temperature: float, displacement: float = 0.01
) -> Dict[str, float]:
    """Compute the harmonic thermal width s of each element of relaxed structures, s^2 = k_B T tr(K^-1) / 3.

    Args:
        structures: a relaxed structure or a list of relaxed structures, each with a calculator able to compute forces
            (not a SinglePointCalculator).
        temperature: temperature, in Kelvin.
        displacement: displacement of each atom in the finite differences, in Angstrom.

    Returns:
        thermal_width_by_element: thermal width of each element, in Angstrom.
    """
    effective_stiffness_by_element = compute_effective_stiffness_by_element(structures, displacement)
    return {
        element: float(np.sqrt(kB * temperature * np.trace(np.linalg.inv(effective_stiffness)) / 3.0))
        for element, effective_stiffness in effective_stiffness_by_element.items()
    }


def compute_effective_stiffness_by_element(
    structures: Union[Atoms, List[Atoms]], displacement: float = 0.01, maximum_residual_force: float = 1.0e-5
) -> Dict[str, np.ndarray]:
    """Compute the effective stiffness K of each element of relaxed structures.

    The IFC matrix Phi of each structure is computed by central finite differences of the forces, and the effective
    stiffness of an element is the inverse of (Phi^+)_ii averaged over the atoms i of this element, in every structure.

    Args:
        structures: a relaxed structure or a list of relaxed structures, each with a calculator able to compute forces
            (not a SinglePointCalculator).
        displacement: displacement of each atom in the finite differences, in Angstrom.
        maximum_residual_force: largest atomic force norm of a relaxed structure, in eV/Angstrom.

    Returns:
        effective_stiffness_by_element: 3x3 effective stiffness of each element, in eV/Angstrom^2.

    Raises:
        ValueError: if no structure is given, or if a structure has no calculator able to compute forces, is not
            relaxed, or is not at a minimum of the energy. The message gives the index of that structure.
    """
    if isinstance(structures, Atoms):
        structures = [structures]
    if len(structures) == 0:
        raise ValueError("No structure is given.")

    symbols, thermal_displacement_matrices = [], []
    for index, atoms in enumerate(structures):
        try:
            thermal_displacement_matrices.append(
                _get_relaxed_thermal_displacement_matrices(atoms, displacement, maximum_residual_force)
            )
        except ValueError as error:
            raise ValueError(f"Structure {index}: {error}") from error
        symbols += atoms.get_chemical_symbols()

    thermal_displacement_matrices = np.concatenate(thermal_displacement_matrices)
    return {
        element: np.linalg.inv(thermal_displacement_matrices[np.array(symbols) == element].mean(axis=0))
        for element in sorted(set(symbols))
    }


def _get_relaxed_thermal_displacement_matrices(
    atoms: Atoms, displacement: float, maximum_residual_force: float
) -> np.ndarray:
    """Get (Phi^+)_ii of each atom of a relaxed structure.

    Args:
        atoms: a relaxed structure, with a calculator able to compute forces (not a SinglePointCalculator).
        displacement: displacement of each atom in the finite differences, in Angstrom.
        maximum_residual_force: largest atomic force norm of a relaxed structure, in eV/Angstrom.

    Returns:
        thermal_displacement_matrices: (Phi^+)_ii of each atom, in Angstrom^2/eV. Dimension [number_of_atoms, 3, 3].

    Raises:
        ValueError: if the structure has no calculator able to compute forces, is not relaxed, or is not at a
            minimum of the energy.
    """
    if atoms.calc is None or isinstance(atoms.calc, SinglePointCalculator):
        raise ValueError("The structure needs a calculator able to compute forces, not a SinglePointCalculator.")

    residual_force = np.linalg.norm(atoms.get_forces(), axis=1).max()
    if residual_force > maximum_residual_force:
        raise ValueError(f"The structure is not relaxed: the largest atomic force is {residual_force:.2e} eV/A, "
                         f"above {maximum_residual_force:.0e} eV/A.")

    return _get_thermal_displacement_matrices(compute_force_constants(atoms, displacement))


def compute_force_constants(atoms: Atoms, displacement: float = 0.01) -> np.ndarray:
    """Compute the IFC matrix Phi_(ia, jb) = -dF_jb / du_ia by central finite differences, symmetrized.

    Args:
        atoms: the structure, with a calculator able to compute forces.
        displacement: displacement of each atom, in Angstrom.

    Returns:
        force_constants: the IFC matrix, in eV/Angstrom^2. Dimension [3 number_of_atoms, 3 number_of_atoms].
    """
    displaced_atoms = atoms.copy()
    displaced_atoms.calc = atoms.calc
    number_of_atoms = len(atoms)
    force_constants = np.zeros((3 * number_of_atoms, 3 * number_of_atoms))
    for atom_index in range(number_of_atoms):
        for direction in range(3):
            displaced_atoms.positions[atom_index, direction] += displacement
            forces_plus = displaced_atoms.get_forces().reshape(-1)
            displaced_atoms.positions[atom_index, direction] -= 2.0 * displacement
            forces_minus = displaced_atoms.get_forces().reshape(-1)
            displaced_atoms.positions[atom_index, direction] += displacement
            force_constants[3 * atom_index + direction] = -(forces_plus - forces_minus) / (2.0 * displacement)
    return 0.5 * (force_constants + force_constants.T)


def _get_thermal_displacement_matrices(force_constants: np.ndarray) -> np.ndarray:
    """Diagonal 3x3 blocks (Phi^+)_ii of the pseudo-inverse of the IFC matrix, without the rigid translations.

    With T the orthonormal rigid translations and P = 1 - T T^T, the translations are the null space of P Phi P, so
    that (P Phi P)^+ = (P Phi P + T T^T)^-1 - T T^T.

    Args:
        force_constants: the IFC matrix, in eV/Angstrom^2. Dimension [3 number_of_atoms, 3 number_of_atoms].

    Returns:
        thermal_displacement_matrices: (Phi^+)_ii of each atom, in Angstrom^2/eV. Dimension [number_of_atoms, 3, 3].

    Raises:
        ValueError: if the structure is not at a minimum of the energy (a mode of negative stiffness).
    """
    number_of_atoms = force_constants.shape[0] // 3
    translations = np.tile(np.eye(3), (number_of_atoms, 1)) / np.sqrt(number_of_atoms)
    translation_projector = translations @ translations.T
    projector = np.eye(3 * number_of_atoms) - translation_projector
    projected_force_constants = projector @ force_constants @ projector

    regularized_force_constants = projected_force_constants + translation_projector
    if np.linalg.eigvalsh(regularized_force_constants).min() <= 0.0:
        raise ValueError("The structure is not at a minimum of the energy: a vibration mode has a negative stiffness.")
    pseudo_inverse = np.linalg.inv(regularized_force_constants) - translation_projector

    return np.stack([pseudo_inverse[3 * i:3 * i + 3, 3 * i:3 * i + 3] for i in range(number_of_atoms)])
