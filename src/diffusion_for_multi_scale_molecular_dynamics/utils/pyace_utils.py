from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from ase import Atoms
from pyace import (ACEAtomicEnvironment, ACEBBasisSet, ACEBEvaluator,
                   ACECalculator, BBasisConfiguration,
                   aseatoms_to_atomicenvironment)
from pyace.asecalc import PyACECalculator


def count_number_total_atoms_per_species_type(
    atomic_env_list: List[ACEAtomicEnvironment],
) -> Dict[int, int]:
    """Helper function to count total number of atoms of each type in dataset

    Adapted from pyace.

    Args:
        atomic_env_list:  list of ACEAtomicEnvironment

    Returns:
        species_type => number
    """
    n_total_atoms_per_species_type = Counter()
    for ae in atomic_env_list:
        n_total_atoms_per_species_type.update(ae.species_type[: ae.n_atoms_real])
    return n_total_atoms_per_species_type


def convert_to_atomic_env_list(atomic_env_list, pot: ACEBBasisSet) -> np.array:
    """Converts a list of Atoms objects or ACEAtomicEnvironment.

    Adapted from pyace.
    If the input is a pandas Series, it will be converted into a numpy array before processing.
    Ensures compatibility with the specified potential's element mappings and cutoff settings.

    Args:
        atomic_env_list: A list or numpy array of `ASE.Atoms` or
            `ACEAtomicEnvironment` objects. May also be a pandas Series,
            which will be converted internally.
        pot (ACEBBasisSet): An object containing the basis set and element
            mappings to index map, as well as the maximum cutoff distance.

    Returns:
         atomic_env_list: all elements converted into `ACEAtomicEnvironment` objects.
    """
    elements_mapper_dict = pot.elements_to_index_map
    if isinstance(atomic_env_list, pd.Series):
        atomic_env_list = atomic_env_list.values
    if isinstance(atomic_env_list[0], Atoms):
        atomic_env_list = np.array(
            [
                aseatoms_to_atomicenvironment(
                    at, cutoff=pot.cutoffmax, elements_mapper_dict=elements_mapper_dict
                )
                for at in atomic_env_list
            ]
        )
    elif not isinstance(atomic_env_list[0], ACEAtomicEnvironment):
        raise ValueError(
            "atomic_env_list should be list of ASE.Atoms or ACEAtomicEnvironment"
        )
    return atomic_env_list


def compute_number_of_functions(pot: ACEBBasisSet) -> List[int]:
    """
    Computes the number of functions for each pair of basis_rank1 and basis elements.

    This function takes an ACEBBasisSet object as input and calculates the sum of
    the lengths of each corresponding pair of basis_rank1 and basis elements,
    returning a list of these sums.

    Args:
        pot (ACEBBasisSet): An object containing basis_rank1 and basis lists,
            where each pair of elements corresponds to a rank-1 basis and its
            corresponding basis set.

    Returns:
        List[int]: A list of integers representing the sum of the lengths of
        basis_rank1 and basis elements for each respective pair.
    """
    return [len(b1) + len(b) for b1, b in zip(pot.basis_rank1, pot.basis)]


def convert_to_bbasis(
    bconf: Union[BBasisConfiguration, ACEBBasisSet, PyACECalculator]
) -> ACEBBasisSet:
    """Converts an input configuration to an ACEBBasisSet.

    Adapted from pyace.

    Args:
        bconf: The input configuration, which can be of type
            BBasisConfiguration, ACEBBasisSet or PyACECalculator.

    Returns:
        ACEBBasisSet: The ACEBBasisSet constructed from the input configuration.
    """
    if isinstance(bconf, BBasisConfiguration):
        bbasis = ACEBBasisSet(bconf)
    elif isinstance(bconf, ACEBBasisSet):
        bbasis = bconf
    elif isinstance(bconf, PyACECalculator):
        bbasis = bconf.basis
    else:
        raise ValueError("Unsupported type of `bconf`: {}".format(type(bconf)))
    return bbasis


def compute_B_projections(
    bconf: Union[BBasisConfiguration, ACEBBasisSet, PyACECalculator],
    atomic_env_list: Union[List[Atoms], List[ACEAtomicEnvironment]],
    structure_ind_list: Optional[List[int]] = None,
) -> Tuple[Dict[int, np.array], Dict[int, np.array]]:
    """Function to compute the B-basis projection using basis configuration

    Adapted from pyace. Modified to not be dependent on maxvolpy.

    Args:
        bconf: BBasisConfiguration from an ASE potential
        atomic_env_list: list of ACEAtomicEnvironment or ASE atoms
        structure_ind_list (optional): list of corresponding indices of structures/atomic envs. Defaults to None.

    Returns:
        A0_projections_dict:
            dictionary {species_type => B-basis projections}
            B-basis projections shape = [n_atoms[species_type], n_funcs[species_type]]

        structure_ind_dict:
            dictionary {species_type => indices of corresponding structure}
            shape = [n_atoms[species_type]]
    """
    if structure_ind_list is None:
        tmp_structure_ind_list = range(len(atomic_env_list))
    else:
        tmp_structure_ind_list = structure_ind_list

    # create BBasis configuration
    bbasis = convert_to_bbasis(bconf)

    n_projections = compute_number_of_functions(bbasis)

    elements_mapper_dict = bbasis.elements_to_index_map
    atomic_env_list = convert_to_atomic_env_list(atomic_env_list, bbasis)

    # count total number of atoms of each species type in whole dataset atomiv_env_list
    n_total_atoms_per_species_type = count_number_total_atoms_per_species_type(
        atomic_env_list
    )

    beval = ACEBEvaluator(bbasis)

    calc = ACECalculator(beval)

    # prepare numpy arrays for A0_projections and  structure_ind_dict
    A0_projections_dict = {
        st: np.zeros(
            (n_total_atoms_per_species_type[st], n_projections[st]), dtype=np.float64
        )
        for _, st in elements_mapper_dict.items()
    }

    structure_ind_dict = {
        st: np.zeros(n_total_atoms_per_species_type[st], dtype=int)
        for _, st in elements_mapper_dict.items()
    }

    cur_inds = [0] * len(elements_mapper_dict)
    for struct_ind, ae in zip(tmp_structure_ind_list, atomic_env_list):
        calc.compute(ae, compute_projections=True)
        basis_projections = calc.projections

        for atom_ind, atom_type in enumerate(ae.species_type[: ae.n_atoms_real]):
            cur_ind = cur_inds[atom_type]
            structure_ind_dict[atom_type][cur_ind] = struct_ind
            cur_basis_proj = np.reshape(basis_projections[atom_ind], (-1,))
            A0_projections_dict[atom_type][cur_ind] = cur_basis_proj
            cur_inds[atom_type] += 1

    return A0_projections_dict, structure_ind_dict
