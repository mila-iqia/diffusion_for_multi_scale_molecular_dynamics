"""Abinit input writing and output parsing, backed by ASE.

This is the ``io`` layer for the Abinit oracle: it writes a single-configuration input and parses the
resulting text output. It deals only in ase.Atoms and plain numbers (no ``calc``/``mlip`` dependency).
Only the text ``.abo`` output is read for now; reading the netCDF GSR file is a possible future addition.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from ase import Atoms
from ase.io.abinit import read_abinit_out, write_abinit_in
from ase.symbols import symbols2numbers

logger = logging.getLogger(__name__)

ABINIT_INPUT_FILE_NAME = "abinit.abi"
ABINIT_OUTPUT_FILE_NAME = "abinit.abo"


def _pseudopotentials_ordered_by_atomic_number(
    pseudopotentials: Dict[str, Union[str, Path]]
) -> Dict[int, Path]:
    """Map each element's atomic number to its pseudopotential path (Abinit orders species by Z)."""
    elements = list(pseudopotentials.keys())
    atomic_numbers = symbols2numbers(elements)
    return {int(atomic_number): Path(pseudopotentials[element])
            for atomic_number, element in zip(atomic_numbers, elements)}


def read_exchange_correlation_from_pseudopotential(pseudopotential_path: Union[str, Path]) -> int:
    """Read the exchange-correlation code (pspxc) from an Abinit pseudopotential header.

    pspxc is the second integer on the '... pspcod,pspxc,lmax,lloc,mmax,r2well' header line; Abinit uses
    it as 'ixc' when the latter is not given explicitly.

    Raises:
        ValueError: if no 'pspxc' header line is found in the pseudopotential.
    """
    with open(pseudopotential_path) as file_descriptor:
        for line in file_descriptor:
            # The header line lists its values, then names them in a trailing comma-separated token, e.g.:
            #     "8   11   2   4   600   0   pspcod,pspxc,lmax,lloc,mmax,r2well"
            if "pspxc" not in line:
                continue
            tokens = line.split()
            values, variable_names = tokens[:-1], tokens[-1].split(",")
            pspxc_position = variable_names.index("pspxc")
            return int(values[pspxc_position])
    raise ValueError(
        f"Could not find 'pspxc' in the pseudopotential header of {pseudopotential_path}; "
        "provide 'ixc' explicitly in the Abinit parameters instead."
    )


def _exchange_correlation_from_pseudopotentials(pseudopotential_paths: List[Path]) -> int:
    """Read pspxc from each pseudopotential and return it, requiring all to agree (a single functional)."""
    values = {read_exchange_correlation_from_pseudopotential(path) for path in pseudopotential_paths}
    if len(values) > 1:
        raise ValueError(
            f"The pseudopotentials disagree on the exchange-correlation code (pspxc): {sorted(values)}. "
            "Use pseudopotentials generated with the same functional, or set 'ixc' explicitly."
        )
    return values.pop()


def add_exchange_correlation_from_pseudopotentials(
    parameters: Dict, pseudopotential_paths: List[Union[str, Path]]
) -> Dict:
    """Return a copy of the Abinit parameters with 'ixc' set from the pseudopotential headers, if it is missing.

    ixc must be set before ASE writes the input, which would otherwise default to 7 (LDA). A warning is logged
    when ixc is added.

    Args:
        parameters: the Abinit input variables.
        pseudopotential_paths: the pseudopotentials, which must all be generated with the same functional.

    Returns:
        parameters_with_ixc: a copy of parameters, with 'ixc' added if it was not given.
    """
    parameters = dict(parameters)
    if "ixc" not in parameters:
        parameters["ixc"] = _exchange_correlation_from_pseudopotentials([Path(path) for path in pseudopotential_paths])
        logger.warning(
            f"No 'ixc' was given to Abinit; setting ixc={parameters['ixc']} from the pseudopotential "
            "header (pspxc), since it must be set before ASE (which would otherwise default to 7/LDA)."
        )
    return parameters


def write_abinit_input(
    atoms: Atoms,
    parameters: Dict,
    pseudopotentials: Dict[str, Union[str, Path]],
    working_directory: Union[str, Path],
) -> Path:
    """Write 'abinit.abi' for a single configuration, copying its pseudopotentials into the working directory.

    Args:
        atoms: the configuration to compute.
        parameters: the Abinit input variables (e.g. ecut, ixc, ngkpt, ...), following the ASE unit
            convention (eV, Angstrom) rather than Abinit's own Hartree/Bohr.
        pseudopotentials: mapping from element symbol to pseudopotential file path.
        working_directory: directory to write the input (and pseudopotential copies) into; created if missing.

    Returns:
        The path to the written 'abinit.abi'.
    """
    working_directory = Path(working_directory)
    working_directory.mkdir(parents=True, exist_ok=True)

    parameters = dict(parameters)
    # The pseudopotentials are copied next to the input, so any directory hint would be wrong.
    parameters.pop("pspdir", None)
    parameters.pop("pp_dirpath", None)

    pseudopotential_by_atomic_number = _pseudopotentials_ordered_by_atomic_number(pseudopotentials)

    # Abinit expects the species (atomic numbers) sorted, with one pseudopotential per present species.
    present_atomic_numbers = sorted(set(int(number) for number in atoms.numbers))
    present_pseudopotentials = [pseudopotential_by_atomic_number[number] for number in present_atomic_numbers]

    # Default the exchange-correlation to the one the pseudopotentials were generated with.
    parameters = add_exchange_correlation_from_pseudopotentials(parameters, present_pseudopotentials)

    local_pseudopotentials = []
    for source_path in present_pseudopotentials:
        destination_path = working_directory / source_path.name
        if not destination_path.exists():
            shutil.copy(source_path, destination_path)
        local_pseudopotentials.append(source_path.name)

    input_path = working_directory / ABINIT_INPUT_FILE_NAME
    with open(input_path, "w") as file_descriptor:
        write_abinit_in(file_descriptor, atoms, parameters, present_atomic_numbers, local_pseudopotentials)
    return input_path


def read_abinit_output(
    working_directory: Union[str, Path]
) -> Tuple[float, np.ndarray, Optional[np.ndarray]]:
    """Parse the Abinit text output ('abinit.abo') for the total energy, forces and (optional) stress.

    Args:
        working_directory: directory containing the finished run's 'abinit.abo'.

    Returns:
        (energy, forces, stress): energy in eV, per-atom forces in the input atom order, and the stress
        tensor if Abinit computed one (otherwise None).
    """
    output_path = Path(working_directory) / ABINIT_OUTPUT_FILE_NAME
    with open(output_path) as file_descriptor:
        results = read_abinit_out(file_descriptor)

    energy = float(results["energy"])
    forces = np.asarray(results["forces"], dtype=float)
    stress = results.get("stress")
    if stress is not None:
        stress = np.asarray(stress, dtype=float)
    return energy, forces, stress
