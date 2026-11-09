"""ARTn input writing, LAMMPS-tail building, and output parsing.

The ARTn 'artn.in' file (a Fortran ``&ARTN_PARAMETERS`` namelist) and the ARTn LAMMPS commands are
generated here from parameters, so no template file or user-supplied artn.in is needed.
"""

import io
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import ase.io
import yaml
from ase import Atoms

from diffusion_for_multi_scale_molecular_dynamics.io.dynamic_driver.artn_input_configuration import \
    ArtnInputConfiguration
from diffusion_for_multi_scale_molecular_dynamics.io.dynamic_driver.calculation_state import \
    CalculationState

INTERRUPTION_MESSAGE = "Failure message: ARTn RESEARCH STOP BEFORE THE END"
SUCCESS_MESSAGE = r"!> CLEANING ARTn \| Fail: 0"
EIGENVALUE_LOST_MESSAGE = "EIGENVALUE LOST"

# Short (~2 word) descriptions of the ARTn macro and micro stages, for the run-summary log lines.
MACRO_STAGE_DESCRIPTIONS = {
    "Bstep": "Basin stage",
    "Sstep": "Saddle stage",
    "Rstep": "Relaxation stage",
}
MICRO_STAGE_DESCRIPTIONS = {
    "void": "State reset",
    "init": "Initial push",
    "perp": "Perp relaxation",
    "eign": "Eigen climb",
    "lanc": "Lanczos step",
    "relx": "FIRE relaxation",
    "over": "Push over",
    "smth": "Smooth transition",
}

# An ARTn step row starts with the step index followed by a macro stage (e.g. '   9   Sstep/smth   ...').
_ARTN_STEP_ROW_PATTERN = re.compile(r"^\s*\d+\s+(?:Bstep|Sstep|Rstep)\b")
# The failure footer names the micro stage the run was interrupted at, e.g. '... failed ( 1 ) at perp ***'.
_ARTN_FAILURE_PATTERN = re.compile(r"ARTn search failed\s*\(\s*\d+\s*\)\s*at\s+(\w+)")
# The 'Failure message:' line carries the reason the run stopped (e.g. 'EIGENVALUE LOST, try to increase ...').
_ARTN_FAILURE_MESSAGE_PATTERN = re.compile(r"Failure message:\s*(.+)")

# The success footer reports the transition energetics (activation energies, reaction dE) and the saddle's
# participating-atom count; the header echoes the displacement threshold used to count them.
_ARTN_FORWARD_ACTIVATION_PATTERN = re.compile(r"forward\s+E_act\s*=\s*([-+]?[\d.]+)")
_ARTN_BACKWARD_ACTIVATION_PATTERN = re.compile(r"backward\s+E_act\s*=\s*([-+]?[\d.]+)")
_ARTN_REACTION_ENERGY_PATTERN = re.compile(r"reaction\s+dE\s*=\s*([-+]?[\d.]+)")
_ARTN_SADDLE_NPART_PATTERN = re.compile(r"DEBRIEF\(SADDLE\).*?npart\s*=\s*(\d+)")
_ARTN_DISPLACEMENT_THRESHOLD_PATTERN = re.compile(r"delr_thr\s*=\s*([\d.]+)")

# The LAMMPS log updates every energy evaluation (artn.out's evalf only updates once per ARTn step); the
# uncertainty halt fires at, and names, the last evaluated step.
LAMMPS_LOG_FILENAME = "log.lammps"
_LAMMPS_HALT_STEP_PATTERN = re.compile(r"halt condition.*?met on step\s+(\d+)")
_LAMMPS_THERMO_ROW_PATTERN = re.compile(r"^\s*(\d+)\s+-?\d", re.MULTILINE)

# Per-search records the ARTn driver accumulates over a run, read back for the interruption summary.
ACCUMULATED_SUMMARY_FILENAME = "accumulated_summary.yaml"


def _format_namelist_value(value) -> str:
    """Format a Python value as a Fortran namelist value (bool, string, list, or number)."""
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, str):
        return f"'{value}'"
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_namelist_value(item) for item in value)
    return str(value)


def write_artn_input_file(path: Union[str, Path], configuration: ArtnInputConfiguration) -> Path:
    """Write the given ARTn input configuration as an '&ARTN_PARAMETERS' namelist to path."""
    lines = ["&ARTN_PARAMETERS"]
    lines += [f"  {key} = {_format_namelist_value(value)}" for key, value in configuration.to_namelist().items()]
    lines += ["/"]

    path = Path(path)
    path.write_text("\n".join(lines) + "\n")
    return path


def build_artn_lammps_tail(artn_library_plugin_path: Union[str, Path]) -> str:
    """Build the ARTn LAMMPS commands: load the plugin, add the ARTn fix, and run a FIRE minimization."""
    return "\n".join([
        f"plugin load {artn_library_plugin_path}",
        "fix artn_fix_id all artn dmax 5.0",
        "timestep 0.001",
        "reset_timestep 0",
        "min_style fire",
        "minimize 1e-4 1e-5 50000 50000",
    ])


def get_calculation_state_from_artn_output(artn_output: str) -> CalculationState:
    """Get calculation state from ARTn output.

    This method determines if the ARTn calculation was successful or interrupted
    by seeking well defined sub-strings in the file content.

    Args:
        artn_output (str): The content of an artn.out file


    Returns:
        state: the parsed status of the calculation.
    """
    match_success = re.search(SUCCESS_MESSAGE, artn_output)
    match_interruption = re.search(INTERRUPTION_MESSAGE, artn_output)

    if match_success and match_interruption:
        raise ValueError("Both the success and the interruption messages are present in the artn.out file. "
                         "Something is wrong; review code!")

    if not match_success and not match_interruption:
        # The run produced an artn.out that reports neither outcome: treat it as a failed run.
        return CalculationState.ERROR

    if match_interruption:
        return CalculationState.INTERRUPTION
    else:
        return CalculationState.SUCCESS


def get_saddle_energy(artn_output: str):
    """Get saddle energy from ARTn output."""
    saddle_energy_pattern = r"\|> DEBRIEF\(SADDLE\) \| dE = (?P<energy>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?) eV"
    match = re.search(saddle_energy_pattern, artn_output)
    return float(match.group('energy'))


def collect_artn_run_information(working_directory: Union[str, Path]) -> Dict:
    """Collect one ARTn search's summary from its working directory (artn.out + log.lammps), for logging.

    Reads the last step row of artn.out (the ARTn step, the macro/micro stage, the lowest eigenvalue and its
    eigenvector stability a1), the micro stage the failure footer reports on an interruption, and the
    energy-evaluation count from log.lammps (the LAMMPS step; artn.out's evalf only updates once per ARTn step,
    so it undercounts).
    """
    working_directory = Path(working_directory)
    artn_output = (working_directory / "artn.out").read_text()
    lammps_log = (working_directory / LAMMPS_LOG_FILENAME).read_text()

    last_step = _parse_last_artn_step(artn_output)
    interrupted_micro_stage = _parse_interrupted_micro_stage(artn_output)
    return dict(
        artn_step=last_step["artn_step"],
        force_evaluations=_parse_lammps_force_evaluations(lammps_log),
        macro_stage=last_step["macro_stage"],
        micro_stage=interrupted_micro_stage or last_step["micro_stage"],
        lowest_eigenvalue=last_step["lowest_eigenvalue"],
        eigenvector_stability=last_step["eigenvector_stability"],
    )


def _parse_last_artn_step(artn_output: str) -> Dict:
    """Parse the last ARTn step row into its step index, macro/micro stage, eigenvalue, a1 and eval count."""
    step_rows = [line for line in artn_output.splitlines() if _ARTN_STEP_ROW_PATTERN.match(line)]
    if not step_rows:
        raise ValueError("No ARTn step rows were found in the artn.out content.")
    tokens = step_rows[-1].split()

    # The stage is one token when macro/micro share a slash ('Sstep/smth'), two when spaced ('Bstep void').
    if "/" in tokens[1]:
        macro_stage, micro_stage = tokens[1].split("/")
        values = tokens[2:]
    else:
        macro_stage, micro_stage = tokens[1], tokens[2]
        values = tokens[3:]

    # values: Etot, init, eign, perp, lanc, relx, Ftot, Fperp, Fpara, eigval, delr, npart, evalf, a1.
    eigenvalue_token = values[9]  # written as '**********' until the first Lanczos eigenvalue is available.
    lowest_eigenvalue = None if set(eigenvalue_token) == {"*"} else float(eigenvalue_token)
    return dict(
        artn_step=int(tokens[0]),
        macro_stage=macro_stage,
        micro_stage=micro_stage,
        lowest_eigenvalue=lowest_eigenvalue,
        eigenvector_stability=float(values[13]),
    )


def _parse_interrupted_micro_stage(artn_output: str) -> Optional[str]:
    """Return the micro stage named in the ARTn failure footer, or None when there is no failure footer."""
    match = _ARTN_FAILURE_PATTERN.search(artn_output)
    return match.group(1) if match else None


def parse_artn_failure_message(artn_output: str) -> Optional[str]:
    """Return the text of the ARTn 'Failure message:' line, or None when the run reported no failure."""
    match = _ARTN_FAILURE_MESSAGE_PATTERN.search(artn_output)
    return match.group(1).strip() if match else None


def _parse_lammps_force_evaluations(lammps_log: str) -> Optional[int]:
    """Number of energy evaluations from log.lammps: the step the uncertainty halt fired at (or the last one)."""
    halt_match = _LAMMPS_HALT_STEP_PATTERN.search(lammps_log)
    if halt_match:
        return int(halt_match.group(1))
    thermo_steps = _LAMMPS_THERMO_ROW_PATTERN.findall(lammps_log)
    return int(thermo_steps[-1]) if thermo_steps else None


def append_artn_search_summary(working_directory: Union[str, Path], record: Dict) -> None:
    """Append one search's record to the accumulated-summary YAML in working_directory (created if absent)."""
    summary_path = Path(working_directory) / ACCUMULATED_SUMMARY_FILENAME
    summaries = read_artn_search_summaries(working_directory)
    summaries.append(record)
    with open(summary_path, "w") as file_descriptor:
        yaml.safe_dump(summaries, file_descriptor, sort_keys=False)


def read_artn_search_summaries(working_directory: Union[str, Path]) -> List[Dict]:
    """Read the accumulated per-search records written this run, or an empty list when none exist yet."""
    summary_path = Path(working_directory) / ACCUMULATED_SUMMARY_FILENAME
    if not summary_path.is_file():
        return []
    with open(summary_path) as file_descriptor:
        return yaml.safe_load(file_descriptor) or []


def read_artn_xyz(xyz_path: Union[str, Path], specorder: List[str]) -> Atoms:
    """Read an ARTn extended-XYZ configuration, mapping its integer species column to chemical symbols.

    ARTn writes the species column as the LAMMPS atom-type index (1-based), not a chemical symbol; each index
    k is replaced by specorder[k - 1] (the mass-sorted element order the LAMMPS data was written with), and the
    'species:I' property is switched to 'species:S', so ASE reads correct elements.
    """
    lines = Path(xyz_path).read_text().splitlines()
    fixed_lines = [lines[0], lines[1].replace("species:I:1", "species:S:1")]
    for line in lines[2:]:
        if not line.strip():
            continue
        tokens = line.split()
        tokens[0] = specorder[int(tokens[0]) - 1]
        fixed_lines.append(" ".join(tokens))

    atoms = ase.io.read(io.StringIO("\n".join(fixed_lines) + "\n"), format="extxyz")
    atoms.pbc = True
    return atoms


def collect_artn_transition_information(working_directory: Union[str, Path]) -> Dict:
    """Collect the transition energetics of a successful ARTn run from artn.out, for logging.

    Reads the forward/backward activation energies and the reaction energy from the success footer, the
    saddle's number of participating atoms, and the displacement threshold those atoms are counted against.
    """
    artn_output = (Path(working_directory) / "artn.out").read_text()
    return dict(
        forward_activation_energy=float(_ARTN_FORWARD_ACTIVATION_PATTERN.search(artn_output).group(1)),
        backward_activation_energy=float(_ARTN_BACKWARD_ACTIVATION_PATTERN.search(artn_output).group(1)),
        reaction_energy=float(_ARTN_REACTION_ENERGY_PATTERN.search(artn_output).group(1)),
        number_of_participating_atoms=int(_ARTN_SADDLE_NPART_PATTERN.search(artn_output).group(1)),
        displacement_threshold=float(_ARTN_DISPLACEMENT_THRESHOLD_PATTERN.search(artn_output).group(1)),
    )
