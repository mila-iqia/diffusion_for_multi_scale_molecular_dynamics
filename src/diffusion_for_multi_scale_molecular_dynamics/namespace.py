"""Namespace.

This module defines string constants to represent recurring concepts that appear
throughout the code base. Confusion and errors are reduced by having one and only one string to
represent these concepts.
"""

from collections import namedtuple
from pathlib import Path

#  r^alpha <-  cartesian position, alpha \in (x,y,z)
# x_i <- relative coordinates i \in (1,2,3)
#
#   r = \sum_{i} x_i a_i, where { a_i } are the basis vectors defining the lattice.

CARTESIAN_POSITIONS = "cartesian_positions"  # position in real cartesian space
RELATIVE_COORDINATES = "relative_coordinates"  # coordinates in the unit cell basis
CARTESIAN_FORCES = "cartesian_forces"

NUMBER_OF_ATOMS = "natom"  # Number of atoms in each sample
NOISY_RELATIVE_COORDINATES = (
    "noisy_relative_coordinates"  # relative coordinates perturbed by diffusion noise
)
NOISY_CARTESIAN_POSITIONS = (
    "noisy_cartesian_positions"  # cartesian positions perturbed by diffusion noise
)
TIME = "time"  # diffusion time
NOISE = "noise_parameter"  # the exploding variance sigma parameter
UNIT_CELL = "unit_cell"  # unit cell definition

ATOM_TYPES = "atom_types"
NOISY_ATOM_TYPES = "noisy_atom_types"
PADDED_ATOM_TYPE = -1

LATTICE_PARAMETERS = "lattice_parameters"
NOISY_LATTICE_PARAMETERS = "noisy_lattice_parameters"

AXL = namedtuple("AXL", ["A", "X", "L"])
AXL_NAME_DICT = {"A": ATOM_TYPES, "X": RELATIVE_COORDINATES, "L": LATTICE_PARAMETERS}

NOISY_AXL_COMPOSITION = "noisy_axl"
AXL_COMPOSITION = "original_axl"

TIME_INDICES = "time_indices"

Q_MATRICES = 'q_matrices'
Q_BAR_MATRICES = 'q_bar_matrices'
Q_BAR_TM1_MATRICES = 'q_bar_tm1_matrices'

# ---- LAMMPS input/output filenames (shared by the oracle and the dynamic drivers) ----
LAMMPS_INPUT_FILENAME = "lammps.in"  # the LAMMPS input script
CONFIGURATION_FILENAME = "configuration.dat"  # the single-point structure data file
INITIAL_CONFIGURATION_FILENAME = "initial_configuration.dat"  # a dynamic driver's starting structure
DUMP_FILENAME = "dump.dump"  # the LAMMPS per-atom text dump (read back with ase)
UNCERTAIN_DUMP_FILENAME = "uncertain.dump"  # a dynamic driver's uncertain-structure dump
ORACLE_EVALUATION_PREFIX = "eval"  # oracle evaluation N is 'eval_N' (an Abinit run directory, or 'eval_N.dump')
ENERGY_FILENAME = "energy.dat"  # the total potential energy written by a single-point run
UNCERTAINTY_FIELD = "c_unc_at"  # the per-atom uncertainty dump column (the "c_" prefix is a LAMMPS idiosyncrasy)


def numbered_filename(filename: str, index: int) -> str:
    """Insert an index before the extension, for example 'dump.dump' with index 3 becomes 'dump_3.dump'."""
    path = Path(filename)
    return f"{path.stem}_{index}{path.suffix}"


# ---- Training-database layout (shared by everyone; the database roots at the working directory) ----
PROVIDED_CONFIGURATIONS_FILENAME = "provided_configurations.traj"  # the user-provided seed configurations
PRECOMPUTATION_DIRECTORY_NAME = "precomputation"  # holds the precomputation model + first_training.traj
FIRST_TRAINING_FILENAME = "first_training.traj"  # the labelled first training set (provided + augmented)
