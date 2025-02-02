"""Namespace.

This module defines string constants to represent recurring concepts that appear
throughout the code base. Confusion and errors are reduced by having one and only one string to
represent these concepts.
"""

from collections import namedtuple

#  r^alpha <-  cartesian position, alpha \in (x,y,z)
# x_i <- relative coordinates i \in (1,2,3)
#
#   r = \sum_{i} x_i a_i, where { a_i } are the basis vectors defining the lattice.

CARTESIAN_POSITIONS = "cartesian_positions"  # position in real cartesian space
RELATIVE_COORDINATES = "relative_coordinates"  # coordinates in the unit cell basis
CARTESIAN_FORCES = "cartesian_forces"

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

LATTICE_PARAMETERS = "lattice_parameters"
NOISY_LATTICE_PARAMETERS = "noisy_lattice_parameters"

AXL = namedtuple("AXL", ["A", "X", "L"])
AXL_NAME_DICT = {"A": ATOM_TYPES, "X": RELATIVE_COORDINATES, "L": UNIT_CELL}

NOISY_AXL_COMPOSITION = "noisy_axl"
AXL_COMPOSITION = "original_axl"

TIME_INDICES = "time_indices"

Q_MATRICES = 'q_matrices'
Q_BAR_MATRICES = 'q_bar_matrices'
Q_BAR_TM1_MATRICES = 'q_bar_tm1_matrices'
