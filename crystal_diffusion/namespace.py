"""Namespace.

This module defines string constants to represent recurring concepts that appear
throughout the code base. Confusion and errors are reduced by having one and only one string to
represent these concepts.
"""

#  r^alpha <-  cartesian position, alpha \in (x,y,z)
# x_i <- relative coordinates i \in (1,2,3)
#
#   r = \sum_{i} x_i a_i, where { a_i } are the basis vectors defining the lattice.

CARTESIAN_POSITIONS = "cartesian_positions"   # position in real cartesian space
RELATIVE_COORDINATES = "relative_coordinates"   # coordinates in the unit cell basis

NOISY_RELATIVE_COORDINATES = "noisy_relative_coordinates"   # relative coordinates perturbed by diffusion noise
TIME = "time"  # diffusion time
