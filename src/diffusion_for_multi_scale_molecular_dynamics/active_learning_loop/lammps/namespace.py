# We define standard names used by  LAMMPS to avoid having "magic" strings in the code.

ID_FIELD = "id"  # the atom id
ELEMENT_FIELD = "element"
POSITIONS_FIELDS = ["x", "y", "z"]  # the atomic cartesian positions
FORCES_FIELDS = ["fx", "fy", "fz"]  # the atomic forces
BOX_FIELD = "box"
ENERGY_FIELD = "PotEng"
UNCERTAINTY_FIELD = "c_unc"  # the prefix "c_" is a LAMMPS idiosyncrasy.
