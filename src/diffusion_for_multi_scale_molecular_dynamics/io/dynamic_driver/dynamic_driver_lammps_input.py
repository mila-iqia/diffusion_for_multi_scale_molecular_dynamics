"""Building the shared LAMMPS input script for a dynamic-driver (ARTn or MD) run.

This is the counterpart to io/lammps' single-point input builder: it assembles the common backbone every
dynamic driver runs -- read the configuration, define the potential, watch the per-atom uncertainty and
halt (dumping the structure) when it crosses the threshold -- and appends the driver-specific dynamics
tail (the ARTn or MD commands).
"""

from diffusion_for_multi_scale_molecular_dynamics.namespace import \
    UNCERTAIN_DUMP_FILENAME


def build_dynamic_driver_lammps_inputs(
    configuration_file_path: str,
    interaction_commands: str,
    uncertainty_field: str,
    dump_fields: str,
    uncertainty_threshold: str,
    group_block: str,
    mass_block: str,
    elements_string: str,
    dynamics_block: str,
    trajectory_dump_block: str,
) -> str:
    """Return the full LAMMPS input script for a dynamic-driver run, with the dynamics tail appended."""
    return f"""# LAMMPS input script for a dynamic-driver (ARTn or MD) run.

variable threshold equal {uncertainty_threshold}

# ---------- Initialize Simulation ---------------------
clear
units       metal
dimension   3
boundary    p p p
atom_style  atomic

# Prevent LAMMPS from sorting atoms, which might confuse the dynamics driver.
atom_modify sort 0 1


# ---------- Read in the starting configuration ---------------------
read_data {configuration_file_path}

{group_block}

{mass_block}


# ---------- Define Pair Style ---------------------
{interaction_commands}


# ---------- Define Interatomic Potential ---------------------
compute max_unc_all all reduce max {uncertainty_field}
variable max_unc equal c_max_unc_all


neighbor     2.0 bin
neigh_modify delay 10 check yes

# balance atoms per cpu
comm_style tiled
balance 1.1 rcb


#  ----------- Define interruption variable
variable continue_run equal "v_max_unc < v_threshold"


# stop simulation if the threshold is violated
fix extreme_extrapolation all halt 1 v_continue_run != 1


# ----------- OUTPUT DUMP
{trajectory_dump_block}
thermo 1
thermo_style custom step pe temp v_max_unc

dump uncertain_dump_id all custom 1 {UNCERTAIN_DUMP_FILENAME} {dump_fields}
dump_modify uncertain_dump_id element {elements_string}

dump_modify uncertain_dump_id skip v_continue_run


# ---------- Dynamics (driver-specific) ---------------------
{dynamics_block}
"""
