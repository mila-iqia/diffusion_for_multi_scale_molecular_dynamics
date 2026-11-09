"""Tests for building the shared dynamic-driver LAMMPS input script."""

from diffusion_for_multi_scale_molecular_dynamics.io.dynamic_driver.dynamic_driver_lammps_input import \
    build_dynamic_driver_lammps_inputs


def test_build_dynamic_driver_lammps_inputs():
    """The script wires in the threshold, potential, uncertainty compute and the driver-specific tail."""
    script = build_dynamic_driver_lammps_inputs(
        configuration_file_path="initial_configuration.dat",
        interaction_commands="pair_style lj/cut 6.0",
        uncertainty_field="c_pe_atom",
        dump_fields="id x y z",
        uncertainty_threshold="0.500000000000",
        group_block="group Si type 1",
        mass_block="mass 1 28.0855",
        elements_string="Si",
        dynamics_block="run 5",
        trajectory_dump_block="dump dump_id all custom 1 dump.yaml id x y z",
    )

    assert "$" not in script  # fully built, no leftover placeholders
    assert "variable threshold equal 0.500000000000" in script
    assert "pair_style lj/cut 6.0" in script
    assert "compute max_unc_all all reduce max c_pe_atom" in script
    assert "dump dump_id all custom 1 dump.yaml id x y z" in script
    assert script.rstrip().endswith("run 5")  # the dynamics tail is appended last
