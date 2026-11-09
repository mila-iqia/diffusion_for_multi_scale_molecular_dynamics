"""Tests for building the MD (NVT) LAMMPS commands."""

from diffusion_for_multi_scale_molecular_dynamics.io.dynamic_driver.md import \
    build_md_lammps_tail


def test_build_md_lammps_tail():
    """The NVT tail carries the temperature, timestep, step count and a damping of 100x the timestep."""
    tail = build_md_lammps_tail(temperature=300.0, timestep=0.001, number_of_steps=5, velocity_seed=42)

    assert "velocity all create 300.0 42 dist gaussian" in tail
    assert "fix nvt_fix_id all nvt temp 300.0 300.0 0.1" in tail  # damping = 100 * 0.001
    assert "timestep 0.001" in tail
    assert "run 5" in tail
