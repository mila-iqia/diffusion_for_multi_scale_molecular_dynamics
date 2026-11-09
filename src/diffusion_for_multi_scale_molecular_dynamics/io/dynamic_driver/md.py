"""Building the MD (NVT) LAMMPS commands for the molecular-dynamics dynamic driver."""

_DEFAULT_VELOCITY_SEED = 12345


def build_md_lammps_tail(
    temperature: float,
    timestep: float,
    number_of_steps: int,
    velocity_seed: int = _DEFAULT_VELOCITY_SEED,
) -> str:
    """Build the NVT MD LAMMPS commands: draw Gaussian velocities, add a Nose-Hoover thermostat, and run.

    Args:
        temperature: NVT thermostat temperature (K).
        timestep: MD timestep (ps, LAMMPS 'metal' units).
        number_of_steps: number of MD steps to run (if the uncertainty stays below the threshold).
        velocity_seed: seed for the initial Gaussian velocity draw.
    """
    thermostat_damping = 100.0 * timestep
    return "\n".join([
        f"velocity all create {temperature} {velocity_seed} dist gaussian mom yes rot yes",
        f"fix nvt_fix_id all nvt temp {temperature} {temperature} {thermostat_damping}",
        f"timestep {timestep}",
        "reset_timestep 0",
        f"run {number_of_steps}",
    ])
