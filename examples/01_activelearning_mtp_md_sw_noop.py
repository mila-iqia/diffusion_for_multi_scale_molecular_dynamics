"""Active learning example: MTP MLIP, MD dynamic driver, Stillinger-Weber oracle, no-op sample maker.

This example is self-contained; every create_* function lists its parameters (defaults left explicit) so the
available knobs are visible, and main() is identical across the examples, so switching a component is a
copy-paste of the matching create_* function.

Required packages (beyond the base install):
    - MLIP-3, providing the 'mlp' executable that fits the MTP: https://gitlab.com/ashapeev/mlip-3.git
    - lammps-mtp-kokkos, a LAMMPS with the mtp/extrapolation pair_style (can be built CPU-only):
      https://github.com/RichardZJM/lammps-mtp-kokkos.git

Notes about this example (the chosen options):
    - MLIP: MTP, started cold (a fresh model). Option B in create_mlip loads a pretrained potential instead.
    - Dynamic driver: NVT molecular dynamics.
    - Oracle: Stillinger-Weber.
    - Sample maker: no-op, which labels the uncertain structure itself, without excision or repaint.
"""

import os
from pathlib import Path

from ase.build import bulk

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.active_learning import \
    ActiveLearning
from diffusion_for_multi_scale_molecular_dynamics.dynamic_driver.md_driver.md_driver import \
    MdDriver
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.stillinger_weber import \
    StillingerWeberPotential
from diffusion_for_multi_scale_molecular_dynamics.mlip.mtp.mtp_configuration import \
    MtpConfiguration
from diffusion_for_multi_scale_molecular_dynamics.mlip.mtp.mtp_mlip import \
    MtpMlip
from diffusion_for_multi_scale_molecular_dynamics.mlip.mtp.mtp_trainer import \
    MtpTrainer
from diffusion_for_multi_scale_molecular_dynamics.oracle import \
    SW_COEFFICIENTS_DIR
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_runner import \
    SubprocessLammpsRunner
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_single_point_calculator import \
    LammpsSinglePointCalculator
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.atom_selector.threshold_atom_selector import (
    ThresholdAtomSelector, ThresholdAtomSelectorParameters)
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.no_op_sample_maker import (
    NoOpSampleMaker, NoOpSampleMakerArguments)
from diffusion_for_multi_scale_molecular_dynamics.utils.structure_utils import \
    label_configurations


def get_setting(value, environment_variable, default=None):
    """Get the value set in this file, else the environment variable, else the default (an error without one)."""
    if value is not None:
        return value
    if environment_variable in os.environ:
        return os.environ[environment_variable]
    if default is not None:
        return default
    raise ValueError(f"Set the value in this file or the environment variable {environment_variable}.")


# --- User configuration (set these for your task) ---
ELEMENT_LIST = ["Si"]
UNCERTAINTY_THRESHOLD = 2.0  # MTP extrapolation grade (gamma); atoms above this are treated as uncertain
WORKING_DIRECTORY = Path("run")
STILLINGER_WEBER_COEFFICIENTS_FILE_PATH = SW_COEFFICIENTS_DIR / "Si.sw"
MPI_EXECUTABLE = "mpirun"  # the MPI launcher (srun on some clusters)

# --- Machine configuration (replace None by your value, or set the DFMSMD_* environment variable) ---
# your LAMMPS executable (built with mtp/extrapolation)
LAMMPS_EXECUTABLE_PATH = Path(get_setting(None, "DFMSMD_LAMMPS_PATH"))
# the MLIP-3 'mlp' executable (fits the MTP)
MLP_EXECUTABLE_PATH = Path(get_setting(None, "DFMSMD_MLP_PATH"))


def main():
    """Run one active learning campaign (identical in every example).

    The campaign repeats these steps until the dynamic driver finds no uncertain structure:
        Stage.DRIVER (run_dynamic_driver):
            1. Run the dynamic driver with the MLIP.
            2. Extract the uncertainty per atom.
        Stage.GENERATE (generate_samples):
            3. Generate the structures to label from the uncertain structure, with the sample maker.
        Stage.ORACLE (oracle_evaluation):
            4. Label the structures with the oracle.
            5. Commit the labelled structures to the training database.
        Stage.TRAIN (_retrain):
            6. Fold the labelled structures into the model and retrain the MLIP.

    With restart_from_stage="auto" the campaign starts from Stage.DRIVER on a fresh run, and on a restart
    resumes just after the last stage that completed successfully.
    """
    initial_configuration = create_initial_configuration()  # DynamicDriver initial configuration
    provided_configurations = create_provided_configurations()  # Initial training configuration for the MLIP

    oracle = create_oracle()
    sample_maker = create_sample_maker()
    dynamic_driver = create_dynamic_driver(initial_configuration)
    mlip = create_mlip()

    provided_configurations = label_configurations(provided_configurations, oracle)

    active_learning = ActiveLearning(
        oracle_single_point_calculator=oracle,
        sample_maker=sample_maker,
        dynamic_driver=dynamic_driver,
        mlip=mlip,
    )
    active_learning.run_campaign(
        uncertainty_threshold=UNCERTAINTY_THRESHOLD,
        working_directory=WORKING_DIRECTORY,
        provided_configurations=provided_configurations,
        maximum_number_of_rounds=100,
        restart_from_stage="auto",  # either {"auto", "driver", "generate", "oracle", "train"}
    )


def create_initial_configuration():
    """Create the (orthogonal) starting configuration for the dynamic driver (512 atoms)."""
    silicon = bulk("Si", "diamond", a=5.43, cubic=True)
    silicon_supercell = silicon.repeat((4, 4, 4))
    return silicon_supercell


def create_provided_configurations():
    """Create a training configuration to start simulations (16 atoms)."""
    silicon = bulk("Si", "diamond", a=5.43)
    silicon_supercell = silicon.repeat((2, 2, 2))
    return [silicon_supercell]


def create_mlip():
    """Create the MTP MLIP to drive and refine."""
    lammps_runner = SubprocessLammpsRunner(
        lammps_executable_path=LAMMPS_EXECUTABLE_PATH, mpi_processors=1, openmp_threads=1,
        mpi_executable=MPI_EXECUTABLE,
    )

    # A. Cold start from a fresh MTP model (default).
    mtp_configuration = MtpConfiguration(
        elements=ELEMENT_LIST, level=6, max_dist=5.0,
        energy_weight=1.0, force_weight=0.01, stress_weight=0.0, site_en_weight=0.0,
        training_params=dict(max_iter=1000, init_params="same", scale_by_force=0.0, bfgs_conv_tol=1e-3),
    )
    mtp_trainer = MtpTrainer(mtp_configuration=mtp_configuration, mlp_executable_path=MLP_EXECUTABLE_PATH)
    return MtpMlip(mtp_trainer=mtp_trainer, lammps_runner=lammps_runner)

    # B. Or load a pretrained MTP potential instead of the cold start above (the configuration must be given,
    #    since the .almtp file does not record the training parameters):
    # mtp_trainer = MtpTrainer.load_checkpoint(
    #     Path("initial_mtp/potential.almtp"),
    #     mtp_configuration=MtpConfiguration(elements=ELEMENT_LIST, level=6, max_dist=5.0),
    #     mlp_executable_path=MLP_EXECUTABLE_PATH,
    # )
    # return MtpMlip(mtp_trainer=mtp_trainer, lammps_runner=lammps_runner)


def create_dynamic_driver(initial_configuration):
    """Create a molecular-dynamics dynamic driver."""
    lammps_runner = SubprocessLammpsRunner(
        lammps_executable_path=LAMMPS_EXECUTABLE_PATH, mpi_processors=1, openmp_threads=1,
        mpi_executable=MPI_EXECUTABLE,
    )
    return MdDriver(lammps_runner=lammps_runner, initial_configuration=initial_configuration,
                    temperature=300.0, timestep=0.001, number_of_steps=1000)


def create_oracle():
    """Create a Stillinger-Weber oracle."""
    lammps_runner = SubprocessLammpsRunner(
        lammps_executable_path=LAMMPS_EXECUTABLE_PATH, mpi_processors=1, openmp_threads=1,
        mpi_executable=MPI_EXECUTABLE,
    )
    # Or run LAMMPS in-process (lower overhead); requires the LAMMPS python binding ('import lammps'):
    # from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_runner import InProcessLammpsRunner
    # lammps_runner = InProcessLammpsRunner()
    potential = StillingerWeberPotential(sw_coefficients_file_path=STILLINGER_WEBER_COEFFICIENTS_FILE_PATH)
    return LammpsSinglePointCalculator(lammps_potential=potential, lammps_runner=lammps_runner,
                                       with_uncertainty=False)


def create_sample_maker():
    """Create a no-op sample maker (labels the uncertain structure itself, without excision or repaint)."""
    atom_selector = ThresholdAtomSelector(ThresholdAtomSelectorParameters(
        algorithm="threshold", uncertainty_threshold=UNCERTAINTY_THRESHOLD,
    ))
    arguments = NoOpSampleMakerArguments(
        element_list=ELEMENT_LIST, algorithm="noop", sample_box_strategy="noop", sample_box_size=None,
    )
    return NoOpSampleMaker(sample_maker_arguments=arguments, atom_selector=atom_selector)


if __name__ == "__main__":
    main()
