"""Active learning example: MTP MLIP, MD dynamic driver, Abinit oracle, excise-and-repaint sample maker.

This example is self-contained; every create_* function lists its parameters (defaults left explicit) so the
available knobs are visible, and main() is identical across the examples, so switching a component is a
copy-paste of the matching create_* function.

Required packages (beyond the base install):
    - MLIP-3, providing the 'mlp' executable that fits the MTP: https://gitlab.com/ashapeev/mlip-3.git
    - lammps-mtp-kokkos, a LAMMPS with the mtp/extrapolation pair_style (can be built CPU-only):
      https://github.com/RichardZJM/lammps-mtp-kokkos.git
    - Abinit, providing the 'abinit' executable: https://www.abinit.org

Notes about this example (the chosen options):
    - MLIP: MTP, started cold (a fresh model). Option B in create_mlip loads a pretrained potential instead.
    - Dynamic driver: NVT molecular dynamics of a 512-atom silicon supercell.
    - Oracle: Abinit (DFT) on 64-atom cells at the Gamma point only, with 18 MPI tasks (one OpenMP thread each)
      split by Abinit itself (paral_kgb 1, autoparal 1); about 20 s per cell on an 18-core laptop.
      Cheap rather than converged: the goal is to check the pipeline.
    - Sample maker: excise-and-repaint. All atoms within 4.2 Å of each uncertain atom (itself, 4 first and
      12 second neighbors) are kept fixed in a 64-atom cubic box (2a = 10.86 Å, larger than the 8.4 Å sphere),
      and the other atoms are repainted by the diffusion model pretrained on Si64
      (references_files/pretrainedcartesianmodel_Si64_64-4_epoch43.ckpt).
"""

import os
from pathlib import Path

from ase.build import bulk

from diffusion_for_multi_scale_molecular_dynamics import TOP_DIR
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.active_learning import \
    ActiveLearning
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.models.axl_diffusion_lightning_model import \
    AXLDiffusionLightningModel
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.dynamic_driver.md_driver.md_driver import \
    MdDriver
from diffusion_for_multi_scale_molecular_dynamics.mlip.mtp.mtp_configuration import \
    MtpConfiguration
from diffusion_for_multi_scale_molecular_dynamics.mlip.mtp.mtp_mlip import \
    MtpMlip
from diffusion_for_multi_scale_molecular_dynamics.mlip.mtp.mtp_trainer import \
    MtpTrainer
from diffusion_for_multi_scale_molecular_dynamics.oracle.abinit_runner import \
    AbinitRunner
from diffusion_for_multi_scale_molecular_dynamics.oracle.abinit_single_point_calculator import \
    AbinitSinglePointCalculator
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_runner import \
    SubprocessLammpsRunner
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.atom_selector.threshold_atom_selector import (
    ThresholdAtomSelector, ThresholdAtomSelectorParameters)
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.excise_and_repaint_sample_maker import (
    ExciseAndRepaintSampleMaker, ExciseAndRepaintSampleMakerArguments)
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.excisor.spherical_excisor import (
    SphericalExcision, SphericalExcisionArguments)
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.generator.predictor_corrector_axl_generator import \
    PredictorCorrectorSamplingParameters
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
ABINIT_MPI_TASKS = 18  # all the cores of the machine; Abinit decides how to split them (autoparal)
REFERENCES_FILES = TOP_DIR / "examples" / "references_files"
DIFFUSION_MODEL_CHECKPOINT_PATH = REFERENCES_FILES / "pretrainedcartesianmodel_Si64_64-4_epoch43.ckpt"
SILICON_PSEUDOPOTENTIAL_PATH = REFERENCES_FILES / "Si.psp8"  # an Abinit pseudopotential for Si, from PseudoDojo
MPI_EXECUTABLE = "mpirun"  # the MPI launcher (srun on some clusters)
SILICON_LATTICE_CONSTANT = 5.43  # Angstrom
RADIAL_CUTOFF = 4.2  # Angstrom, between the 2nd (3.84 Å) and 3rd (4.50 Å) neighbor shells; MTP and excision

# --- Machine configuration (replace None by your value, or set the DFMSMD_* environment variable) ---
# your LAMMPS executable (built with mtp/extrapolation)
LAMMPS_EXECUTABLE_PATH = Path(get_setting(None, "DFMSMD_LAMMPS_PATH"))
# the MLIP-3 'mlp' executable (fits the MTP)
MLP_EXECUTABLE_PATH = Path(get_setting(None, "DFMSMD_MLP_PATH"))
# one thread per MPI task
ABINIT_COMMAND = get_setting(None, "DFMSMD_ABINIT_COMMAND",
                             default="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 abinit")


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
    silicon = bulk("Si", "diamond", a=SILICON_LATTICE_CONSTANT, cubic=True)
    silicon_supercell = silicon.repeat((4, 4, 4))
    return silicon_supercell


def create_provided_configurations():
    """Create a training configuration to start simulations (64 atoms, the same cell as the oracle samples)."""
    silicon = bulk("Si", "diamond", a=SILICON_LATTICE_CONSTANT, cubic=True)
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
        elements=ELEMENT_LIST, level=8, max_dist=RADIAL_CUTOFF,
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
    """Create an Abinit oracle (parameters in the ASE unit convention: eV, Angstrom)."""
    abinit_runner = AbinitRunner(abinit_command=ABINIT_COMMAND, mpi_runner=MPI_EXECUTABLE,
                                 number_of_mpi_tasks=ABINIT_MPI_TASKS)
    parameters = dict(ecut=550.0, ngkpt=[1, 1, 1], nshiftk=1, shiftk=[0.0, 0.0, 0.0], toldfe=1.0e-6, nstep=100,
                      occopt=3, tsmear=0.1,  # Fermi-Dirac smearing (eV): disordered samples can be metallic
                      paral_kgb=1, autoparal=1,  # bands/FFT parallelism (only one k-point), chosen by Abinit
                      prtwf=0, prtden=0,  # no wavefunction/density files (their MPI-IO write can hang)
                      nsym=1)  # no symmetry: the samples have none, and a perfect Si64 exceeds Abinit's maxnsym
    return AbinitSinglePointCalculator(
        parameters=parameters,
        pseudopotentials={"Si": SILICON_PSEUDOPOTENTIAL_PATH},
        abinit_runner=abinit_runner,
        excluded_output_extensions=["_WFK"],
    )


def create_sample_maker():
    """Create an excise-and-repaint sample maker, which repaints 64-atom boxes with the pretrained diffusion model."""
    atom_selector = ThresholdAtomSelector(ThresholdAtomSelectorParameters(
        algorithm="threshold", uncertainty_threshold=UNCERTAINTY_THRESHOLD,
    ))
    environment_excisor = SphericalExcision(SphericalExcisionArguments(
        algorithm="spherical_cutoff", radial_cutoff=RADIAL_CUTOFF,
    ))
    sample_box_size = [2 * SILICON_LATTICE_CONSTANT] * 3  # Si64
    number_of_samples_per_substructure = 1
    arguments = ExciseAndRepaintSampleMakerArguments(
        element_list=ELEMENT_LIST, algorithm="excise_and_repaint", sample_box_strategy="fixed",
        sample_box_size=sample_box_size, max_constrained_substructure=4,
        number_of_samples_per_substructure=number_of_samples_per_substructure, sample_edit_radius=None,
    )
    # The same noise schedule as the training of the diffusion model; each corrector kick is about 1.26 sigma.
    noise_parameters = NoiseParameters(
        total_time_steps=1000, schedule_type="linear", sigma_min_cart=1.088740e-3, sigma_max_cart=2.177481,
        corrector_step_epsilon=1.6 * 1.088740e-3**2,
    )
    sampling_parameters = PredictorCorrectorSamplingParameters(
        algorithm="predictor_corrector", spatial_dimension=3, num_atom_types=len(ELEMENT_LIST), number_of_atoms=64,
        number_of_samples=number_of_samples_per_substructure, use_fixed_lattice_parameters=True,
        cell_dimensions=sample_box_size, number_of_corrector_steps=2, one_atom_type_transition_per_step=True,
        atom_type_greedy_sampling=True, atom_type_transition_in_corrector=False, progress_bar=False,
    )
    diffusion_model = AXLDiffusionLightningModel.load_from_checkpoint(
        DIFFUSION_MODEL_CHECKPOINT_PATH, map_location="cpu",
    ).axl_network
    diffusion_model.eval()
    return ExciseAndRepaintSampleMaker(
        sample_maker_arguments=arguments,
        atom_selector=atom_selector,
        environment_excisor=environment_excisor,
        noise_parameters=noise_parameters,
        sampling_parameters=sampling_parameters,
        diffusion_model=diffusion_model,
        device="cpu",
    )


if __name__ == "__main__":
    main()
