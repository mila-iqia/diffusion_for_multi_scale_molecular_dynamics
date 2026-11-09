"""Train an EGNN diffusion model on crystalline Si8, with target score matching at TEMPERATURE.

A pretrained model is provided in references_files/pretrainedcartesianmodel_Si64_64-4_epoch43.ckpt, used by
04_generate_and_repaint.py and 05_activelearning_mtp_md_abinit_repaint.py. It is a Cartesian EGNN (64 hidden
dimensions, 4 layers, 7.5 Angstrom radial cutoff) trained on Si64 (epoch 43) with a linear noise schedule from
1.09e-3 to 2.18 Angstrom, without target score matching or normalize="smooth".

main() trains a new model. Uncomment another call in main() to:
    - train_from_checkpoint: resume the training from a checkpoint, with a new patience and learning rate.
    - active_train: train one epoch at a time. After each epoch, generate NUMBER_OF_GENERATED_STRUCTURE_PER_EPOCH
      cells, label the bad ones with Stillinger-Weber and append them to bad_frames.traj, which the next epoch reads.
      The rounds stop when every generated cell is good, or after NUMBER_OF_ROUNDS rounds. Running the script again
      resumes from the last checkpoint in ACTIVE_TRAINING_DIRECTORY; keep bad_frames.traj with it, since it is part of
      the training state.

Target score matching needs the thermal width of each element. It is computed from the interatomic force constants
of the first training frame, relaxed with Stillinger-Weber, and written to thermal_widths.json.

Required packages (beyond the base install):
    - The LAMMPS python binding ('import lammps'), for Stillinger-Weber through ase's LAMMPSlib.

Notes about this example (the chosen options):
    - Score network: Cartesian EGNN with a 5 Angstrom radial cutoff and a smooth cutoff envelope. normalize="smooth"
      keeps the direction between two close atoms at nearly full strength down to about 0.1 Angstrom. The bond
      direction edge feature gives the orientation of each bond relative to the cell axes.
    - Data: an 80/20 train/validation split of references_files/si8_database.traj (2 000 frames of Si8). The
      training frames are repeated TRAINING_DATA_REPETITIONS times per epoch, each time with a new noise.
    - Noise: exponential schedule from 1e-2 to 5 Angstrom. The same schedule must be used for sampling.
    - Target: target score matching, with the forces of the training frames at TEMPERATURE.
    - Bad configuration (active_train): a cell that is not diamond, i.e. not every atom has exactly 4 neighbors
      closer than 2.6 Angstrom (see is_bad_configuration).
"""

import json
import multiprocessing
import warnings
from pathlib import Path

import ase.io
import numpy as np
import torch
from ase.calculators.lammpslib import LAMMPSlib
from ase.calculators.singlepoint import SinglePointCalculator
from ase.neighborlist import neighbor_list
from ase.optimize import BFGS
from lightning import Trainer
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.loggers import TensorBoardLogger

from diffusion_for_multi_scale_molecular_dynamics import TOP_DIR
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.callbacks.checkpoint_restart import \
    restart_from_checkpoint
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.callbacks.epoch_summary_callback import \
    EpochSummaryLogger
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.data_module.diffusion.ase_for_diffusion_data_module import (  # noqa
    ASEForDiffusionDataModule, ASEForDiffusionDataModuleParameters)
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.data_module.utils import \
    AXL_to_traj
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.loss.loss_parameters import (
    AtomTypeLossParameters, MSELossParameters)
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.models.axl_diffusion_lightning_model import (
    AXLDiffusionLightningModel, AXLDiffusionParameters)
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.models.optimizer import \
    OptimizerParameters
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.models.scheduler import \
    ReduceLROnPlateauSchedulerParameters
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.models.target_score_matching import \
    TargetScoreMatchingParameters
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, AXL_COMPOSITION)
from diffusion_for_multi_scale_molecular_dynamics.oracle import \
    SW_COEFFICIENTS_DIR
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.generator.langevin_generator import \
    LangevinGenerator
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.generator.predictor_corrector_axl_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.sampling.diffusion_sampling import \
    create_batch_of_samples
from diffusion_for_multi_scale_molecular_dynamics.score_network.cartesian_egnn_score_network import \
    CartesianEGNNScoreNetworkParameters
from diffusion_for_multi_scale_molecular_dynamics.utils.interatomic_force_constants import \
    compute_thermal_width_by_element

# Silence noisy warnings from packages.
warnings.filterwarnings("ignore", message=r".*isinstance\(treespec, LeafSpec\).*")
warnings.filterwarnings("ignore", message=r".*persistent_workers.*")
warnings.filterwarnings("ignore", message=r".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")

# --- User configuration (set these for your task) ---
ELEMENT_LIST = ["Si"]
NUMBER_OF_ATOMS_PER_FRAME = 8  # the (max) number of atoms per frame in TRAJECTORY_FILE_PATH
HIDDEN_DIMENSIONS_SIZE = 128
N_LAYERS = 4
WORKING_DIRECTORY = Path("run_si8")
TRAJECTORY_FILE_PATH = TOP_DIR / "examples" / "references_files" / "si8_database.traj"
TRAINING_DATA_REPETITIONS = 10  # each epoch goes through the training frames this many times, with new noise
TEMPERATURE = 300.0  # K, the temperature of the MD that produced the training frames
STILLINGER_WEBER_COEFFICIENTS_FILE_PATH = SW_COEFFICIENTS_DIR / "Si.sw"

# --- Active training configuration (active_train only) ---
ACTIVE_TRAINING_DIRECTORY = WORKING_DIRECTORY / "active_training"
NUMBER_OF_ROUNDS = 100  # at most this many epochs, each followed by a generation
NUMBER_OF_GENERATED_STRUCTURE_PER_EPOCH = 8
BAD_FRAMES_PATH = ACTIVE_TRAINING_DIRECTORY / "bad_frames.traj"
SUMMARY_PATH = ACTIVE_TRAINING_DIRECTORY / "summary.txt"
CHECKPOINTS_DIRECTORY = ACTIVE_TRAINING_DIRECTORY / "models" / "checkpoints"


def main():
    """Train a new diffusion model."""
    # num_workers must be parallelized through fork instead of spawn to avoid crashing on Mac.
    multiprocessing.set_start_method("fork", force=True)
    WORKING_DIRECTORY.mkdir(parents=True, exist_ok=True)
    train_new_model()
    # train_from_checkpoint()
    # active_train()


def train_new_model():
    """Train a new diffusion model."""
    data_module = create_data_module()
    model = create_diffusion_model()
    trainer = create_trainer()
    trainer.fit(model, datamodule=data_module)
    data_module.clean_up()  # delete the HF datasets working cache now that training is done


def train_from_checkpoint():
    """Resume training from a previous checkpoint, with a new patience and learning rate."""
    # Here, we chose epoch=9.
    checkpoint_path = WORKING_DIRECTORY / "models" / "version_0" / "checkpoints" / "epoch=9.ckpt"
    new_patience = 100
    new_learning_rate = 5e-5

    data_module = create_data_module()
    model = create_diffusion_model()
    trainer = create_trainer()
    restart_from_checkpoint(trainer, new_patience, new_learning_rate)
    trainer.fit(model, datamodule=data_module, ckpt_path=str(checkpoint_path))


def active_train():
    """Train, generate, and add the bad generated cells to bad_frames.traj, until every generated cell is good."""
    ACTIVE_TRAINING_DIRECTORY.mkdir(parents=True, exist_ok=True)
    first_round = get_number_of_trained_epochs()
    if first_round >= NUMBER_OF_ROUNDS:
        print(f"{first_round} epochs are already trained: increase NUMBER_OF_ROUNDS to train more.")
        return

    reference_frame = ase.io.read(TRAJECTORY_FILE_PATH, index=0)
    data_module = create_data_module()
    model = create_diffusion_model(bad_frames_path=BAD_FRAMES_PATH)

    for round_index in range(first_round, NUMBER_OF_ROUNDS):
        train_one_more_epoch(model, data_module, round_index)
        generated_cells = generate_cells(model, reference_frame)
        append_summary_line(round_index, generated_cells)
        bad_cells = [atoms for atoms in generated_cells if is_bad_configuration(atoms)]
        print(f"Round {round_index}: {len(bad_cells)}/{len(generated_cells)} generated cells are bad.")
        if not bad_cells:
            break
        append_bad_frames(label_with_stillinger_weber(bad_cells))

    data_module.clean_up()  # delete the HF datasets working cache now that training is done


def create_data_module():
    """Create the data module: an 80/20 train/validation split of TRAJECTORY_FILE_PATH."""
    train_trajectory_path, validation_trajectory_path = create_train_validation_split()
    hyper_params = ASEForDiffusionDataModuleParameters(
        data_source="ase_trajectory",
        elements=ELEMENT_LIST,
        batch_size=64,
        num_workers=4,
        max_atom=NUMBER_OF_ATOMS_PER_FRAME,
        use_fixed_lattice_parameters=True,
        noise_parameters=create_noise_parameters(),
    )
    return ASEForDiffusionDataModule(
        processed_dataset_dir=str(WORKING_DIRECTORY / "processed_data"),
        hyper_params=hyper_params,
        train_trajectory_list=[str(train_trajectory_path)],
        validation_trajectory_list=[str(validation_trajectory_path)],
        working_cache_dir=str(WORKING_DIRECTORY / "cache"),
    )


def create_train_validation_split():
    """Write an 80/20 train/validation split of TRAJECTORY_FILE_PATH, the training frames repeated."""
    frames = ase.io.read(TRAJECTORY_FILE_PATH, index=":")
    split_index = int(0.8 * len(frames))

    data_directory = WORKING_DIRECTORY / "data"
    data_directory.mkdir(parents=True, exist_ok=True)
    train_trajectory_path = data_directory / "train_conf.traj"
    validation_trajectory_path = data_directory / "validation_conf.traj"
    ase.io.write(str(train_trajectory_path), frames[:split_index] * TRAINING_DATA_REPETITIONS)
    ase.io.write(str(validation_trajectory_path), frames[split_index:])
    return train_trajectory_path, validation_trajectory_path


def create_noise_parameters():
    """Create the noise schedule shared by training and sampling."""
    return NoiseParameters(
        total_time_steps=1000, schedule_type="exponential", sigma_min_cart=1e-2, sigma_max_cart=5.,
        corrector_step_epsilon=1.6e-4,
    )


def create_stillinger_weber_calculator():
    """Create a Stillinger-Weber ase calculator, run in-process by LAMMPS."""
    return LAMMPSlib(
        lmpcmds=["pair_style sw", f"pair_coeff * * {STILLINGER_WEBER_COEFFICIENTS_FILE_PATH} {' '.join(ELEMENT_LIST)}"],
        log_file=None, keep_alive=True,
    )


def compute_thermal_width():
    """Thermal width of each element at TEMPERATURE, from the IFC of the first frame relaxed with Stillinger-Weber.

    The thermal widths are written to thermal_widths.json.
    """
    relaxed_frame = ase.io.read(TRAJECTORY_FILE_PATH, index=0)
    relaxed_frame.calc = create_stillinger_weber_calculator()
    BFGS(relaxed_frame, logfile=None).run(fmax=1e-6)

    thermal_width = compute_thermal_width_by_element(relaxed_frame, TEMPERATURE)
    (WORKING_DIRECTORY / "thermal_widths.json").write_text(json.dumps(thermal_width, indent=2))
    print(f"Thermal widths at {TEMPERATURE} K (Angstrom): {thermal_width}")
    return thermal_width


def create_diffusion_model(bad_frames_path=None):
    """Create the EGNN-based AXL diffusion model, trained with target score matching (and the bad frames if given)."""
    score_network_parameters = CartesianEGNNScoreNetworkParameters(
        num_atom_types=len(ELEMENT_LIST), n_layers=N_LAYERS,
        coordinate_hidden_dimensions_size=HIDDEN_DIMENSIONS_SIZE, coordinate_n_hidden_dimensions=N_LAYERS,
        message_hidden_dimensions_size=HIDDEN_DIMENSIONS_SIZE, message_n_hidden_dimensions=N_LAYERS,
        node_hidden_dimensions_size=HIDDEN_DIMENSIONS_SIZE, node_n_hidden_dimensions=N_LAYERS,
        radial_cutoff=5., smooth_cutoff=True, normalize="smooth", bond_direction_feature=True,
    )
    loss_parameters = AXL(
        A=AtomTypeLossParameters(lambda_weight=0.0),
        X=MSELossParameters(lambda_weight=1.0),
        L=MSELossParameters(lambda_weight=0.0),
    )
    thermal_width = compute_thermal_width()
    target_score_matching_parameters = TargetScoreMatchingParameters(
        temperature=TEMPERATURE,
        thermal_width=thermal_width,
        minimum_sigma=create_noise_parameters().sigma_min_cart,
        maximum_sigma=max(thermal_width.values()),
        bad_frames_path=bad_frames_path,
        maximum_force=5.0,
        bad_frames_lambda_weight=1.0,
    )
    optimizer_parameters = OptimizerParameters(name="adamw", learning_rate=1.0e-4, weight_decay=5.0e-8)
    scheduler_parameters = ReduceLROnPlateauSchedulerParameters(factor=0.9, patience=4)

    diffusion_parameters = AXLDiffusionParameters(
        score_network_parameters=score_network_parameters,
        loss_parameters=loss_parameters,
        optimizer_parameters=optimizer_parameters,
        scheduler_parameters=scheduler_parameters,
        elements=ELEMENT_LIST,
        target_score_matching_parameters=target_score_matching_parameters,
    )
    return AXLDiffusionLightningModel(diffusion_parameters)


def create_trainer():
    """Create the trainer, with checkpointing, early stopping, TensorBoard logging and a per-epoch summary log."""
    models_directory = str(WORKING_DIRECTORY / "models")

    checkpoint_callback = ModelCheckpoint(
        monitor="validation_epoch_loss", mode="min",
        save_top_k=-1, every_n_epochs=1, filename="{epoch}",
    )
    early_stopping_callback = EarlyStopping(monitor="validation_epoch_loss", mode="min", patience=25)
    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")
    epoch_summary_callback = EpochSummaryLogger(early_stopping_callback, WORKING_DIRECTORY / "training.log")
    logger = TensorBoardLogger(save_dir=models_directory, name="")

    return Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor_callback, epoch_summary_callback],
        logger=logger,
        max_epochs=25,
        log_every_n_steps=1,
    )


# --- ACTIVE TRAINING ---
def get_last_checkpoint_path():
    """Get the checkpoint of the last trained epoch, epoch=N.ckpt with the highest N (None without one)."""
    checkpoint_paths = [path for path in CHECKPOINTS_DIRECTORY.glob("epoch=*.ckpt")
                        if path.stem.removeprefix("epoch=").isdigit()]
    return max(checkpoint_paths, key=lambda path: int(path.stem.removeprefix("epoch=")), default=None)


def get_number_of_trained_epochs():
    """Get the number of epochs already trained, from the name of the last checkpoint (0 without one)."""
    last_checkpoint_path = get_last_checkpoint_path()
    return 0 if last_checkpoint_path is None else int(last_checkpoint_path.stem.removeprefix("epoch=")) + 1


def train_one_more_epoch(model, data_module, round_index):
    """Train one more epoch, resuming from the last checkpoint if there is one."""
    models_directory = str(ACTIVE_TRAINING_DIRECTORY / "models")
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(CHECKPOINTS_DIRECTORY), monitor="validation_epoch_loss", mode="min",
        save_top_k=-1, every_n_epochs=1, filename="{epoch}", save_last="link", enable_version_counter=False,
    )
    early_stopping_callback = EarlyStopping(monitor="validation_epoch_loss", mode="min", patience=25)
    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")
    epoch_summary_callback = EpochSummaryLogger(early_stopping_callback, ACTIVE_TRAINING_DIRECTORY / "training.log")
    trainer = Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor_callback, epoch_summary_callback],
        logger=TensorBoardLogger(save_dir=models_directory, name="", version=0),
        max_epochs=round_index + 1,
        log_every_n_steps=1,
    )
    last_checkpoint_path = get_last_checkpoint_path()
    trainer.fit(model, datamodule=data_module,
                ckpt_path=None if last_checkpoint_path is None else str(last_checkpoint_path))


def generate_cells(model, reference_frame):
    """Generate NUMBER_OF_GENERATED_STRUCTURE_PER_EPOCH cells, with the cell and atoms of the reference frame."""
    sampling_parameters = PredictorCorrectorSamplingParameters(
        algorithm="predictor_corrector",
        spatial_dimension=3,
        num_atom_types=len(ELEMENT_LIST),
        number_of_atoms=len(reference_frame),
        number_of_samples=NUMBER_OF_GENERATED_STRUCTURE_PER_EPOCH,
        sample_batchsize=NUMBER_OF_GENERATED_STRUCTURE_PER_EPOCH,
        use_fixed_lattice_parameters=True,
        cell_dimensions=list(reference_frame.cell.lengths()),
        number_of_corrector_steps=2,
        progress_bar=True,
    )
    generator = LangevinGenerator(
        noise_parameters=create_noise_parameters(), sampling_parameters=sampling_parameters,
        axl_network=model.axl_network,
    )
    print(f"Generating {NUMBER_OF_GENERATED_STRUCTURE_PER_EPOCH} cells of {reference_frame.get_chemical_formula()}.")
    model.eval()
    with torch.no_grad():
        samples_batch = create_batch_of_samples(
            generator=generator, sampling_parameters=sampling_parameters, device=model.device,
        )
    axl_batch = samples_batch[AXL_COMPOSITION]
    axl_list = [
        AXL(A=A.cpu().numpy(), X=X.cpu().numpy(), L=L.cpu().numpy())
        for A, X, L in zip(axl_batch.A, axl_batch.X, axl_batch.L)
    ]
    return AXL_to_traj(axl_list, ELEMENT_LIST)


def is_bad_configuration(atoms):
    """Check if a configuration is not diamond: not every atom has exactly 4 neighbors closer than 2.6 Angstrom."""
    coordination_numbers = np.bincount(neighbor_list("i", atoms, 2.6), minlength=len(atoms))
    return not np.all(coordination_numbers == 4)


def label_with_stillinger_weber(cells):
    """Store the Stillinger-Weber energy and forces of each cell in a SinglePointCalculator."""
    calculator = create_stillinger_weber_calculator()
    for atoms in cells:
        atoms.calc = calculator
        energy, forces = atoms.get_potential_energy(), atoms.get_forces()
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    return cells


def append_bad_frames(cells):
    """Append the bad cells to BAD_FRAMES_PATH, read again by the model at the start of the next epoch."""
    with ase.io.Trajectory(str(BAD_FRAMES_PATH), "a") as trajectory:
        for atoms in cells:
            trajectory.write(atoms)


# --- STRUCTURE SUMMARY ---
BOND_CUTOFF = 2.6  # Angstrom
OVERLAP_DISTANCE = 2.0  # Angstrom
TETRAHEDRAL_ANGLE = np.degrees(np.arccos(-1.0 / 3.0))


def append_summary_line(epoch, cells):
    """Append the structure metrics of the cells generated after this epoch to SUMMARY_PATH.

    diamond: fraction of the cells in which every atom has exactly 4 neighbors closer than BOND_CUTOFF.
    <2A: fraction of the atoms whose closest neighbor is closer than OVERLAP_DISTANCE.
    closest: closest pair of atoms, in Angstrom.
    4-coord: fraction of the atoms with exactly 4 neighbors closer than BOND_CUTOFF.
    angle: mean |bond angle - 109.47| over every pair of bonds of every atom, in degrees.
    """
    if not SUMMARY_PATH.exists():
        header = f"{'epoch':>5s} {'diamond':>7s} {'<2A':>6s} {'closest':>7s} {'4-coord':>7s} {'angle':>6s}"
        SUMMARY_PATH.write_text(header + "\n")

    closest_distances, coordination_numbers, angle_deviations = [], [], []
    for atoms in cells:
        distances = atoms.get_all_distances(mic=True)
        np.fill_diagonal(distances, np.inf)
        closest_distances.append(distances.min(axis=1))

        atom_indices, bond_vectors = neighbor_list("iD", atoms, BOND_CUTOFF)
        coordination_numbers.append(np.bincount(atom_indices, minlength=len(atoms)))
        unit_bond_vectors = bond_vectors / np.linalg.norm(bond_vectors, axis=1, keepdims=True)
        for atom_index in range(len(atoms)):
            atom_bonds = unit_bond_vectors[atom_indices == atom_index]
            cosines = (atom_bonds @ atom_bonds.T)[np.triu_indices(len(atom_bonds), k=1)]
            angle_deviations.append(np.abs(np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0))) - TETRAHEDRAL_ANGLE))

    closest_distances = np.concatenate(closest_distances)
    diamond_fraction = np.mean([np.all(coordination == 4) for coordination in coordination_numbers])
    four_coordinated_fraction = np.mean(np.concatenate(coordination_numbers) == 4)
    angle_deviations = np.concatenate(angle_deviations)
    mean_angle_deviation = angle_deviations.mean() if angle_deviations.size else float("nan")

    with open(SUMMARY_PATH, "a") as summary_file:
        summary_file.write(f"{epoch:5d} {diamond_fraction:7.3f} {np.mean(closest_distances < OVERLAP_DISTANCE):6.3f} "
                           f"{closest_distances.min():7.3f} {four_coordinated_fraction:7.3f} "
                           f"{mean_angle_deviation:6.2f}\n")


if __name__ == "__main__":
    main()
