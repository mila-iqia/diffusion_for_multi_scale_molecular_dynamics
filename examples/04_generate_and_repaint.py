"""Generate new Si8 cells with a trained diffusion model: from pure noise, and by excise-and-repaint.

This example uses the pretrained model of references_files/pretrainedcartesianmodel_Si64_64-4_epoch43.ckpt, trained on
Si64. To use a model trained with 03_train_diffusion_model.py instead, point model_path in main() to one of its
checkpoints and use its noise schedule in NOISE_PARAMETERS (exponential, 1e-2 to 5 ang).

Notes about this example (the chosen options):
    - Generation: predictor-corrector, from pure noise, in a fixed cubic cell at the density of the reference.
    - Excise-and-repaint: one random atom and its 2 closest neighbors are kept fixed, the other atoms are repainted.
    - Noise: the same schedule as the training of the pretrained model (linear, 1.09e-3 to 2.18 ang).
    - Corrector: corrector_step_epsilon (ang^2) is 1.6 sigma_min^2, so that each corrector kick is about 1.26 sigma.
    - Structure analysis: the fraction of diamond cells, the bond angle deviation and the Stillinger-Weber energy
      of the generated cells are compared with the training data, which gives their expected values.
"""

from pathlib import Path

import ase.io
import numpy as np
import torch
from ase.neighborlist import neighbor_list

from diffusion_for_multi_scale_molecular_dynamics import TOP_DIR
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.data_module.utils import (
    AXL_to_traj, traj_to_AXL)
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.models.axl_diffusion_lightning_model import \
    AXLDiffusionLightningModel
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.stillinger_weber import \
    StillingerWeberPotential
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, AXL_COMPOSITION)
from diffusion_for_multi_scale_molecular_dynamics.oracle import \
    SW_COEFFICIENTS_DIR
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_runner import \
    InProcessLammpsRunner
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_single_point_calculator import \
    LammpsSinglePointCalculator
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.base_sample_maker import \
    BaseExciseSampleMaker
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.excisor.nearest_neighbors_excisor import (
    NearestNeighborsExcision, NearestNeighborsExcisionArguments)
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.generator.constrained_langevin_generator import \
    ConstrainedLangevinGenerator
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.generator.langevin_generator import \
    LangevinGenerator
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.generator.predictor_corrector_axl_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.generator.sampling_constraint import \
    SamplingConstraint
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.sampling.diffusion_sampling import \
    create_batch_of_samples

# --- User configuration (set these for your machine and task) ---
ELEMENT_LIST = ["Si"]
REFERENCE_TRAJECTORY_PATH = TOP_DIR / "examples" / "references_files" / "si8_database.traj"
WORKING_DIRECTORY = Path("generated_si8")
NUMBER_OF_ATOMS = 8
NUMBER_OF_SAMPLES = 16
EXCISION_NUMBER_OF_NEIGHBORS = 2  # excise the pivot atom plus its this many closest neighbors
NOISE_PARAMETERS = NoiseParameters(
    total_time_steps=1000,
    schedule_type="linear",
    sigma_min_cart=1.088740e-3,
    sigma_max_cart=2.177481,
    corrector_step_epsilon=1.6 * 1.088740e-3**2,
)


def main():
    """Generate new cells from pure noise, excise-and-repaint one atom's environment, and analyse the cells."""
    model_path = TOP_DIR / "examples" / "references_files" / "pretrainedcartesianmodel_Si64_64-4_epoch43.ckpt"
    # or a checkpoint of 03_train_diffusion_model.py, e.g. Path("run_si8/models/version_0/checkpoints/epoch=24.ckpt")
    model = load_diffusion_model(model_path)
    cell_dimensions = create_cell_dimensions()
    WORKING_DIRECTORY.mkdir(parents=True, exist_ok=True)

    new_cells = generate_new_cells(model, cell_dimensions)
    ase.io.write(str(WORKING_DIRECTORY / "new_cells.traj"), new_cells)

    repainted_cells = excise_and_repaint(model, cell_dimensions)
    ase.io.write(str(WORKING_DIRECTORY / "repainted_cells.traj"), repainted_cells)

    training_data = ase.io.read(REFERENCE_TRAJECTORY_PATH, index=":")
    analyze_structures({"training data": training_data, "new cells": new_cells, "repainted cells": repainted_cells})


def load_diffusion_model(model_path):
    """Load the trained diffusion model from the checkpoint model_path."""
    model = AXLDiffusionLightningModel.load_from_checkpoint(model_path, map_location="cpu")
    model.eval()
    return model


def create_cell_dimensions():
    """Cubic cell side length for NUMBER_OF_ATOMS, at the same density as REFERENCE_TRAJECTORY_PATH."""
    reference_atoms = ase.io.read(REFERENCE_TRAJECTORY_PATH, index=0)
    reference_side_length = reference_atoms.cell.lengths()[0]
    density_scaling = (NUMBER_OF_ATOMS / len(reference_atoms)) ** (1 / 3)
    return [reference_side_length * density_scaling] * 3


def create_sampling_parameters(cell_dimensions):
    """Predictor-corrector sampling parameters, shared by the generation and the excise-and-repaint."""
    return PredictorCorrectorSamplingParameters(
        algorithm="predictor_corrector",
        spatial_dimension=3,
        num_atom_types=len(ELEMENT_LIST),
        number_of_atoms=NUMBER_OF_ATOMS,
        number_of_samples=NUMBER_OF_SAMPLES,
        sample_batchsize=NUMBER_OF_SAMPLES,
        use_fixed_lattice_parameters=True,
        cell_dimensions=cell_dimensions,
        record_samples=False,
        record_samples_corrector_steps=False,
        record_atom_type_update=False,
        number_of_corrector_steps=2,
        one_atom_type_transition_per_step=True,
        atom_type_greedy_sampling=True,
        atom_type_transition_in_corrector=False,
        progress_bar=True,
    )


def generate_new_cells(model, cell_dimensions):
    """Generate brand-new NUMBER_OF_ATOMS cells, without any constraint."""
    sampling_parameters = create_sampling_parameters(cell_dimensions)
    generator = LangevinGenerator(
        noise_parameters=NOISE_PARAMETERS, sampling_parameters=sampling_parameters, axl_network=model.axl_network,
    )
    print(f"Generating {NUMBER_OF_SAMPLES} cells of {''.join(ELEMENT_LIST)}{NUMBER_OF_ATOMS}.")
    with torch.no_grad():
        samples_batch = create_batch_of_samples(
            generator=generator, sampling_parameters=sampling_parameters, device=model.device,
        )
    return extract_structures(samples_batch)


def excise_and_repaint(model, cell_dimensions):
    """Excise one random atom and its EXCISION_NUMBER_OF_NEIGHBORS closest neighbors, and repaint the other atoms."""
    reference_atoms = ase.io.read(REFERENCE_TRAJECTORY_PATH, index=0)
    structure = traj_to_AXL([reference_atoms], ELEMENT_LIST)[0]

    central_atom_index = np.random.randint(len(reference_atoms))
    excisor = NearestNeighborsExcision(
        NearestNeighborsExcisionArguments(number_of_neighbors=EXCISION_NUMBER_OF_NEIGHBORS),
    )
    substructures, _ = excisor.excise_environments(structure, np.array([central_atom_index]), center_atoms=True)

    new_box_lattice_parameters = np.array(cell_dimensions + [0., 0., 0.])
    substructure_in_new_box = BaseExciseSampleMaker.embed_structure_in_new_box(
        substructures[0], new_box_lattice_parameters,
    )

    sampling_constraint = SamplingConstraint(
        elements=ELEMENT_LIST,
        constrained_relative_coordinates=torch.FloatTensor(substructure_in_new_box.X),
        constrained_atom_types=torch.LongTensor(substructure_in_new_box.A),
        constrained_indices=torch.arange(len(substructure_in_new_box.X)),
    )
    sampling_parameters = create_sampling_parameters(cell_dimensions)
    generator = ConstrainedLangevinGenerator(
        noise_parameters=NOISE_PARAMETERS, sampling_parameters=sampling_parameters,
        axl_network=model.axl_network, sampling_constraints=sampling_constraint,
    )
    print(f"Repainting {NUMBER_OF_SAMPLES} cells of {''.join(ELEMENT_LIST)}{NUMBER_OF_ATOMS} around "
          f"{len(substructure_in_new_box.X)} excised atoms.")
    with torch.no_grad():
        samples_batch = create_batch_of_samples(
            generator=generator, sampling_parameters=sampling_parameters, device=model.device,
        )
    return extract_structures(samples_batch)


def extract_structures(samples_batch):
    """Extract every sample of a create_batch_of_samples() batch as a list of ase.Atoms objects."""
    axl_batch = samples_batch[AXL_COMPOSITION]
    axl_list = [
        AXL(A=A.cpu().numpy(), X=X.cpu().numpy(), L=L.cpu().numpy())
        for A, X, L in zip(axl_batch.A, axl_batch.X, axl_batch.L)
    ]
    return AXL_to_traj(axl_list, ELEMENT_LIST)


# --- STRUCTURE ANALYSIS ---
BOND_CUTOFF = 2.6  # ang; crystalline Si has exactly 4 neighbors below this, the second shell is at ~3.8 ang
TETRAHEDRAL_ANGLE = np.degrees(np.arccos(-1.0 / 3.0))  # 109.47 degrees
STILLINGER_WEBER_COEFFICIENTS_FILE_PATH = SW_COEFFICIENTS_DIR / "Si.sw"


def analyze_structures(sets_of_cells):
    """Print the fraction of diamond cells, the bond angle deviation and the Stillinger-Weber energy of each set.

    The training data gives the expected values of good generated cells.
    """
    print(f"{'':16s} {'diamond cells':>13s} {'angle deviation (deg)':>21s} {'SW energy (eV/atom)':>22s}")
    for name, cells in sets_of_cells.items():
        energies = compute_stillinger_weber_energies_per_atom(cells)
        print(f"{name:16s} {compute_diamond_cell_fraction(cells):13.3f} {compute_mean_angle_deviation(cells):21.2f} "
              f"{energies.mean():12.4f} +- {energies.std():.4f}")


def compute_diamond_cell_fraction(cells):
    """Fraction of the cells in which every atom has exactly 4 neighbors closer than BOND_CUTOFF."""
    is_diamond = [
        np.all(np.bincount(neighbor_list("i", atoms, BOND_CUTOFF), minlength=len(atoms)) == 4) for atoms in cells
    ]
    return np.mean(is_diamond)


def compute_mean_angle_deviation(cells):
    """Mean of |bond angle - 109.47| over every pair of bonds (closer than BOND_CUTOFF) of every atom, in degrees."""
    deviations = []
    for atoms in cells:
        atom_indices, bond_vectors = neighbor_list("iD", atoms, BOND_CUTOFF)
        unit_bond_vectors = bond_vectors / np.linalg.norm(bond_vectors, axis=1, keepdims=True)
        for atom_index in range(len(atoms)):
            atom_bonds = unit_bond_vectors[atom_indices == atom_index]
            cosines = (atom_bonds @ atom_bonds.T)[np.triu_indices(len(atom_bonds), k=1)]
            deviations.append(np.abs(np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0))) - TETRAHEDRAL_ANGLE))
    return np.concatenate(deviations).mean()


def compute_stillinger_weber_energies_per_atom(cells):
    """Stillinger-Weber energy per atom of every cell, in eV/atom, computed with LAMMPS."""
    potential = StillingerWeberPotential(sw_coefficients_file_path=STILLINGER_WEBER_COEFFICIENTS_FILE_PATH)
    # Requires the LAMMPS python binding ('import lammps'). Otherwise, use a SubprocessLammpsRunner, as in
    # 01_activelearning_mtp_md_sw_noop.py.
    calculator = LammpsSinglePointCalculator(lammps_potential=potential, lammps_runner=InProcessLammpsRunner())
    return np.array([result.energy / len(result.atoms) for result in calculator.calculate_many(cells)])


if __name__ == "__main__":
    main()
