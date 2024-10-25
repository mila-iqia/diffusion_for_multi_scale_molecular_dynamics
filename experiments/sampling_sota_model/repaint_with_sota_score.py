import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from diffusion_for_multi_scale_molecular_dynamics import DATA_DIR
from diffusion_for_multi_scale_molecular_dynamics.analysis import (
    PLEASANT_FIG_SIZE, PLOT_STYLE_PATH)
from diffusion_for_multi_scale_molecular_dynamics.generators.constrained_langevin_generator import (
    ConstrainedLangevinGenerator, ConstrainedLangevinGeneratorParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.instantiate_diffusion_model import \
    load_diffusion_model
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.variance_sampler import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps import \
    get_energy_and_forces_from_lammps
from diffusion_for_multi_scale_molecular_dynamics.utils.logging_utils import \
    setup_analysis_logger

logger = logging.getLogger(__name__)
setup_analysis_logger()


experiments_dir = Path("/home/mila/r/rousseab/experiments/")
model_dir = experiments_dir / "checkpoints/sota_model/"
state_dict_path = model_dir / "last_model-epoch=199-step=019600_state_dict.ckpt"
config_path = model_dir / "config_backup.yaml"

repaint_dir = Path("/home/mila/r/rousseab/experiments/draw_sota_samples/repaint")
repaint_dir.mkdir(exist_ok=True)

plt.style.use(PLOT_STYLE_PATH)

device = torch.device("cuda")

number_of_samples = 1000
total_time_steps = 100
number_of_corrector_steps = 1

acell = 5.43  # Angstroms.
box = np.diag([acell, acell, acell])

number_of_atoms, spatial_dimension = 8, 3
atom_types = np.ones(number_of_atoms, dtype=int)

constrained_relative_coordinates = np.array(
    [[0.5, 0.5, 0.25], [0.5, 0.5, 0.5], [0.5, 0.5, 0.75]], dtype=np.float32
)

if __name__ == "__main__":
    logger.info("Setting up parameters")

    unit_cells = torch.Tensor(box).repeat(number_of_samples, 1, 1).to(device)

    noise_parameters = NoiseParameters(
        total_time_steps=total_time_steps, sigma_min=0.001, sigma_max=0.5
    )

    logger.info("Loading state dict")
    with open(str(state_dict_path), "rb") as fd:
        state_dict = torch.load(fd)

    with open(str(config_path), "r") as fd:
        hyper_params = yaml.load(fd, Loader=yaml.FullLoader)
    logger.info("Instantiate model")
    pl_model = load_diffusion_model(hyper_params)
    pl_model.load_state_dict(state_dict=state_dict)
    pl_model.to(device)
    pl_model.eval()

    sigma_normalized_score_network = pl_model.sigma_normalized_score_network

    sampling_parameters = ConstrainedLangevinGeneratorParameters(
        number_of_corrector_steps=number_of_corrector_steps,
        spatial_dimension=spatial_dimension,
        number_of_atoms=number_of_atoms,
        number_of_samples=number_of_samples,
        cell_dimensions=3 * [acell],
        constrained_relative_coordinates=constrained_relative_coordinates,
        record_samples=True,
    )

    position_generator = ConstrainedLangevinGenerator(
        noise_parameters=noise_parameters,
        sampling_parameters=sampling_parameters,
        sigma_normalized_score_network=sigma_normalized_score_network,
    )

    logger.info("Drawing constrained samples")

    with torch.no_grad():
        samples = position_generator.sample(
            number_of_samples=number_of_samples, device=device, unit_cell=unit_cells
        )

        batch_relative_positions = samples.cpu().numpy()
        batch_positions = np.dot(batch_relative_positions, box)

    position_generator.sample_trajectory_recorder.write_to_pickle(
        repaint_dir / "repaint_trajectories.pkl"
    )

    logger.info("Compute energy from Oracle")
    list_energy = []

    lammps_work_directory = repaint_dir / "samples"
    lammps_work_directory.mkdir(exist_ok=True)

    for idx, positions in enumerate(batch_positions):
        energy, forces = get_energy_and_forces_from_lammps(
            positions,
            box,
            atom_types,
            tmp_work_dir=str(lammps_work_directory),
            pair_coeff_dir=DATA_DIR,
        )
        list_energy.append(energy)
        src = os.path.join(lammps_work_directory, "dump.yaml")
        dst = os.path.join(lammps_work_directory, f"dump_{idx}.yaml")
        os.rename(src, dst)

    energies = np.array(list_energy)

    with open(repaint_dir / "energies.pt", "wb") as fd:
        torch.save(torch.from_numpy(energies), fd)

    logger.info("Plotting energy distributions")
    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle("Energy Distribution for Repaint Structures,")

    common_params = dict(density=True, bins=50, histtype="stepfilled", alpha=0.25)

    ax1 = fig.add_subplot(111)
    ax1.hist(energies, **common_params, label="Sampled Energies", color="red")

    ax1.set_xlabel("Energy (eV)")
    ax1.set_ylabel("Density")
    ax1.legend(loc="upper right", fancybox=True, shadow=True, ncol=1, fontsize=12)
    fig.tight_layout()
    fig.savefig(repaint_dir / f"energy_samples_repaint_{number_of_atoms}_atoms.png")
    plt.close(fig)
