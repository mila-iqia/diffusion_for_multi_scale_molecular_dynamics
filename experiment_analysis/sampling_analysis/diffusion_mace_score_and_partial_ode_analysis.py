import logging
import tempfile
from pathlib import Path

import einops
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from crystal_diffusion.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from crystal_diffusion.analysis.generator_sample_analysis_utils import \
    PartialODEPositionGenerator
from crystal_diffusion.models.position_diffusion_lightning_model import \
    PositionDiffusionLightningModel
from crystal_diffusion.namespace import (CARTESIAN_FORCES, NOISE,
                                         NOISY_RELATIVE_COORDINATES, TIME,
                                         UNIT_CELL)
from crystal_diffusion.oracle.lammps import get_energy_and_forces_from_lammps
from crystal_diffusion.samplers.noisy_relative_coordinates_sampler import \
    NoisyRelativeCoordinatesSampler
from crystal_diffusion.samplers.variance_sampler import NoiseParameters
from crystal_diffusion.utils.tensor_utils import \
    broadcast_batch_tensor_to_all_dimensions

logger = logging.getLogger(__name__)

plt.style.use(PLOT_STYLE_PATH)


# Some hardcoded paths and parameters. Change as needed!
base_data_dir = Path("/Users/bruno/courtois/difface_ode/run1")
position_samples_dir = base_data_dir / "diffusion_position_samples"
energy_data_directory = base_data_dir / "energy_samples"
model_path = base_data_dir / "best_model" / "best_model-epoch=016-step=001666.ckpt"

partial_samples_dir = base_data_dir / "partial_samples"
partial_samples_dir.mkdir(exist_ok=True)


canonical_relative_coordinates = torch.tensor([[0.00, 0.00, 0.00],
                                               [0.00, 0.50, 0.50],
                                               [0.50, 0.00, 0.50],
                                               [0.00, 0.50, 0.50],
                                               [0.25, 0.25, 0.25],
                                               [0.25, 0.75, 0.75],
                                               [0.75, 0.25, 0.75],
                                               [0.75, 0.75, 0.25]])

reference_relative_coordinates = torch.tensor([[0.0166, 0.0026, 0.9913],
                                               [0.9936, 0.4954, 0.5073],
                                               [0.4921, 0.9992, 0.4994],
                                               [0.4954, 0.5009, 0.9965],
                                               [0.2470, 0.2540, 0.2664],
                                               [0.2481, 0.7434, 0.7445],
                                               [0.7475, 0.2483, 0.7489],
                                               [0.7598, 0.7563, 0.2456]])

sigma_min = 0.001
sigma_max = 0.5
total_time_steps = 100

noise_parameters = NoiseParameters(total_time_steps=total_time_steps, sigma_min=sigma_min, sigma_max=sigma_max)


cell_dimensions = torch.tensor([5.43, 5.43, 5.43])

number_of_atoms = 8
spatial_dimension = 3
batch_size = 100
if __name__ == '__main__':
    noisy_relative_coordinates_sampler = NoisyRelativeCoordinatesSampler()
    unit_cell = torch.diag(torch.Tensor(cell_dimensions)).unsqueeze(0).repeat(batch_size, 1, 1)

    x0 = einops.repeat(reference_relative_coordinates, "n d -> b n d", b=batch_size)

    x_bad = torch.rand(x0.shape)

    model = PositionDiffusionLightningModel.load_from_checkpoint(model_path)
    model.eval()

    list_tf = np.linspace(0.1, 1, 10)

    atom_types = np.ones(number_of_atoms, dtype=int)

    on_manifold_dataset = []
    off_manifold_dataset = []
    with torch.no_grad():
        for tf in tqdm(list_tf, 'times'):
            times = torch.ones(batch_size) * tf
            sigmas = sigma_min ** (1.0 - times) * sigma_max ** times

            broadcast_sigmas = broadcast_batch_tensor_to_all_dimensions(batch_values=sigmas, final_shape=x0.shape)
            xt = noisy_relative_coordinates_sampler.get_noisy_relative_coordinates_sample(x0, broadcast_sigmas)

            noise_parameters.total_time_steps = int(100 * tf) + 1
            generator = PartialODEPositionGenerator(noise_parameters,
                                                    number_of_atoms,
                                                    spatial_dimension,
                                                    model.sigma_normalized_score_network,
                                                    initial_relative_coordinates=xt,
                                                    record_samples=True,
                                                    tf=tf)

            logger.info("Generating Samples")
            batch_relative_coordinates = generator.sample(number_of_samples=batch_size,
                                                          device=torch.device('cpu'),
                                                          unit_cell=unit_cell)
            sample_output_path = str(partial_samples_dir / f"diffusion_position_sample_time={tf:2.1f}.pt")
            # write trajectories to disk and reset to save memory
            generator.sample_trajectory_recorder.write_to_pickle(sample_output_path)
            logger.info("Done Generating Samples")

            batch_cartesian_positions = torch.bmm(batch_relative_coordinates, unit_cell)

            box = unit_cell[0].numpy()

            list_energy = []

            logger.info("Compute energy from Oracle")
            with tempfile.TemporaryDirectory() as tmp_work_dir:
                for positions in batch_cartesian_positions.numpy():
                    energy, forces = get_energy_and_forces_from_lammps(positions,
                                                                       box,
                                                                       atom_types,
                                                                       tmp_work_dir=tmp_work_dir)
                    list_energy.append(energy)

            energies = torch.tensor(list_energy)
            energy_output_path = str(partial_samples_dir / f"diffusion_energies_sample_time={tf:2.1f}.pt")
            with open(energy_output_path, 'wb') as fd:
                torch.save(energies, fd)

            batch = {NOISY_RELATIVE_COORDINATES: xt,
                     NOISE: sigmas.unsqueeze(-1),
                     TIME: times.unsqueeze(-1),
                     UNIT_CELL: unit_cell,
                     CARTESIAN_FORCES: torch.zeros_like(x0)
                     }
            normalized_scores = model.sigma_normalized_score_network(batch)
            rms_norm_scores = torch.sqrt((torch.linalg.norm(normalized_scores, dim=2) ** 2).mean(dim=1))
            on_manifold_dataset.append(rms_norm_scores.numpy())

            bad_batch = {NOISY_RELATIVE_COORDINATES: x_bad,
                         NOISE: sigmas.unsqueeze(-1),
                         TIME: times.unsqueeze(-1),
                         UNIT_CELL: unit_cell,
                         CARTESIAN_FORCES: torch.zeros_like(x0)
                         }
            bad_normalized_scores = model.sigma_normalized_score_network(bad_batch)
            bad_rms_norm_scores = torch.sqrt((torch.linalg.norm(bad_normalized_scores, dim=2) ** 2).mean(dim=1))
            off_manifold_dataset.append(bad_rms_norm_scores.numpy())

    labels = []

    def add_label(violin, label):
        """Add some labels to the plot."""
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))

    fig1 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig1.suptitle('Mean Score Norm')

    ax1 = fig1.add_subplot(111)

    add_label(ax1.violinplot(on_manifold_dataset, positions=list_tf, widths=0.05, showmeans=True),
              "ON Manifold")
    add_label(ax1.violinplot(off_manifold_dataset, positions=list_tf, widths=0.05, showmeans=True),
              "OFF Manifold")

    ax1.set_xlabel('Diffusion Time')
    ax1.set_ylabel('RMS Normalized Score')
    ax1.set_yscale('log')

    ax1.legend(*zip(*labels), loc=0)
    fig1.tight_layout()
    plt.show()
