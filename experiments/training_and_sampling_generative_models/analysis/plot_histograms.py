import pickle
from pathlib import Path

import torch
from matplotlib import pyplot as plt

from diffusion_for_multi_scale_molecular_dynamics.callbacks.sampling_visualization_callback import \
    SamplingVisualizationCallback

artifact_directory = Path(__file__).parent / "data_artifacts"
image_directory = Path(__file__).parent / "images"
image_directory.mkdir(exist_ok=True)

base_experiment_dir = Path("/Users/brunorousseau/courtois/july26/")

dataset_name = "Si_diffusion_1x1x1"
# dataset_name = "Si_diffusion_2x2x2"
# dataset_name = "Si_diffusion_3x3x3"
# dataset_name = "SiGe_diffusion_1x1x1"
# dataset_name = "SiGe_diffusion_2x2x2"
# dataset_name = "SiGe_diffusion_3x3x3"

match dataset_name:
    case "Si_diffusion_1x1x1":
        experiment_dir = base_experiment_dir / "july26_si_egnn_1x1x1/run1/"
        best_epoch = 39
        list_epochs = [36, 37, 38, 39, 40, 41, 42]
    case "SiGe_diffusion_1x1x1":
        experiment_dir = base_experiment_dir / "july26_sige_egnn_1x1x1/run1/"
        best_epoch = 81
        list_epochs = [best_epoch]
    case "Si_diffusion_2x2x2":
        experiment_dir = base_experiment_dir / "july26_si_egnn_2x2x2/run1/"
        best_epoch = 64
        list_epochs = [best_epoch]
    case "SiGe_diffusion_2x2x2":
        experiment_dir = base_experiment_dir / "july26_sige_egnn_2x2x2/run1/"
        best_epoch = 68
        list_epochs = [best_epoch]
    case "Si_diffusion_3x3x3":
        experiment_dir = base_experiment_dir / "july26_si_egnn_3x3x3/run1/"
        best_epoch = 4
        list_epochs = [best_epoch]
    case "SiGe_diffusion_3x3x3":
        experiment_dir = base_experiment_dir / "july26_sige_egnn_3x3x3/run1/"
        best_epoch = 1
        list_epochs = [best_epoch]

energy_samples_dir = experiment_dir / "output/energy_samples"
distance_samples_dir = experiment_dir / "output/distance_samples"

if __name__ == "__main__":

    output_file = artifact_directory / f"reference_energies_{dataset_name}.pkl"
    with open(output_file, "rb") as fd:
        reference_energies = pickle.load(fd).numpy()

    output_file = artifact_directory / f"reference_structures_{dataset_name}.pkl"
    with open(output_file, "rb") as fd:
        reference_distances = pickle.load(fd).numpy()

    for epoch in list_epochs:
        sample_energies = torch.load(
            energy_samples_dir / f"energies_sample_epoch={epoch}.pt"
        ).numpy()
        sample_distances = torch.load(
            distance_samples_dir / f"distances_sample_epoch={epoch}.pt"
        ).numpy()

        fig = SamplingVisualizationCallback._plot_energy_histogram(
            sample_energies=sample_energies,
            validation_dataset_energies=reference_energies,
            epoch=epoch,
        )
        ax = fig.gca()
        ax.legend(loc="upper right", fancybox=True, shadow=True, ncol=1, fontsize=12)
        fig.savefig(
            image_directory / f"energy_histogram_epoch_{epoch}_{dataset_name}.png"
        )
        plt.close(fig)

        fig = SamplingVisualizationCallback._plot_distance_histogram(
            sample_distances=sample_distances,
            validation_dataset_distances=reference_distances,
            epoch=epoch,
        )
        ax = fig.gca()
        ax.legend(loc="upper left", fancybox=True, shadow=True, ncol=1, fontsize=16)
        fig.savefig(
            image_directory / f"distance_histogram_epoch_{epoch}_{dataset_name}.png"
        )
        plt.close(fig)
