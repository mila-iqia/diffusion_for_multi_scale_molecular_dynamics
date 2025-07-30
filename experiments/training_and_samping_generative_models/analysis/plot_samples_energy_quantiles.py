import pickle
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from diffusion_for_multi_scale_molecular_dynamics.analysis import \
    PLEASANT_FIG_SIZE

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
        sample_top_dir = base_experiment_dir / "july26_si_egnn_1x1x1/samples_from_run1/"
    case "SiGe_diffusion_1x1x1":
        sample_top_dir = base_experiment_dir / "july26_sige_egnn_1x1x1/samples_from_run1/"
    case "Si_diffusion_2x2x2":
        experiment_dir = base_experiment_dir / "july26_si_egnn_2x2x2/samples_from_run1/"
    case "SiGe_diffusion_2x2x2":
        experiment_dir = base_experiment_dir / "july26_sige_egnn_2x2x2/samples_from_run1/"
    case "Si_diffusion_3x3x3":
        experiment_dir = base_experiment_dir / "july26_si_egnn_3x3x3/samples_from_run1/"
    case "SiGe_diffusion_3x3x3":
        experiment_dir = base_experiment_dir / "july26_sige_egnn_3x3x3/samples_from_run1/"


list_T = [1000, 10000]

if __name__ == "__main__":

    list_q = np.linspace(0, 1, 101)

    output_file = artifact_directory / f"reference_energies_{dataset_name}.pkl"
    with open(output_file, "rb") as fd:
        reference_energies = pickle.load(fd).numpy()
    dataset_quantiles = np.quantile(reference_energies, list_q)

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle(f"Sampling Energy Quantiles\n{dataset_name}")
    ax = fig.add_subplot(111)

    label = "Validation Dataset"
    ax.plot(100 * list_q, dataset_quantiles, "--", lw=5, color="green", label=label)

    for T in list_T:
        sample_energies = torch.load(sample_top_dir / f'sample_output_T={T}' / "energies.pt")
        sample_quantiles = np.quantile(sample_energies, list_q)

        label = f"Samples with T={T} Time Steps"

        ax.plot(100 * list_q, sample_quantiles, "-", lw=3, label=label)

    ax.set_xlabel("Quantile (%)")
    ax.set_ylabel("Energy (eV)")
    ax.set_xlim(-0.1, 100.1)
    ax.legend(loc="upper left", fancybox=True, shadow=True, ncol=1, fontsize=12)
    fig.tight_layout()

    fig.savefig(image_directory / f"sample_energy_quantiles_{dataset_name}.png")
    plt.close(fig)
