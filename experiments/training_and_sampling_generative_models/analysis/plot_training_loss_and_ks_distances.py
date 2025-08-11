import json
import pickle
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from diffusion_for_multi_scale_molecular_dynamics.analysis import (
    PLEASANT_FIG_SIZE, PLOT_STYLE_PATH)

plt.style.use(PLOT_STYLE_PATH)


comet_download_dir = Path("/Users/brunorousseau/courtois/july26/Comet_Downloads/")

artifact_directory = Path(__file__).parent / "data_artifacts"
image_directory = Path(__file__).parent / "images"
image_directory.mkdir(exist_ok=True)

dataset_name = "Si_diffusion_1x1x1"
dataset_title = "Si 1x1x1"

# dataset_name = "Si_diffusion_2x2x2"
# dataset_title = "Si 2x2x2"

# dataset_name = "Si_diffusion_3x3x3"
# dataset_title = "Si 3x3x3"

# dataset_name = "SiGe_diffusion_1x1x1"
# dataset_title = "SiGe 1x1x1"

# dataset_name = "SiGe_diffusion_2x2x2"
# dataset_title = "SiGe 2x2x2"

# dataset_name = "SiGe_diffusion_3x3x3"
# dataset_title = "SiGe 3x3x3"

data_file = artifact_directory / f"ks_distances_{dataset_name}.pkl"

json_data = comet_download_dir / f"train_epoch_loss_chart_data_{dataset_name}.json"

if __name__ == "__main__":

    json_data = json.load(open(json_data))

    for data_dict in json_data:
        if "validation_epoch_loss" in data_dict["name"]:
            list_val_epochs = data_dict["x"]
            list_val_loss = data_dict["y"]
        elif "train_epoch_loss" in data_dict["name"]:
            list_train_epochs = data_dict["x"]
            list_train_loss = data_dict["y"]

    with open(data_file, "rb") as fd:
        data = pickle.load(fd)

    list_ks_epochs = data["epochs"]
    list_energy_ks_distances = data["energy_ks_distances"]
    list_structure_ks_distances = data["structure_ks_distances"]

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle(f"Loss During Training: {dataset_title}")
    ax1 = fig.add_subplot(111)
    ax1.plot(
        list_train_epochs,
        list_train_loss,
        ".-",
        color="blue",
        label="Loss over Training Dataset",
    )
    ax1.plot(
        list_val_epochs,
        list_val_loss,
        ".--",
        color="green",
        label="Loss over Validation Dataset",
    )

    ax1.legend(loc=0)
    ax1.set_xlim(list_train_epochs[0], list_train_epochs[-1])
    ax1.set_xlabel("Training Epoch")
    ax1.set_ylabel("Loss")
    fig.tight_layout()
    fig.savefig(image_directory / f"training_loss_{dataset_name}.png")

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle(f"Kolmogorov Smirnov Distances: {dataset_title}")
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.plot(list_ks_epochs, list_energy_ks_distances, ".-", color="green")
    ax2.plot(list_ks_epochs, list_structure_ks_distances, ".-", color="green")

    ymax1 = np.max(list_energy_ks_distances)
    ymax2 = np.max(list_structure_ks_distances)

    for ax, ymax in zip([ax1, ax2], [ymax1, ymax2]):
        ax.set_xlabel("Training Epoch")
        ax.set_ylabel("KS distance")
        ax.set_xlim(1, list_ks_epochs[-1])
        ax.set_ylim(ymin=0, ymax=ymax + 0.01)

    ax1.set_title("Total Energies")
    ax2.set_title("Interatomic Distances")

    fig.tight_layout()
    fig.savefig(image_directory / f"ks_distances_{dataset_name}.png")
