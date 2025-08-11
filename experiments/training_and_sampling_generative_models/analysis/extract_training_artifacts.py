"""Extact Training Artifacts.

It can be time-consuming to extract the information we want to plot. This script performs
the preprocessing, so that plotting scripts can be light and quick.
"""

import glob
import logging
import pickle
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

from diffusion_for_multi_scale_molecular_dynamics import DATA_DIR
from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.lammps_for_diffusion_data_module import (
    LammpsDataModuleParameters, LammpsForDiffusionDataModule)
from diffusion_for_multi_scale_molecular_dynamics.metrics.kolmogorov_smirnov_metrics import \
    KolmogorovSmirnovMetrics
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import (
    get_positions_from_coordinates,
    map_lattice_parameters_to_unit_cell_vectors)
from diffusion_for_multi_scale_molecular_dynamics.utils.logging_utils import \
    setup_analysis_logger
from diffusion_for_multi_scale_molecular_dynamics.utils.structure_utils import \
    compute_distances_in_batch

setup_analysis_logger()


def get_epochs_and_ks_distances(
    path_template_string: str, reference_values: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    """Get epochs and KS distances."""
    list_epochs = []
    list_ks_distances = []
    for data_file_path in tqdm(glob.glob(path_template_string), "loading data"):
        epoch = int(re.match(epoch_pattern, Path(data_file_path).name).group("epoch"))

        predicted_values = torch.load(data_file_path)

        ks_metric = KolmogorovSmirnovMetrics()
        ks_metric.register_reference_samples(reference_values)
        ks_metric.register_predicted_samples(predicted_values)
        ks_distance, _ = ks_metric.compute_kolmogorov_smirnov_distance_and_pvalue()

        list_epochs.append(epoch)
        list_ks_distances.append(ks_distance)

    list_epochs = np.array(list_epochs)
    list_ks_distances = np.array(list_ks_distances)
    order = np.argsort(list_epochs)
    list_epochs = list_epochs[order]
    list_ks_distances = list_ks_distances[order]

    return list_epochs, list_ks_distances


artifact_directory = Path("./data_artifacts")
artifact_directory.mkdir(exist_ok=True)

epoch_pattern = r".*=(?P<epoch>\d+)\.pt"


# dataset_name = "Si_diffusion_1x1x1"
# dataset_name = "Si_diffusion_2x2x2"
# dataset_name = "Si_diffusion_3x3x3"
# dataset_name = "SiGe_diffusion_1x1x1"
# dataset_name = "SiGe_diffusion_2x2x2"
dataset_name = "SiGe_diffusion_3x3x3"


base_experiment_dir = Path("/Users/brunorousseau/courtois/july26/")

match dataset_name:
    case "Si_diffusion_1x1x1":
        experiment_dir = base_experiment_dir / "july26_si_egnn_1x1x1/run1/"
        max_atom = 8
        structure_factor_max_distance = 5.0
        element_list = ["Si"]
    case "SiGe_diffusion_1x1x1":
        experiment_dir = base_experiment_dir / "july26_sige_egnn_1x1x1/run1/"
        max_atom = 8
        structure_factor_max_distance = 5.0
        element_list = ["Si", "Ge"]
    case "Si_diffusion_2x2x2":
        experiment_dir = base_experiment_dir / "july26_si_egnn_2x2x2/run1/"
        max_atom = 64
        structure_factor_max_distance = 10.0
        element_list = ["Si"]
    case "SiGe_diffusion_2x2x2":
        experiment_dir = base_experiment_dir / "july26_sige_egnn_2x2x2/run1/"
        max_atom = 64
        structure_factor_max_distance = 10.0
        element_list = ["Si", "Ge"]
    case "Si_diffusion_3x3x3":
        experiment_dir = base_experiment_dir / "july26_si_egnn_3x3x3/run1/"
        max_atom = 216
        structure_factor_max_distance = 15.0
        element_list = ["Si"]
    case "SiGe_diffusion_3x3x3":
        experiment_dir = base_experiment_dir / "july26_sige_egnn_3x3x3/run1/"
        max_atom = 216
        structure_factor_max_distance = 15.0
        element_list = ["Si", "Ge"]


energy_samples_dir = experiment_dir / "output/energy_samples"
distance_samples_dir = experiment_dir / "output/distance_samples"

lammps_run_dir = DATA_DIR / dataset_name
processed_dataset_dir = lammps_run_dir / "processed"
cache_dir = lammps_run_dir / "cache"

batch_size = 128
data_params = LammpsDataModuleParameters(
    batch_size=batch_size,
    max_atom=max_atom,
    noise_parameters=NoiseParameters(total_time_steps=1),  # dummy. Not used.
    use_optimal_transport=False,
    use_fixed_lattice_parameters=True,
    elements=element_list,
)

if __name__ == "__main__":

    logging.info(f"Extracting training artifacts for {dataset_name}...")

    datamodule = LammpsForDiffusionDataModule(
        lammps_run_dir=lammps_run_dir,
        processed_dataset_dir=processed_dataset_dir,
        hyper_params=data_params,
        working_cache_dir=cache_dir,
    )
    datamodule.setup()

    logging.info("Extracting the validation dataloader...")
    validation_dataloader = datamodule.val_dataloader()

    structure_ks_metric = KolmogorovSmirnovMetrics()
    energy_ks_metric = KolmogorovSmirnovMetrics()
    logging.info("Iterating over the validation dataloader...")
    for batch in tqdm(validation_dataloader, "batches"):
        reference_energies = batch["potential_energy"]
        energy_ks_metric.register_reference_samples(reference_energies)

        basis_vectors = map_lattice_parameters_to_unit_cell_vectors(
            batch["lattice_parameters"]
        )
        cartesian_positions = get_positions_from_coordinates(
            relative_coordinates=batch["relative_coordinates"],
            basis_vectors=basis_vectors,
        )

        reference_distances = compute_distances_in_batch(
            cartesian_positions=cartesian_positions,
            unit_cell=basis_vectors,
            max_distance=structure_factor_max_distance,
        )
        structure_ks_metric.register_reference_samples(reference_distances)

    reference_energies = energy_ks_metric.reference_samples_metric.compute()
    reference_structures = structure_ks_metric.reference_samples_metric.compute()

    output_file = artifact_directory / f"reference_energies_{dataset_name}.pkl"
    with open(output_file, "wb") as fd:
        pickle.dump(reference_energies, fd)

    output_file = artifact_directory / f"reference_structures_{dataset_name}.pkl"
    with open(output_file, "wb") as fd:
        pickle.dump(reference_structures, fd)

    logging.info("Computing Energy KS distances...")
    list_energy_epochs, list_energy_ks_distances = get_epochs_and_ks_distances(
        path_template_string=str(energy_samples_dir / "energies_sample_epoch=*.pt"),
        reference_values=reference_energies,
    )

    logging.info("Computing Interatomic Distance KS distances...")
    list_structure_epochs, list_structure_ks_distances = get_epochs_and_ks_distances(
        path_template_string=str(distance_samples_dir / "distances_sample_epoch=*.pt"),
        reference_values=reference_structures,
    )

    torch.testing.assert_close(list_structure_epochs, list_energy_epochs)

    list_epochs = list_energy_epochs

    ks_distances_dictionary = dict(
        epochs=list_epochs,
        energy_ks_distances=list_energy_ks_distances,
        structure_ks_distances=list_structure_ks_distances,
    )

    logging.info("Dumping to file...")
    output_file = artifact_directory / f"ks_distances_{dataset_name}.pkl"
    with open(output_file, "wb") as fd:
        pickle.dump(ks_distances_dictionary, fd)

    logging.info("Done!")
