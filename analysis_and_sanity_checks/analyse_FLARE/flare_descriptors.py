import logging
import pickle
from pathlib import Path

import einops
import numpy as np
import pandas as pd
from flare_pp import Structure as FlareStructure
from matplotlib import pyplot as plt
from pymatgen.core import Structure
from pymatgen.io.lammps.data import LammpsData
from tqdm import tqdm

from diffusion_for_multi_scale_molecular_dynamics import TOP_DIR
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.trainer.flare_trainer import (
    FlareConfiguration, FlareTrainer)
from diffusion_for_multi_scale_molecular_dynamics.analysis import (
    PLEASANT_FIG_SIZE, PLOT_STYLE_PATH)

logging.basicConfig(level=logging.INFO)

plt.style.use(PLOT_STYLE_PATH)


def get_structure_normalized_ace_descriptors(structure: Structure, flare_trainer: FlareTrainer):
    """Get structure normalized ACE descriptors."""
    cell = structure.lattice.matrix
    species = [0] * len(structure)
    positions = structure.cart_coords

    flare_structure = FlareStructure(cell, species, positions,
                                     flare_configuration.cutoff,
                                     flare_trainer._descriptor_calculators)

    # This stores as an array inside
    flare_structure.compute_descriptors()
    descriptors = flare_structure.descriptors[0].descriptors[0]
    n, d = descriptors.shape
    descriptor_norms = np.linalg.norm(descriptors, axis=1)

    normalized_descriptors = descriptors / einops.repeat(descriptor_norms, "n -> n d", d=d)
    return normalized_descriptors


def get_random_pair_indices(size_array1: int, size_array2: int, number_of_samples: int, exclude_identity: bool):
    """Get random pair indices."""
    count = 0

    list_pairs = []

    while count < number_of_samples:
        idx1 = np.random.randint(size_array1)
        idx2 = np.random.randint(size_array2)
        if exclude_identity and idx1 == idx2:
            continue

        pair = (idx1, idx2)
        if pair not in list_pairs:
            list_pairs.append(pair)
            count += 1

    all_indices = np.array(list_pairs)

    return all_indices[:, 0], all_indices[:, 1]


element_list = ["Si"]
variance_type = "local"
seed = 42

# We assume that FLARE models have been pre-trained and are available here.

analysis_dir = TOP_DIR / "analysis_and_sanity_checks/analyse_FLARE"

data_dir = TOP_DIR / "experiments/active_learning/pretraining_flare/data/"
amorphous_si_file_path = TOP_DIR / "experiments/active_learning/amorphous_silicon/reference/initial_configuration.dat"

amorphous_exp_dir = Path("/Users/brunorousseau/courtois/july26/active_learning/amorphous_silicon/excise_and_repaint/")
oracle_path = amorphous_exp_dir / "output/run1/campaign_1/round_1/oracle"

images_dir = analysis_dir / "images"
images_dir.mkdir(parents=True, exist_ok=True)

flare_configuration = FlareConfiguration(
    cutoff=5.0,
    elements=element_list,
    n_radial=12,
    lmax=3,
    initial_sigma=1.0,
    initial_sigma_e=0.1,
    initial_sigma_f=0.001,
    initial_sigma_s=1.0,
    variance_type="local"
)

number_of_samples = 1000

if __name__ == "__main__":

    flare_trainer = FlareTrainer(flare_configuration)

    lammps_data = LammpsData.from_file(amorphous_si_file_path, atom_style='atomic')
    amorphous_si_structure = lammps_data.structure

    list_crystal_descriptors = []
    with open(data_dir / "valid_labelled_structures.pkl", "rb") as fd:
        list_valid_labelled_structures = pickle.load(fd)
        for labelled_structure in list_valid_labelled_structures:
            crystal_descriptors = get_structure_normalized_ace_descriptors(labelled_structure.structure, flare_trainer)
            list_crystal_descriptors.append(crystal_descriptors)
    crystal_descriptors = np.concatenate(list_crystal_descriptors)

    df = pd.read_pickle(oracle_path / "oracle_single_point_calculations.pkl")
    list_repaint_descriptors = []
    for idx, structure in enumerate(df['structure'].values):
        repaint_descriptors = get_structure_normalized_ace_descriptors(structure, flare_trainer)
        list_repaint_descriptors.append(repaint_descriptors)
    repaint_descriptors = np.concatenate(list_repaint_descriptors)

    amorphous_descriptors = get_structure_normalized_ace_descriptors(amorphous_si_structure, flare_trainer)

    descriptors = dict(crystal=crystal_descriptors, repaint=repaint_descriptors, amorphous=amorphous_descriptors)

    descriptor_subsets = dict()
    for key, descriptors in tqdm(descriptors.items(), "Random Descriptors"):
        number_of_descriptors = len(descriptors)
        ids1, ids2 = get_random_pair_indices(size_array1=number_of_descriptors,
                                             size_array2=number_of_descriptors,
                                             number_of_samples=number_of_samples,
                                             exclude_identity=True)
        descriptor_subsets[key] = (descriptors[ids1], descriptors[ids2])

    figsize = (PLEASANT_FIG_SIZE[0], 1.5 * PLEASANT_FIG_SIZE[1])
    fig = plt.figure(figsize=figsize)
    fig.suptitle(f"Cosine Similarity between Descriptors\n"
                 f"Sampling randomly {number_of_samples} ACE descriptors for each Group")

    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    common_dict = dict(alpha=0.5, histtype='step', bins=50, lw=2)

    list_key_pairs = [('crystal', 'crystal'), ('amorphous', 'amorphous'), ('crystal', 'amorphous')]
    list_colors = ['green', 'red', 'blue']
    for (key1, key2), color in zip(list_key_pairs, list_colors):
        label = f'{key1} - {key2}'
        cosine_distances = einops.einsum(descriptor_subsets[key1][0],
                                         descriptor_subsets[key2][1],
                                         'batch space, batch space -> batch')
        ax1.hist(cosine_distances, **common_dict, label=label, color=color)

    list_key_pairs = [('crystal', 'repaint'), ('amorphous', 'repaint'), ('repaint', 'repaint')]
    list_colors = ['green', 'red', 'purple']
    for (key1, key2), color in zip(list_key_pairs, list_colors):
        label = f'{key1} - {key2}'
        cosine_distances = einops.einsum(descriptor_subsets[key1][0],
                                         descriptor_subsets[key2][1],
                                         'batch space, batch space -> batch')
        ax2.hist(cosine_distances, **common_dict, label=label, color=color)

    for ax in [ax1, ax2]:
        ax.legend(loc=0, fontsize=16)
        ax.set_yscale('log')
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Count')
        ax.set_xlim(0., 1.01)
    fig.tight_layout()
    fig.savefig(images_dir / "flare_descriptor_cosine_similarity.png")
