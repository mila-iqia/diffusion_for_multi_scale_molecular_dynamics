"""Analyse flowMM samples.

This script reads in the samples from flowMM as CIFs and compute their energies.
"""
import glob
import tempfile
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from pymatgen.core import Structure
from tqdm import tqdm

from crystal_diffusion import DATA_DIR
from crystal_diffusion.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from crystal_diffusion.callbacks.sampling_callback import \
    DiffusionSamplingCallback
from crystal_diffusion.data.diffusion.data_loader import (
    LammpsForDiffusionDataModule, LammpsLoaderParameters)
from crystal_diffusion.oracle.lammps import get_energy_and_forces_from_lammps

plt.style.use(PLOT_STYLE_PATH)

cif_dir = Path("/Users/bruno/Desktop/flowmm/si_1x1x1/sample_cif")

if __name__ == "__main__":
    dataset_name = "si_diffusion_1x1x1"
    data_params = LammpsLoaderParameters(batch_size=128, max_atom=8)
    lammps_run_dir = str(DATA_DIR / dataset_name)
    processed_dataset_dir = str(DATA_DIR / dataset_name / "processed")
    cache_dir = str(DATA_DIR / dataset_name / "cache")

    datamodule = LammpsForDiffusionDataModule(
        lammps_run_dir=lammps_run_dir,
        processed_dataset_dir=processed_dataset_dir,
        hyper_params=data_params,
        working_cache_dir=cache_dir,
    )
    datamodule.setup()

    validation_dataset_energies = datamodule.valid_dataset[:][
        "potential_energy"
    ].numpy()

    list_structures = []
    list_file_names = []
    for cif_file in tqdm(glob.glob(str(cif_dir / "*.cif")), "Reading CIF"):
        structure = Structure.from_file(cif_file)
        list_file_names.append(Path(cif_file).name)
        list_structures.append(structure)

    list_file_names = np.array(list_file_names)

    list_a = []
    list_b = []
    list_c = []

    for structure in list_structures:
        a, b, c = structure.lattice.abc
        list_a.append(a)
        list_b.append(b)
        list_c.append(c)

    exact_acell = 5.43
    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle("Distribution of Lattice Dimension for Si 1x1x1 Samples")
    ax = fig.add_subplot(111)
    ax.vlines(
        exact_acell,
        0,
        10,
        color="black",
        label=f"Exact value: {exact_acell:3.2f} Angstrom",
    )
    common_params = dict(density=True, bins=25, histtype="stepfilled", alpha=0.25)
    ax.hist(list_a, **common_params, label="a", color="green")
    ax.hist(list_b, **common_params, label="b", color="blue")
    ax.hist(list_c, **common_params, label="c", color="red")
    ax.legend(loc=0)
    ax.set_xlabel("Cell dimension (Angstrom)")
    ax.set_ylabel("Density")
    plt.show()

    list_energies = []
    list_energies_exact_box = []
    exact_box = np.diag(3 * [exact_acell])
    atom_types = np.ones(8, dtype=int)
    with tempfile.TemporaryDirectory() as tmp_work_dir:
        for structure in tqdm(list_structures, "Computing Energies"):
            box = np.diag(np.diag(structure.lattice.matrix))
            energy, forces = get_energy_and_forces_from_lammps(
                structure.frac_coords @ box, box, atom_types, tmp_work_dir=tmp_work_dir
            )
            list_energies.append(energy)

            energy, forces = get_energy_and_forces_from_lammps(
                structure.frac_coords @ exact_box,
                exact_box,
                atom_types,
                tmp_work_dir=tmp_work_dir,
            )
            list_energies_exact_box.append(energy)

    sample_energies = np.array(list_energies)
    sample_energies_exact_box = np.array(list_energies_exact_box)

    fig1 = DiffusionSamplingCallback._plot_energy_histogram(
        sample_energies, validation_dataset_energies, epoch=0
    )
    fig1.suptitle("Comparing Samples with Reference: Sampled Box")
    plt.show()

    fig2 = DiffusionSamplingCallback._plot_energy_histogram(
        sample_energies_exact_box, validation_dataset_energies, epoch=0
    )
    fig2.suptitle("Comparing Samples with Reference: Exact Box")
    plt.show()

    list_q = np.linspace(0, 1, 101)
    list_validation_energy_quantiles = np.quantile(validation_dataset_energies, list_q)
    list_sample_energy_quantiles = np.quantile(sample_energies, list_q)
    list_sample_energy_exact_box_quantiles = np.quantile(
        sample_energies_exact_box, list_q
    )

    fig3 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig3.suptitle("Energy Quantiles")
    ax1 = fig3.add_subplot(211)
    ax2 = fig3.add_subplot(212)
    ax1.set_title("Zoom In View")
    ax2.set_title("Global View")
    for ax in [ax1, ax2]:
        ax.plot(
            100 * list_q,
            list_validation_energy_quantiles,
            "-",
            lw=2,
            color="green",
            label="Validation Energies",
        )
        ax.plot(
            100 * list_q,
            list_sample_energy_quantiles,
            "-",
            lw=3,
            alpha=0.4,
            color="red",
            label="Sampled Energies, Sampled Box",
        )
        ax.plot(
            100 * list_q,
            list_sample_energy_exact_box_quantiles,
            "-",
            lw=2,
            alpha=0.4,
            color="blue",
            label="Sampled Energies, Exact Box",
        )

        ax.set_xlim([-0.1, 100.1])
        ax.set_xlabel("Quantile (%)")
        ax.set_ylabel("Energy (eV)")
    ax2.legend(loc=0)
    ax1.set_ylim(
        [
            validation_dataset_energies.min() - 0.5,
            validation_dataset_energies.min() + 1.0,
        ]
    )
    fig3.tight_layout()
    plt.show()
