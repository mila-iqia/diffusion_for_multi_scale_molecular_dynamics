"""Convert trajectories to fixed box.

This script takes in existing trajectories sampled from flowMM and forces the box to be the
exact one at all time steps. This will show more clearly the relative coordinates trajectories.
"""
import glob
from pathlib import Path

from matplotlib import pyplot as plt
from pymatgen.core import Lattice, Structure
from tqdm import tqdm

from crystal_diffusion.analysis import PLOT_STYLE_PATH

plt.style.use(PLOT_STYLE_PATH)

cif_dir = Path("/Users/bruno/Desktop/flowmm/si_1x1x1/sample_cif")

trajectory_dir = Path("/Users/bruno/Desktop/flowmm/si_1x1x1/trajectories_cif")
fixed_box_trajectory_dir = Path(
    "/Users/bruno/Desktop/flowmm/si_1x1x1/fixed_box_trajectories_cif"
)


acell = 5.43

if __name__ == "__main__":
    exact_lattice = Lattice.from_parameters(
        a=acell, b=acell, c=acell, alpha=90.0, beta=90.0, gamma=90.0
    )

    sample_directories = glob.glob(str(trajectory_dir / "sample_index_*"))

    for sample_directory in sample_directories:
        print(f"Doing {sample_directory}")
        sample_directory = Path(sample_directory)
        output_directory = fixed_box_trajectory_dir / sample_directory.name
        output_directory.mkdir(exist_ok=True, parents=True)

        cif_files = glob.glob(str(sample_directory / "*.cif"))

        for cif_file in tqdm(cif_files, "CIF files"):
            structure = Structure.from_file(cif_file)
            species = structure.species
            frac_coords = structure.frac_coords
            fixed_structure = Structure(
                lattice=exact_lattice, species=species, coords=frac_coords
            )

            output_cif_file = str(output_directory / Path(cif_file).name)
            fixed_structure.to(output_cif_file, fmt="cif")
