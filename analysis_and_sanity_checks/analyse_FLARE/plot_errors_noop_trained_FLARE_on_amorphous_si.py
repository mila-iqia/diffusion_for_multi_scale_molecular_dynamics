from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from pymatgen.io.lammps.data import LammpsData

from diffusion_for_multi_scale_molecular_dynamics import DATA_DIR, TOP_DIR
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.lammps.lammps_runner import \
    LammpsRunner
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.mapped_flare_single_point_calculator import \
    MappedFlareSinglePointCalculator  # noqa
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.stillinger_weber_single_point_calculator import \
    StillingerWeberSinglePointCalculator  # noqa
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.utils import \
    compute_errors_and_uncertainties
from diffusion_for_multi_scale_molecular_dynamics.analysis import (
    PLEASANT_FIG_SIZE, PLOT_STYLE_PATH)

plt.style.use(PLOT_STYLE_PATH)

lammps_executable_path = Path("/Users/brunorousseau/sources/lammps/build/lmp")

sw_coefficients_file_path = DATA_DIR / "stillinger_weber_coefficients/Si.sw"

# We assume that FLARE models have been pre-trained and are available here.
amorphous_si_file_path = TOP_DIR / "experiments/active_learning/amorphous_silicon/reference/initial_configuration.dat"

analysis_dir = TOP_DIR / "analysis_and_sanity_checks/analyse_FLARE"
images_dir = analysis_dir / "images"
images_dir.mkdir(parents=True, exist_ok=True)

# Trained FLARE on NoOp
checkpoint_directory = Path("/Users/brunorousseau/courtois/july26/active_learning/amorphous_silicon/"
                            "noop/output/run1/campaign_1/round_59/FLARE_mapped_coefficients/")

pair_coeff_file_path = checkpoint_directory / "lmp59.flare"
mapped_uncertainty_file_path = checkpoint_directory / "map_unc_lmp59.flare"

if __name__ == "__main__":

    lammps_runner = LammpsRunner(
        lammps_executable_path, mpi_processors=4, openmp_threads=4
    )

    lammps_data = LammpsData.from_file(amorphous_si_file_path, atom_style='atomic')
    amorphous_si_structure = lammps_data.structure
    oracle = StillingerWeberSinglePointCalculator(lammps_runner=lammps_runner,
                                                  sw_coefficients_file_path=sw_coefficients_file_path)
    list_amorphous_si_labelled_structures = [oracle.calculate(amorphous_si_structure)]

    ground_truth_forces = list_amorphous_si_labelled_structures[0].forces

    ground_truth_force_norms = np.linalg.norm(ground_truth_forces, axis=1)

    mapped_flare_calculator = MappedFlareSinglePointCalculator(
        lammps_runner=lammps_runner,
        pair_coeff_file_path=pair_coeff_file_path,
        mapped_uncertainty_file_path=mapped_uncertainty_file_path)

    results = compute_errors_and_uncertainties(mapped_flare_calculator, list_amorphous_si_labelled_structures)

    force_errors = results['all_force_errors']
    uncertainties = results['all_uncertainties']

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle(
        "Active Learning of FLARE on NoOp Amorphous Silicon\nModels applied to initial Amorphous Silicon structure")
    ax1 = fig.add_subplot(111)
    ax1.set_title(rf"Force RMSE = {results['mean_force_rmse']: 3.3f} eV / $\AA$")
    ax1.hexbin(uncertainties, force_errors, gridsize=100, bins="log", cmap="inferno")

    ax1.set_xlabel("Mapped FLARE Uncertainty")
    ax1.set_ylabel(r"Mapped FLARE Force Error (eV / $\AA$)")

    fig.tight_layout()
    fig.savefig(images_dir / "amorphous_flare_decalibrated.png")
    plt.close(fig)

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle("Force Norm Histogram\nSW forces on Initial Amorphous Silicon Structure")
    ax1 = fig.add_subplot(111)
    ax1.hist(ground_truth_force_norms, bins=100, histtype="stepfilled", color='green', alpha=0.75)

    force_rmse = results['mean_force_rmse']
    ax1.vlines(force_rmse, *ax1.get_ylim(), color='red', label='Mean Force RMSE')

    ax1.set_ylabel("SW Force Norms")
    ax1.legend(loc=0)
    ax1.set_xlim(xmin=0)

    fig.tight_layout()

    plt.show()
