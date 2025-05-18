import glob
import pickle
import time
from pathlib import Path

import numpy as np
from flare.bffs.sgp import SGP_Wrapper
from matplotlib import pyplot as plt

from flare_experiment.utilities import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from flare_experiment.utilities.analysis_utils import linear_fit_and_r2
from flare_experiment.utilities.single_point_calculator import (
    FlareSinglePointCalculator, MappedFlareSinglePointCalculator,
    StillingerWeberSinglePointCalculator)

plt.style.use(PLOT_STYLE_PATH)

lammps_executable_path = "/home/user/sources/lammps/build/lmp"


flare_training_data = Path("/home/user/experiments/flare_experiment/training_data/md_training_data")

record_path = Path("/home/user/experiments/flare_experiment/recorded_models/version0/")

sgp_model_path = str(record_path / "sgp0.json")
pair_coeff_path = str(record_path / "lmp0.flare")
map_unc_path = str(record_path / "map_unc_lmp0.flare")

sw_coefficients_file_path = "/home/user/experiments/potentials/Si.sw"

if __name__ == '__main__':


    # The uncertainty calculation is a complete mess. It is very unclear what exactly is being calculated.
    # It seems like the SGP with DTC calculates uncertainty on forces, whereas the mapped SGP computes uncertainties
    # on unknowable local energies. Let's plot these things...

    sgp_model, _ = SGP_Wrapper.from_file(sgp_model_path)
    #_, sigma_e, sigma_f, _ = sgp_model.hyps  # THIS LINE LEADS TO A SEGFAULT!!!
    #_, sigma_e, sigma_f, _ = copy(sgp_model.sparse_gp.hyperparameters) # This also creates a seg fault! WTF!

    flare_calculator = FlareSinglePointCalculator(sgp_model)

    mapped_flare_calculator = MappedFlareSinglePointCalculator(lammps_executable_path, pair_coeff_path, map_unc_path)

    sw_calculator = StillingerWeberSinglePointCalculator(lammps_executable_path, sw_coefficients_file_path)

    flare_unc = []
    mapped_flare_unc = []

    energy_errors = []
    force_normed_errors = []

    mapping_energy_differences = []
    mapping_force_differences = []

    idx = 0
    for pickle_filename in glob.glob(str(flare_training_data / "*.pkl")):
        with open(pickle_filename, "rb") as fd:
            labelled_structure = pickle.load(fd)

        idx += 1
        print(idx)

        atoms = labelled_structure.atoms
        t1 = time.time()
        sw_results = sw_calculator.calculate(atoms)
        t2 = time.time()
        print(f"SW time: {t2 -t1:5.3e} seconds")

        t1 = time.time()
        flare_results = flare_calculator.calculate(atoms)
        t2 = time.time()
        print(f"FLARE time: {t2 -t1:5.3e} seconds")

        t1 = time.time()
        mapped_flare_results = mapped_flare_calculator.calculate(atoms)
        t2 = time.time()
        print(f"MAPPED FLARE time: {t2 -t1:5.3e} seconds")

        norm_force_uncertainties = np.linalg.norm(flare_results.force_uncertainties, axis=1)
        flare_unc.append(norm_force_uncertainties)
        mapped_flare_unc.append(mapped_flare_results.energy_uncertainties)

        energy_errors.append(sw_results.energy - flare_results.energy)
        force_normed_errors.append(np.linalg.norm(sw_results.forces - flare_results.forces, axis=1))

        mapping_energy_differences.append(flare_results.energy - mapped_flare_results.energy)
        mapping_force_differences.append(flare_results.forces - mapped_flare_results.forces)

    flare_force_unc = np.array(flare_unc).flatten()
    mapped_flare_energy_unc = np.array(mapped_flare_unc).flatten()

    mapping_energy_differences = np.array(mapping_energy_differences).flatten()
    mapping_force_differences = np.array(mapping_force_differences).flatten()

    energy_errors = np.array(energy_errors)
    force_normed_errors = np.array(force_normed_errors).flatten()

    fig0 = plt.figure(figsize=(1.5 * PLEASANT_FIG_SIZE[0], PLEASANT_FIG_SIZE[1]))
    fig0.suptitle("Comparing FLARE Errors to Uncertainties")
    ax01 = fig0.add_subplot(121)
    ax02 = fig0.add_subplot(122)

    ax01.plot(flare_force_unc, force_normed_errors, 'go', label='Data')
    xfit, yfit, r2 = linear_fit_and_r2(x=flare_force_unc, y=force_normed_errors)
    ax01.plot(xfit, yfit, 'r--', label=f'Linear Fit: R2 = {r2:4.2f}')
    ax01.legend(loc=0)
    ax01.set_xlabel("Norm of Force Uncertainty for SGP")
    ax01.set_ylabel("Norm of Force Errors for SGP")

    ax02.plot(mapped_flare_energy_unc, force_normed_errors, 'go', label='Data')
    xfit, yfit, r2 = linear_fit_and_r2(x=mapped_flare_energy_unc, y=force_normed_errors)
    ax02.plot(xfit, yfit, 'r--', label=f'Linear Fit: R2 = {r2:4.2f}')
    ax02.legend(loc=0)
    ax02.set_xlabel("Local Energy Uncertainty for MGP")
    ax02.set_ylabel("Norm of Force Errors for SGP")

    for ax in [ax01, ax02]:
        ax.set_xlim(xmin=0)
        ax.set_ylim(ymin=0)

    fig0.tight_layout()
    plt.show()

    fig1 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig1.suptitle("Comparing FLARE's SGP and the Map to LAMMPS")
    ax1 = fig1.add_subplot(111)

    ax1.plot(flare_force_unc, mapped_flare_energy_unc, 'go', label='Data')
    xfit, yfit, r2 = linear_fit_and_r2(x=flare_force_unc, y=mapped_flare_energy_unc)
    ax1.plot(xfit, yfit, 'r--', label=f'Linear Fit: R2 = {r2:4.2f}')
    ax1.legend(loc=0)
    ax1.set_xlabel("Norm of Force Uncertainty for SGP")
    ax1.set_ylabel("Energy Uncertainty for Mapped GP")
    ax1.set_xlim(xmin=0)
    ax1.set_ylim(ymin=0)

    fig1.tight_layout()
    plt.show()

    fig2 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig2.suptitle("Comparing FLARE's SGP and the Map to LAMMPS")
    ax2 = fig2.add_subplot(211)
    ax3 = fig2.add_subplot(212)

    ax2.set_xlabel("Total Energy Difference between Mapped vs Not Mapped")
    ax3.set_xlabel("Forces Difference between Mapped vs Not Mapped")

    ax2.hist(mapping_energy_differences)
    ax3.hist(mapping_force_differences)


    fig2.tight_layout()
    plt.show()
