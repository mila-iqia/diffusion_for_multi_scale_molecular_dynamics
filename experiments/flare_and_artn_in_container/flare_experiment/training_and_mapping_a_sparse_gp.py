import glob
import pickle
import shutil
from pathlib import Path

from flare.bffs.sgp import SGP_Wrapper
from flare.bffs.sgp.sparse_gp import optimize_hyperparameters, compute_negative_likelihood_grad_stable
from flare.bffs.sgp.calculator import SGP_Calculator
from flare.bffs.sgp._C_flare import B2, NormalizedDotProduct, SparseGP, Structure
from matplotlib import pyplot as plt

import numpy as np

from flare_experiment.utilities import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from flare_experiment.utilities.analysis_utils import linear_fit_and_r2
from flare_experiment.utilities.hyperparameter_trainer import HyperparametersTrainer
from flare_experiment.utilities.single_point_calculator import FlareSinglePointCalculator

plt.style.use(PLOT_STYLE_PATH)

flare_training_data = Path("/home/user/experiments/flare_experiment/training_data/md_training_data")

recorded_model_dir = Path("/home/user/experiments/flare_experiment/recorded_models/version0")
recorded_model_dir.mkdir(exist_ok=True, parents=True)

# Define many-body descriptor.
cutoff = 5.0  # A

n_species = 1  # Si
n_radial = 6  # Number of radial basis functions
lmax = 3  # Largest L included in spherical harmonics
radial_basis = "chebyshev"  # Radial basis set
cutoff_name = "quadratic"  # Cutoff function
radial_hyps = [0, cutoff]
cutoff_hyps = []
descriptor_settings = [n_species, n_radial, lmax]

# Define a B2 object.
B2_descriptor = B2(radial_basis, cutoff_name, radial_hyps, cutoff_hyps, descriptor_settings)

# The GP class can take a list of descriptors as input, but here
# we'll use a single descriptor.
descriptor_calculators = [B2_descriptor]

# Define kernel function.
sigma = 2.0
power = 2
dot_product_kernel = NormalizedDotProduct(sigma, power)

# Define the GP hyperparameters
sigma_e = 1.0
sigma_f = 0.01
sigma_s = 0.1
initial_guess = np.array([sigma, sigma_e, sigma_f, sigma_s])


# Define a list of kernels.
# There needs to be one kernel for each descriptor.
kernels = [dot_product_kernel]

# Si == 14
species_numbers_map = {14: 0}

if __name__ == '__main__':

    train_labelled_structures = []
    for pickle_filename in glob.glob(str(flare_training_data / "md_train_structure_*.pkl")):
        with open(pickle_filename, "rb") as fd:
            labelled_structure = pickle.load(fd)
            train_labelled_structures.append(labelled_structure)

    test_labelled_structures = []
    for pickle_filename in glob.glob(str(flare_training_data / "md_test_structure_*.pkl")):
        with open(pickle_filename, "rb") as fd:
            labelled_structure = pickle.load(fd)
            test_labelled_structures.append(labelled_structure)

    sgp_model = SGP_Wrapper(kernels=kernels,
                            descriptor_calculators=descriptor_calculators,
                            cutoff=cutoff,
                            sigma_e=sigma_e,
                            sigma_f=sigma_f,
                            sigma_s=sigma_s,
                            species_map=species_numbers_map,
                            variance_type="DTC",
                            energy_training=True,
                            force_training=True,
                            stress_training=False,
                            single_atom_energies=None, # well defined mechanism to take out atomic reference energies.
                            max_iterations=100,  # max interations of BFGS optimization
                            opt_method="BFGS")
    sgp_model.sparse_gp.Kuu_jitter = 1e-4


    number_of_random_points = 16

    for labelled_structure in train_labelled_structures:
        # Careful! the inputs "atom_indices" and "custom_range" are easily confused.
        sgp_model.update_db(structure=labelled_structure.atoms,
                            forces=labelled_structure.forces,
                            energy=labelled_structure.energy,
                            mode="random",
                            custom_range=[number_of_random_points])

    minimize_options = {"disp": True, "ftol": 1e-8, "gtol": 1e-8, "maxiter": 200}

    trainer = HyperparametersTrainer(method="BFGS", minimize_options=minimize_options)

    optimization_result, nll_values = trainer.train(sgp_model)
    learned_sigma_f = optimization_result.x[2]

    filename = "sgp0.json"
    sgp_model.write_model(filename)
    shutil.move(filename, str(recorded_model_dir / filename))

    # Output the coefficients
    coeff_filename = "lmp0.flare"
    SGP_Calculator(sgp_model, use_mapping=True).build_map(filename=coeff_filename,
                                                          contributor="Version 0",
                                                          map_uncertainty=True)
    uncertainty_filename = f"map_unc_{coeff_filename}"

    destination_dir = Path("/home/user/experiments/flare_experiment/mapped_coefficients/")
    for filename in [coeff_filename, uncertainty_filename]:
        shutil.move(filename, str(recorded_model_dir / filename))

    flare_single_point_calculator = FlareSinglePointCalculator(sgp_model)

    number_of_sparse_points = sgp_model.sparse_gp.n_sparse
    number_of_force_labels = sgp_model.sparse_gp.n_force_labels


    fig = plt.figure(figsize=(PLEASANT_FIG_SIZE[0], 1.2 * PLEASANT_FIG_SIZE[0]))
    fig.suptitle(f"Comparing Actual and Flare SGP results\n "
                 f"|U| ={number_of_sparse_points}, {number_of_force_labels} force labels")
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(212)

    list_ax = [ax1, ax2, ax3, ax4]

    ax1.plot(-nll_values,'bo-')
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Log Likelihood")


    for color, label, labelled_structures in zip(['green', 'red'], ['Train', 'Test'],
                                                 [train_labelled_structures, test_labelled_structures]):
        energy_errors = []
        force_errors = []
        force_uncertainties = []
        for labelled_structure in labelled_structures:
            results = flare_single_point_calculator.calculate(labelled_structure.atoms)
            energy_errors.append(labelled_structure.energy - results.energy)
            force_errors.append(np.linalg.norm(labelled_structure.forces - results.forces, axis=1))
            force_uncertainties.append(np.linalg.norm(results.force_uncertainties, axis=1))

        energy_errors = np.array(energy_errors)
        force_errors = np.array(force_errors).flatten()
        force_uncertainties = np.array(force_uncertainties).flatten()

        ax2.hist(energy_errors, color=color, alpha=0.5, label=label)
        ax3.hist(force_errors, color=color, alpha=0.5, label=label)
        ax4.scatter(force_uncertainties / learned_sigma_f, np.abs(force_errors), color=color, alpha=0.5, label=label)
        if label == 'Train':
            x = force_uncertainties / learned_sigma_f
            y = np.abs(force_errors)
            xfit, yfit, r2 = linear_fit_and_r2(x=x, y=y)

            ax4.plot(xfit, yfit, 'k--', label=f'Linear Fit: R2 = {r2:3.2f}')

    ax2.set_xlabel("Energy Error")
    ax2.set_ylabel("Count")

    ax3.set_xlabel("Force Error")
    ax3.set_ylabel("Count")

    ax4.set_title("$\sigma_f$" + f" = {learned_sigma_f:5.3e}")
    ax4.set_xlabel("Force Uncertainty / $\sigma_f$")
    ax4.set_ylabel("Force Norm Error")
    ax4.set_xlim(xmin=0)
    ax4.set_ylim(ymin=0)

    for ax in list_ax:
        ax.legend(loc=0)

    fig.tight_layout()
    plt.show()
