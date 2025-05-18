import glob
import pickle
from copy import copy
from pathlib import Path

import numpy as np
from flare.bffs.sgp import SGP_Wrapper
from flare.bffs.sgp._C_flare import B2, NormalizedDotProduct
from flare.bffs.sgp.calculator import SGP_Calculator
from flare.bffs.sgp.sparse_gp import (compute_negative_likelihood,
                                      compute_negative_likelihood_grad_stable,
                                      optimize_hyperparameters)
from matplotlib import pyplot as plt

from flare_experiment.utilities import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from flare_experiment.utilities.hyperparameter_trainer import \
    HyperparametersTrainer

plt.style.use(PLOT_STYLE_PATH)


flare_training_data = Path("/home/user/experiments/flare_experiment/training_data/md_training_data")

# Define many-body descriptor.
cutoff = 5.0  # A

n_species = 1  # Si
n_radial = 6 # Number of radial basis functions
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

    for pickle_filename in glob.glob(str(flare_training_data / "*.pkl")):
        with open(pickle_filename, "rb") as fd:
            labelled_structure = pickle.load(fd)

        # Careful! the inputs "atom_indices" and "custom_range" are easily confused.
        sgp_model.update_db(structure=labelled_structure.atoms,
                            forces=labelled_structure.forces,
                            energy=labelled_structure.energy,
                            mode="random",
                            custom_range=[4])
    sgp_model.sparse_gp.Kuu_jitter = 1e-4
    minimize_options = {"disp": True, "ftol": 1e-8, "gtol": 1e-8, "maxiter": 200}

    trainer = HyperparametersTrainer(method="BFGS", minimize_options=minimize_options)

    optimization_result, nll_values = trainer.train(sgp_model)

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle("Evolution of NLL During minimization")
    ax1 = fig.add_subplot(111)
    ax1.plot(nll_values, '-o')
    ax1.set_ylabel('NLL')
    ax1.set_xlabel('step')
    fig.tight_layout()
    plt.show()

    #================================================================================
    optimized_hyperparameters = optimization_result.x

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle("NLL Around Equilibrium Point")
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    list_ax = [ax1, ax2, ax3, ax4]
    list_labels = ['sigma', 'sigma_e', 'sigma_f', 'sigma_s']

    for idx, (ax, label) in enumerate(zip(list_ax, list_labels)):

        list_x = np.linspace(0.5 * optimized_hyperparameters[idx],
                             1.5 * optimized_hyperparameters[idx], 11)
        list_nll = []

        for x in list_x:
            hyperparameters = copy(optimized_hyperparameters)
            hyperparameters[idx] = x
            nll = compute_negative_likelihood(hyperparameters,
                                              sgp_model.sparse_gp,
                                              print_vals=True)
            list_nll.append(nll)

        ax.plot(list_x, list_nll, '-o', label=label)
        ax.set_ylabel('NLL')
        ax.set_xlabel(label)

        ax.vlines(optimized_hyperparameters[idx], *ax.set_ylim(), label=f'Optimal {label}')

    fig.tight_layout()
    plt.show()

