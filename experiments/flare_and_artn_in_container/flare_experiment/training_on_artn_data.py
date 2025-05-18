from pathlib import Path

import numpy as np
from flare.bffs.sgp import SGP_Wrapper
from flare.bffs.sgp._C_flare import B2, NormalizedDotProduct
from flare.bffs.sgp.calculator import SGP_Calculator
from flare.bffs.sgp.sparse_gp import optimize_hyperparameters, compute_negative_likelihood_grad_stable
from matplotlib import pyplot as plt

from flare_experiment.utilities import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from flare_experiment.utilities.labelled_structure import LabelledStructure
from flare_experiment.utilities.single_point_calculator import FlareSinglePointCalculator
from flare_experiment.utilities.utils import parse_lammps_dump

plt.style.use(PLOT_STYLE_PATH)


exact_dump_path = "/home/user/experiments/tutorials/Si-vac.LAMMPS.d/dump.yaml"


recorded_model_dir = Path("/home/user/experiments/flare_experiment/recorded_models/cheating")
recorded_model_dir.mkdir(exist_ok=True, parents=True)

# Define many-body descriptor.
cutoff = 4.0  # A

n_species = 1  # Si
n_radial = 4  # Number of radial basis functions
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
sigma = 1.0
power = 2
dot_product_kernel = NormalizedDotProduct(sigma, power)

# Define the GP hyperparameters
sigma_e = 0.01
sigma_f = 0.001
sigma_s = 0.1
initial_guess = np.array([sigma, sigma_e, sigma_f, sigma_s])


# Define a list of kernels.
# There needs to be one kernel for each descriptor.
kernels = [dot_product_kernel]

# Si == 14
species_numbers_map = {14: 0}
#variance_type = 'DTC'
variance_type="local"

if __name__ == '__main__':

    results = parse_lammps_dump(exact_dump_path)

    list_labelled_structures = []
    for atoms, forces, energy in zip(results['atoms'], results['forces'], results['energy']):
        list_labelled_structures.append(LabelledStructure(atoms=atoms,
                                                          forces=forces,
                                                          energy=energy,
                                                          active_set_indices=np.arange(len(forces))))

    sgp_model = SGP_Wrapper(kernels=kernels,
                            descriptor_calculators=descriptor_calculators,
                            cutoff=cutoff,
                            sigma_e=sigma_e,
                            sigma_f=sigma_f,
                            sigma_s=sigma_s,
                            species_map=species_numbers_map,
                            variance_type=variance_type,
                            energy_training=True,
                            force_training=True,
                            stress_training=False,
                            single_atom_energies=None, # well defined mechanism to take out atomic reference energies.
                            max_iterations=100,  # max interations of BFGS optimization
                            opt_method="BFGS")
    sgp_model.sparse_gp.Kuu_jitter = 1e-8

    # Add a single structure completely
    labelled_structure = list_labelled_structures[0]

    sgp_model.update_db(structure=labelled_structure.atoms,
                        forces=labelled_structure.forces,
                        energy=labelled_structure.energy,
                        mode="all")


    flare_single_point_calculator = FlareSinglePointCalculator(sgp_model)


    for idx, labelled_structure in enumerate(list_labelled_structures[1:8]):

        reference_positions = labelled_structure.atoms.positions

        print("Compute Before")
        before_results = flare_single_point_calculator.calculate(atoms=labelled_structure.atoms)
        error = np.linalg.norm(before_results.atoms.positions - reference_positions)
        assert error < 1.0e-5

        before_force_errors = np.linalg.norm(labelled_structure.forces - before_results.forces, axis=1)
        if variance_type == 'local':
            before_uncertainties = before_results.energy_uncertainties
        else:
            before_uncertainties = np.linalg.norm(before_results.force_uncertainties,axis=1)

        print("Adding to DB and retraining")
        # Careful! the inputs "atom_indices" and "custom_range" are easily confused.
        sgp_model.update_db(structure=labelled_structure.atoms,
                            forces=labelled_structure.forces,
                            energy=labelled_structure.energy,
                            mode="uncertain",
                            custom_range=[8]
                            )

        # Capture the indices that were actually added
        added_indices = sgp_model.sparse_gp.sparse_indices[0][-1]

        after_results = flare_single_point_calculator.calculate(atoms=labelled_structure.atoms)
        error = np.linalg.norm(after_results.atoms.positions - reference_positions)
        assert error < 1.0e-5

        after_force_errors = np.linalg.norm(labelled_structure.forces - after_results.forces, axis=1)
        after_uncertainties = after_results.energy_uncertainties
        if variance_type == 'local':
            after_uncertainties = after_results.energy_uncertainties
        else:
            after_uncertainties = np.linalg.norm(after_results.force_uncertainties,axis=1)


        fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
        fig.suptitle(f"{variance_type} : Comparing Before and After adding to training set: run {idx}")
        ax1 = fig.add_subplot(111)


        ax1.plot(before_uncertainties, before_force_errors, 'ro', alpha=0.25, label='Before')
        ax1.plot(after_uncertainties, after_force_errors, 'go', alpha=0.25, label='After')

        ax1.plot(before_uncertainties[added_indices], before_force_errors[added_indices],
                 'r*', ms=15, alpha=1.0, label='Before Most Uncertain')
        ax1.plot(after_uncertainties[added_indices], after_force_errors[added_indices],
                 'g*', ms=15, alpha=1.0, label='After Most Uncertain')

        list_x = before_uncertainties[added_indices]
        list_y = before_force_errors[added_indices]
        list_dx = after_uncertainties[added_indices] - before_uncertainties[added_indices]
        list_dy = after_force_errors[added_indices] - before_force_errors[added_indices]

        for x, y, dx, dy in zip(list_x, list_y, list_dx, list_dy):
            ax1.plot([x, x+dx], [y, y+dy], 'k-', label='__nolabel__')

        ax1.set_xlabel(f"Atomwise Uncertainty ({variance_type})")
        ax1.set_ylabel("Norm of Force Errors (eV/A)")
        ax1.set_xlim(xmin=0)
        ax1.set_ylim(ymin=0)
        ax1.legend(loc=0)
        plt.show()
