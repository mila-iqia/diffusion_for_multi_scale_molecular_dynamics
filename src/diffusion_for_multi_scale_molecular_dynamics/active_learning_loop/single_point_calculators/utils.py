import numpy as np


def compute_errors_and_uncertainties(single_point_calculator, list_labelled_structures):
    """Compute errors and uncertainties over a given dataset with a given single point calculator."""
    # All values for all atoms
    list_all_force_errors = []
    list_all_uncertainties = []

    # Aggregated values per labelled_structure
    list_force_rmse_per_structure = []
    list_energy_errors_per_structure = []

    for labelled_structure in list_labelled_structures:
        result = single_point_calculator.calculate(
            structure=labelled_structure.structure
        )

        force_errors = np.linalg.norm(result.forces - labelled_structure.forces, axis=1)
        list_all_force_errors.append(force_errors)

        list_all_uncertainties.append(result.uncertainties)

        force_rmse = np.sqrt(np.mean(force_errors**2))
        list_force_rmse_per_structure.append(force_rmse)

        energy_error = result.energy - labelled_structure.energy
        list_energy_errors_per_structure.append(energy_error)

    list_all_force_errors = np.concatenate(list_all_force_errors)
    list_all_uncertainties = np.concatenate(list_all_uncertainties)

    list_force_rmse_per_structure = np.array(list_force_rmse_per_structure)
    list_energy_errors_per_structure = np.array(list_energy_errors_per_structure)

    mean_force_rmse = np.mean(list_force_rmse_per_structure)
    energy_rmse = np.sqrt(np.mean(np.array(list_energy_errors_per_structure) ** 2))

    results = dict(
        all_force_errors=list_all_force_errors,
        all_uncertainties=list_all_uncertainties,
        force_rmse_per_structure=list_force_rmse_per_structure,
        energy_error_per_structure=list_energy_errors_per_structure,
        mean_force_rmse=mean_force_rmse,
        energy_rmse=energy_rmse,
    )

    return results
