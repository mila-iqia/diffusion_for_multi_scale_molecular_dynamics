import shutil
import time
from pathlib import Path

import numpy as np
from flare.bffs.sgp import SGP_Wrapper
from flare.bffs.sgp.calculator import SGP_Calculator
from flare_experiment.utilities import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from flare_experiment.utilities.single_point_calculator import (
    MappedFlareSinglePointCalculator, StillingerWeberSinglePointCalculator)
from flare_experiment.utilities.utils import parse_lammps_dump
from matplotlib import pyplot as plt

plt.style.use(PLOT_STYLE_PATH)

#old_version = 1
#threshold = 0.0025

#old_version = 2
#threshold = 0.01

#old_version = 3
#threshold = 0.001

#old_version = 4
#threshold = 0.001

old_version = 5
threshold = 0.0001

new_version = old_version + 1


lammps_executable_path = "/home/user/sources/lammps/build/lmp"

flare_training_data = Path("/home/user/experiments/flare_experiment/training_data/md_training_data")

record_path = Path(f"/home/user/experiments/flare_experiment/recorded_models/")

old_sgp_model_name = f"sgp{old_version}.json"
old_pair_coeff_name =f"lmp{old_version}.flare"
old_map_unc_name = f"map_unc_{old_pair_coeff_name}"

new_sgp_model_name = f"sgp{new_version}.json"
new_pair_coeff_name = f"lmp{new_version}.flare"
new_map_unc_name =  f"map_unc_{new_pair_coeff_name}"

sw_coefficients_file_path = "/home/user/experiments/potentials/Si.sw"

new_data_dump_file_path = f"/home/user/experiments/flare_experiment/artn/trajectory{old_version}/uncertain_dump.yaml"


if __name__ == '__main__':

    parsed_dump_dict = parse_lammps_dump(new_data_dump_file_path)
    atoms = parsed_dump_dict['atoms'][0]
    uncertainties = parsed_dump_dict['uncertainties'][0]

    uncertain_atom_indices = list(np.arange(len(atoms))[uncertainties > threshold])
    print(f"Adding {len(uncertain_atom_indices)} environments to active set")

    sw_calculator = StillingerWeberSinglePointCalculator(lammps_executable_path, sw_coefficients_file_path)
    labelled_structure = sw_calculator.calculate(atoms)

    #old_sgp_model, _ = SGP_Wrapper.from_file(str(record_path / old_sgp_model_name))
    #old_flare_calculator = FlareSinglePointCalculator(old_sgp_model)

    sgp_model, _ = SGP_Wrapper.from_file(str(record_path / old_sgp_model_name))

    # Add the new structure!
    sgp_model.update_db(structure=labelled_structure.atoms,
                        forces=labelled_structure.forces,
                        energy=labelled_structure.energy,
                        mode="specific",
                        update_qr=True,
                        custom_range=uncertain_atom_indices)
    sgp_model.sgp_var = None
    sgp_model.sparse_gp.update_matrices_QR()

    #new_flare_calculator = FlareSinglePointCalculator(sgp_model)

    """
    minimize_options = {"disp": True, "ftol": 1e-8, "gtol": 1e-8, "maxiter": 200}
    trainer = HyperparametersTrainer(method="BFGS", minimize_options=minimize_options)
    optimization_result, nll_values = trainer.train(sgp_model)
    learned_sigma_f = optimization_result.x[2]
    """

    # Map!
    sgp_model.write_model(new_sgp_model_name)
    shutil.move(new_sgp_model_name, str(record_path / new_sgp_model_name))

    # Output the coefficients and write uncertainty file
    version = f"Version {new_version}"
    sgp_model.write_mapping_coefficients(new_pair_coeff_name, version, 0)
    sgp_model.write_varmap_coefficients(new_map_unc_name, version, 0)

    for filename in [new_pair_coeff_name, new_map_unc_name]:
        shutil.move(filename, str(record_path / filename))

    old_mapped_flare_calculator = MappedFlareSinglePointCalculator(lammps_executable_path,
                                                            str(record_path / old_pair_coeff_name),
                                                            str(record_path / old_map_unc_name))

    new_mapped_flare_calculator = MappedFlareSinglePointCalculator(lammps_executable_path,
                                                            str(record_path / new_pair_coeff_name),
                                                            str(record_path / new_map_unc_name))


    fig = plt.figure(figsize=(1.5 * PLEASANT_FIG_SIZE[0], PLEASANT_FIG_SIZE[1]))
    fig.suptitle("Comparing FLARE Errors to Uncertainties Before and After ")
    ax1 = fig.add_subplot(111)

    #list_calculators = [old_flare_calculator, new_flare_calculator]
    list_calculators = [ old_mapped_flare_calculator, new_mapped_flare_calculator ]
    #list_calculators = [old_flare_calculator, new_flare_calculator]

    for color, calculator, label in zip(['red', 'green'], list_calculators, ['Old', 'New']):

        t1 = time.time()
        mapped_flare_results = calculator.calculate(atoms)
        t2 = time.time()
        print(f"MAPPED FLARE time: {t2 - t1:5.3e} seconds")

        force_normed_errors = np.linalg.norm(labelled_structure.forces - mapped_flare_results.forces, axis=1)
        energy_uncertainties = mapped_flare_results.energy_uncertainties
        #force_uncertainties = np.linalg.norm(mapped_flare_results.force_uncertainties, axis=1)

        ax1.plot(energy_uncertainties, force_normed_errors, 'o', alpha=0.5, color=color, label=f"{label} MGP model")
        ax1.legend(loc=0)

    ax1.set_xlabel("Norm of Force Uncertainty for MGP")
    ax1.set_ylabel("Norm of Force Errors for MGP")
    plt.show()
