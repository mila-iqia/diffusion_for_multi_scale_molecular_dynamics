from pathlib import Path
import pickle

import numpy as np

from maml.apps.pes import MTPotential
from matplotlib import pyplot as plt
from pymatgen.core import Structure

from src.diffusion_for_multi_scale_molecular_dynamics.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH

plt.style.use(PLOT_STYLE_PATH)


def get_positions(structure: Structure):
    positions = []
    for site in structure:
        positions.append(site.coords)

    return np.array(positions)


def get_nbh_grade(nbh_file_path: str) -> np.ndarray:
    with open(nbh_file_path, "r")  as fd:
            lines = fd.readlines()

    read_nbh = False
    list_gamma = []
    for line in lines:
        if 'Energy' in line:
            break

        if read_nbh:
            value = float(line.replace('\n', '').split()[-1])
            list_gamma.append(value)

        if 'AtomData' in line:
            read_nbh = True

    list_gamma = np.array(list_gamma)

    return list_gamma


level = 12
ground_truth_data_directory = Path("/home/user/diffusion_for_multi_scale_molecular_dynamics/experiments/sanity_checking_MTP/mtp_experiments/ground_truth_data")
mtp_experiment_path = Path(f"/home/user/diffusion_for_multi_scale_molecular_dynamics/experiments/sanity_checking_MTP/mtp_experiments/level_{level}")
mtp_experiment_path.mkdir(exist_ok=True)



if __name__ == '__main__':

    mtp_potential = MTPotential()
    mtp_potential.elements = ['Si']

    list_round_indices = [1, 2 , 3, 4, 5, 6, 7]

    for round_index in list_round_indices:

        data_file_path = str(ground_truth_data_directory / f"data_{round_index}.pkl")

        # To verify, you can also load it back
        with open(data_file_path, 'rb') as f:
            loaded_data = pickle.load(f)

        gt_structure = loaded_data['structure']
        gt_forces = loaded_data['forces']
        gt_energy = loaded_data['energy']

        fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
        fig.suptitle(f"Comparing MTP Level {level} Errors to Uncertainties Before and After for Round {round_index}")
        ax1 = fig.add_subplot(111)

        uncertain_indices = None
        list_x_src = None
        list_y_src = None
        list_x_dst = None
        list_y_dst = None


        for color, label in zip(['red', 'green'], ['before', 'after']):
            output_file_path = str(mtp_experiment_path / f"eval_{round_index}_{label}.cfg.0")
            nbh_file_path = str(mtp_experiment_path / f'eval_{round_index}_nbh_grade_{label}.cfg.0')

            list_outputs, df = mtp_potential.read_cfgs(output_file_path)

            data_dict = list_outputs[0]
            pred_structure = Structure.from_dict(data_dict['structure'])

            position_error = np.linalg.norm(get_positions(gt_structure) - get_positions(pred_structure))
            assert position_error < 1.0e-5

            outputs = data_dict['outputs']

            pred_energy = outputs['energy']
            pred_forces = np.array(outputs['forces'])
            gamma_uncertainties = get_nbh_grade(nbh_file_path)

            force_normed_errors = np.linalg.norm(gt_forces - pred_forces, axis=1)

            if uncertain_indices is None:
                uncertain_indices = np.argsort(gamma_uncertainties)[::-1][:4]
                list_x_src = gamma_uncertainties[uncertain_indices]
                list_y_src = force_normed_errors[uncertain_indices]
            else:
                list_x_dst = gamma_uncertainties[uncertain_indices]
                list_y_dst = force_normed_errors[uncertain_indices]

            ax1.semilogy(gamma_uncertainties, force_normed_errors, 'o', alpha=0.5, color=color, label=f"{label}")

            ax1.plot(gamma_uncertainties[uncertain_indices], force_normed_errors[uncertain_indices],
                     '*', color=color, ms=15, alpha=1.0, label=f'{label} Most Uncertain')

        for x1, y1, x2, y2 in zip(list_x_src, list_y_src, list_x_dst, list_y_dst):
            ax1.plot([x1, x2], [y1, y2], 'k-', label='__nolabel__')


        ax1.legend(loc=0)
        ax1.set_xlabel("Gamma nbh Uncertainty")
        ax1.set_ylabel("Norm of Force Errors")
        plt.show()

