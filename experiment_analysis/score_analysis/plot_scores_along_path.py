import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from crystal_diffusion import ANALYSIS_RESULTS_DIR
from crystal_diffusion.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from crystal_diffusion.score.wrapped_gaussian_score import \
    get_sigma_normalized_score
from crystal_diffusion.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from crystal_diffusion.utils.logging_utils import setup_analysis_logger
from crystal_diffusion.utils.tensor_utils import \
    broadcast_batch_tensor_to_all_dimensions

setup_analysis_logger()

logger = logging.getLogger(__name__)

plt.style.use(PLOT_STYLE_PATH)

base_path = Path("/Users/bruno/courtois/scores_along_a_path")

model_name1 = 'diffusion_mace_ode-run7'
data_path1 = base_path / "diffusion_mace_ode_run7_path_scores.pkl"

model_name2 = 'mlp_jun12-run5'
data_path2 = base_path / "mlp_jun12_run5_path_scores.pkl"

output_dir = ANALYSIS_RESULTS_DIR / "scores_along_path"
output_dir.mkdir(exist_ok=True)


x0 = torch.tensor([[0.00, 0.00, 0.00],
                   [0.00, 0.50, 0.50],
                   [0.50, 0.00, 0.50],
                   [0.50, 0.50, 0.00],
                   [0.25, 0.25, 0.25],
                   [0.25, 0.75, 0.75],
                   [0.75, 0.25, 0.75],
                   [0.75, 0.75, 0.25]])


lambda_data = 0.0075  # estimated from the training data.
if __name__ == '__main__':

    data1 = torch.load(data_path1, map_location=torch.device('cpu'))
    data2 = torch.load(data_path2, map_location=torch.device('cpu'))

    logging.info(f"Checking that {data_path1} and {data_path2} are consistent")
    # Double check that the reference data is the same for both models.
    for key in ['epsilon', 'trajectory_sigmas', 'times', 'sigmas']:
        values1 = data1[key]
        values2 = data2[key]
        torch.testing.assert_close(values1, values2)

    epsilon = data1['epsilon']
    trajectory_lambda = data1['trajectory_sigmas']
    sigmas = data1['sigmas']

    number_of_trajectory_points = len(trajectory_lambda)

    normalized_scores1 = data1['normalized_scores']
    normalized_scores2 = data2['normalized_scores']

    logging.info("Computing the x0 targets")
    # xt is different points along a trajectory
    # shape: [batch_size = number_of_trajectory_points, natoms, spatial dimension]
    xt = torch.stack([x0 + lbda * epsilon for lbda in trajectory_lambda])
    # yes, it's silly to add sigma * epsilon only to remove it after, but I think it makes the script clearer.
    delta_relative_coordinates = map_relative_coordinates_to_unit_cell(xt - x0)

    list_target_normalized_scores_x0 = []
    list_target_normalized_scores_x_data = []

    for sigma in sigmas:
        broadcast_sigmas = broadcast_batch_tensor_to_all_dimensions(
            batch_values=sigma * torch.ones(number_of_trajectory_points), final_shape=xt.shape
        )
        target_normalized_scores = get_sigma_normalized_score(delta_relative_coordinates, broadcast_sigmas, kmax=4)
        list_target_normalized_scores_x0.append(target_normalized_scores)

        broadcast_sigmas = broadcast_batch_tensor_to_all_dimensions(
            batch_values=(sigma + lambda_data) * torch.ones(number_of_trajectory_points), final_shape=xt.shape)
        target_normalized_scores = get_sigma_normalized_score(delta_relative_coordinates, broadcast_sigmas, kmax=4)
        list_target_normalized_scores_x_data.append(target_normalized_scores)

    target_normalized_scores_x0 = torch.stack(list_target_normalized_scores_x0)
    target_normalized_scores_x_data = torch.stack(list_target_normalized_scores_x_data)

    logging.info("Generating plots...")
    scores = [target_normalized_scores_x0, target_normalized_scores_x_data, normalized_scores1, normalized_scores2]
    labels = ['${\\bf x}_0$ TARGET', '${\\bf x}_0 +\\lambda_{data} \\epsilon$ TARGET',
              f'Model {model_name1}', f'Model {model_name2}']
    linestyles = ['--', '--', '-', '-']

    for time_idx in tqdm(range(1, len(sigmas)), "time"):
        reference_sigma = sigmas[time_idx]
        list_x = trajectory_lambda / reference_sigma

        on_manifold_range_min = lambda_data / reference_sigma
        on_manifold_range_max = lambda_data / reference_sigma + 2.

        idx_min = torch.where(list_x < on_manifold_range_min)[0][-1]
        try:
            idx_max = torch.where(list_x > on_manifold_range_max)[0][0]
        except KeyError:
            idx_max = len(list_x)
        in_range_mask = torch.zeros_like(list_x, dtype=torch.bool)
        in_range_mask[idx_min:idx_max + 1] = True

        for space_index, component in zip([0, 1, 2], ['X', 'Y', 'Z']):
            for atom_idx in range(8):
                fig1 = plt.figure(figsize=PLEASANT_FIG_SIZE)
                fig1.suptitle(f"Comparing the {component} component of the Score Atom {atom_idx}"
                              f"\n $\\sigma(t)$ = {reference_sigma:4.3f}")

                ax1 = fig1.add_subplot(121)
                ax2 = fig1.add_subplot(122)

                for score, label, ls in zip(scores, labels, linestyles):
                    list_y = score[time_idx, :, atom_idx, space_index]
                    ax1.plot(list_x[in_range_mask], list_y[in_range_mask], ls=ls, label='__nolabel__')
                    ax2.plot(list_x, list_y, ls=ls, label=label)

                ax1.axvspan(on_manifold_range_min, on_manifold_range_max,
                            ymin=0, ymax=1, alpha=0.1, color='gray', label='On Manifold')
                ax2.axvspan(on_manifold_range_min, on_manifold_range_max,
                            ymin=0, ymax=1, alpha=0.1, color='gray', label='__nolabel__')

                ax1.legend(loc=0)
                ax2.legend(loc=0)
                ax1.set_xlim([on_manifold_range_min - 0.1, on_manifold_range_max + 0.1])
                ax2.set_xlim(xmin=0)

                ax1.set_title('Zoom In')
                ax2.set_title('Broad View')
                for ax in [ax1, ax2]:
                    ax.set_xlabel('$\\lambda / \\sigma(t)$')
                    ax.set_ylabel('$\\sigma(t)\\times {\\bf s}( x(\\lambda), \\sigma(t))$')

                fig1.tight_layout()
                fig1.savefig(output_dir / f"path_scores_{component}_atom_idx={atom_idx}_time_idx={time_idx}.png")
                plt.close(fig1)
