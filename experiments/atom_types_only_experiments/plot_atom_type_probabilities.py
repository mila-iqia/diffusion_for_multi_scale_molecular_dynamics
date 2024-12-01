import einops
from matplotlib import pyplot as plt
from tqdm import tqdm

from diffusion_for_multi_scale_molecular_dynamics import ROOT_DIR
from diffusion_for_multi_scale_molecular_dynamics.analysis import \
    PLOT_STYLE_PATH
from diffusion_for_multi_scale_molecular_dynamics.analysis.sample_trajectory_analyser import \
    SampleTrajectoryAnalyser
from diffusion_for_multi_scale_molecular_dynamics.data.element_types import \
    ElementTypes
from diffusion_for_multi_scale_molecular_dynamics.utils.d3pm_utils import (
    class_index_to_onehot, get_probability_at_previous_time_step)
from diffusion_for_multi_scale_molecular_dynamics.utils.logging_utils import \
    setup_analysis_logger
from diffusion_for_multi_scale_molecular_dynamics.utils.tensor_utils import \
    broadcast_batch_matrix_tensor_to_all_dimensions

setup_analysis_logger()

plt.style.use(PLOT_STYLE_PATH)

base_path = ROOT_DIR / "../experiments/atom_types_only_experiments/experiments"
data_path = base_path / "output/run1/trajectory_samples"
pickle_path = data_path / "trajectories_sample_epoch=999.pt"

elements = ["Si", "Ge"]

element_types = ElementTypes(elements)

num_classes = len(elements) + 1
if __name__ == "__main__":

    analyser = SampleTrajectoryAnalyser(pickle_path, num_classes=num_classes)

    time_indices, predictions_axl = analyser.extract_axl(axl_key="model_predictions_i")
    _, composition_axl = analyser.extract_axl(axl_key="composition_i")

    nsamples, ntimes, natoms = composition_axl.A.shape

    batched_predictions = einops.rearrange(
        predictions_axl.A, "samples time ... -> (samples time) ..."
    )
    batched_at = einops.rearrange(
        composition_axl.A, "samples time ... -> (samples time) ..."
    )
    batched_at_onehot = class_index_to_onehot(batched_at, num_classes=num_classes)

    final_shape = (ntimes, nsamples, natoms)

    q_matrices = broadcast_batch_matrix_tensor_to_all_dimensions(
        batch_values=analyser.noise.q_matrix, final_shape=final_shape
    )
    batched_q_matrices = einops.rearrange(
        q_matrices, "times samples ... -> (samples times) ..."
    )

    q_bar_matrices = broadcast_batch_matrix_tensor_to_all_dimensions(
        batch_values=analyser.noise.q_bar_matrix, final_shape=final_shape
    )
    batched_q_bar_matrices = einops.rearrange(
        q_bar_matrices, "times samples ... -> (samples times) ..."
    )

    q_bar_tm1_matrices = broadcast_batch_matrix_tensor_to_all_dimensions(
        batch_values=analyser.noise.q_bar_tm1_matrix, final_shape=final_shape
    )
    batched_q_bar_tm1_matrices = einops.rearrange(
        q_bar_tm1_matrices, "times samples ... -> (samples times) ..."
    )

    batched_probabilities = get_probability_at_previous_time_step(
        batched_predictions,
        batched_at_onehot,
        batched_q_matrices,
        batched_q_bar_matrices,
        batched_q_bar_tm1_matrices,
        small_epsilon=1.0e-12,
        probability_at_zeroth_timestep_are_logits=True,
    )

    probabilities = einops.rearrange(
        batched_probabilities,
        "(samples times) ... -> samples times ...",
        samples=nsamples,
        times=ntimes,
    )

    raw_probabilities = einops.rearrange(
        batched_predictions.softmax(dim=-1),
        "(samples times) ... -> samples times ...",
        samples=nsamples,
        times=ntimes,
    )

    output_dir = base_path / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    masked_atom_type = num_classes - 1

    list_colors = ["green", "blue", "red"]
    list_elements = []
    list_element_idx = []

    for element_id in element_types.element_ids:
        element_types.get_element(element_id)
        list_elements.append(element_types.get_element(element_id))
        list_element_idx.append(element_id)

    list_elements.append("MASK")
    list_element_idx.append(masked_atom_type)

    for traj_idx in tqdm(range(10), "TRAJ"):

        fig = plt.figure(figsize=(14.4, 6.6))

        fig.suptitle("Prediction Probability")
        ax1 = fig.add_subplot(241)
        ax2 = fig.add_subplot(242)
        ax3 = fig.add_subplot(243)
        ax4 = fig.add_subplot(244)
        ax5 = fig.add_subplot(245)
        ax6 = fig.add_subplot(246)
        ax7 = fig.add_subplot(247)
        ax8 = fig.add_subplot(248)
        list_ax = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]

        for atom_idx, ax in enumerate(list_ax):
            ax.set_title(f"Atom {atom_idx}")

            mask = composition_axl.A[traj_idx, :, atom_idx] == masked_atom_type
            unmask_time = time_indices[mask].min()

            ax.vlines(unmask_time, -0.1, 1.1, lw=2, color="k", label="Unmasking Time")
            list_elements.append("MASK")
            list_element_idx.append(masked_atom_type)

            for element_idx, element, color in zip(
                list_element_idx, list_elements, list_colors
            ):
                p = probabilities[traj_idx, :, atom_idx, element_idx]
                ax.semilogy(time_indices, p, c=color, label=f"{element}", alpha=0.5)

            for element_idx, element, color in zip(
                list_element_idx[:-1], list_elements[:-1], list_colors[:-1]
            ):
                raw_p = raw_probabilities[traj_idx, :, atom_idx, element_idx]
                ax.semilogy(
                    time_indices,
                    raw_p,
                    "--",
                    lw=2,
                    c=color,
                    label=f"RAW {element}",
                    alpha=0.25,
                )

            ax.set_xlabel("Time Index")
            ax.set_ylabel("Probability")
            ax.set_xlim(time_indices[-1], time_indices[0])

        ax1.legend(loc=0)
        fig.tight_layout()
        fig.savefig(output_dir / f"traj_{traj_idx}.png")
        plt.close(fig)
