"""From a .pt file with a list of AXL to .xyz files readable by ovito."""

from pathlib import Path

import torch

from diffusion_for_multi_scale_molecular_dynamics.analysis.ovito_utilities.trajectory_io import \
    create_xyz_files
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL

TRAJECTORY_PATH = (
    "/Users/simonblackburn/projects/courtois2024/experiments/score_on_a_path"
)
TRAJECTORY_PATH = Path(TRAJECTORY_PATH)

SIGMA_INDEX = 0


def main():
    """Read a trajectory file and create XYZ files formatted for OVITO."""
    trajectory_file = TRAJECTORY_PATH / "interpolated_positions.pt"

    axls = torch.load(trajectory_file, map_location="cpu")

    axls_tensors = AXL(
        A=torch.stack([axl.A for axl in axls]),
        X=torch.stack([axl.X for axl in axls]),
        L=torch.stack([axl.L for axl in axls]),
    )

    model_predictions_file = TRAJECTORY_PATH / "model_predictions.pt"
    model_predictions = torch.load(model_predictions_file, map_location="cpu")
    selected_sigma = model_predictions["sigma"][SIGMA_INDEX]
    score_axls = model_predictions["model_predictions"]
    sigma_normalized_score = torch.stack(
        [axl.X[SIGMA_INDEX, :, :] for axl in score_axls]
    )  # time, n_atom, spatial_dimension

    atom_divergence = model_predictions["jacobians"][:, SIGMA_INDEX, :, :]
    atom_divergence = (
        torch.diagonal(atom_divergence, dim1=1, dim2=2)
        .view(atom_divergence.shape[0], -1, sigma_normalized_score.shape[-1])
        .sum(dim=-1, keepdim=True)
    )  # num_sample, num_atom, 1

    atomic_properties = dict(
        sigma_normalized_score=sigma_normalized_score,
        atomic_divergence=atom_divergence,
    )

    output_path = TRAJECTORY_PATH / f"ovito_visualization_sigma_{selected_sigma:0.4f}"
    output_path.mkdir(parents=True, exist_ok=True)

    create_xyz_files(
        elements=["Si"],
        visualization_artifacts_path=output_path,
        trajectory_index=None,
        trajectory_axl_compositions=axls_tensors,
        atomic_properties=atomic_properties,
    )


if __name__ == "__main__":
    main()
