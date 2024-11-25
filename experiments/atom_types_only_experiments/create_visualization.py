import numpy as np
import torch

from diffusion_for_multi_scale_molecular_dynamics import ROOT_DIR
from diffusion_for_multi_scale_molecular_dynamics.analysis.sample_trajectory_analyser import \
    SampleTrajectoryAnalyser
from diffusion_for_multi_scale_molecular_dynamics.data.element_types import \
    ElementTypes
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.utils.logging_utils import \
    setup_analysis_logger
from diffusion_for_multi_scale_molecular_dynamics.utils.ovito_utils import \
    create_cif_files

setup_analysis_logger()

base_path = ROOT_DIR / "../experiments/atom_types_only_experiments/experiments"
data_path = base_path / "output/run1/trajectory_samples"
pickle_path = data_path / "trajectories_sample_epoch=99.pt"
visualization_artifacts_path = data_path / "trajectory_cif_files"

elements = ["Si", "Ge"]
num_classes = len(elements) + 1
element_types = ElementTypes(elements)

trajectory_indices = np.arange(10)


if __name__ == "__main__":

    analyser = SampleTrajectoryAnalyser(pickle_path, num_classes)
    time_indices, trajectory_axl = analyser.extract_axl("composition_i")

    reverse_order = np.argsort(time_indices)[::-1]

    # Torch can't deal with indices in reverse order
    a = trajectory_axl.A
    new_a = torch.from_numpy(a.numpy()[:, reverse_order])
    x = trajectory_axl.X
    new_x = torch.from_numpy(x.numpy()[:, reverse_order])
    lattice = trajectory_axl.L
    new_l = torch.from_numpy(lattice.numpy()[:, reverse_order])

    reverse_time_order_trajectory_axl = AXL(A=new_a, X=new_x, L=new_l)

    for trajectory_index in trajectory_indices:
        create_cif_files(
            elements=elements,
            visualization_artifacts_path=visualization_artifacts_path,
            trajectory_index=trajectory_index,
            trajectory_axl_compositions=reverse_time_order_trajectory_axl,
        )
