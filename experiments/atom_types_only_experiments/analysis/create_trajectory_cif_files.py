import numpy as np
import torch

from diffusion_for_multi_scale_molecular_dynamics import TOP_DIR
from diffusion_for_multi_scale_molecular_dynamics.analysis.ovito_utilities.trajectory_io import \
    create_cif_files
from diffusion_for_multi_scale_molecular_dynamics.analysis.sample_trajectory_analyser import \
    SampleTrajectoryAnalyser
from diffusion_for_multi_scale_molecular_dynamics.data.element_types import \
    ElementTypes
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_lattice_parameters_to_unit_cell_vectors
from diffusion_for_multi_scale_molecular_dynamics.utils.logging_utils import \
    setup_analysis_logger

setup_analysis_logger()

base_path = TOP_DIR / "experiments/atom_types_only_experiments/"

exp_path = base_path / "experiments/"
data_path = exp_path / "output/run1/trajectory_samples"
pickle_path = data_path / "trajectories_sample_epoch=99.pt"

visualization_artifacts_path = base_path / "analysis/trajectory_cif_files"
visualization_artifacts_path.mkdir(parents=True, exist_ok=True)

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
    lattice_parameters = trajectory_axl.L

    new_lattice_parameters = torch.from_numpy(lattice_parameters.numpy()[:, reverse_order])

    basis_vectors = map_lattice_parameters_to_unit_cell_vectors(new_lattice_parameters)

    reverse_time_order_trajectory_axl = AXL(A=new_a, X=new_x, L=basis_vectors)

    for trajectory_index in trajectory_indices:
        create_cif_files(
            elements=elements,
            visualization_artifacts_path=visualization_artifacts_path,
            trajectory_index=trajectory_index,
            trajectory_axl_compositions=reverse_time_order_trajectory_axl,
        )
