"""Position to cif files.

A simple script to extract the diffusion positions from a pickle on disk and output
in cif format for visualization.
"""

from pymatgen.core import Lattice, Structure

from crystal_diffusion import TOP_DIR
from crystal_diffusion.utils.sample_trajectory import \
    PredictorCorrectorSampleTrajectory

# Hard coding some paths to local results. Modify as needed...
epoch = 35
sample_idx = 0
experiment_name = "mace_plus_prediction_head/run2"
# experiment_name = "mlp/run1"

trajectory_top_output_directory = TOP_DIR / "experiments/diffusion_mace_harmonic_data/output/"

trajectory_data_directory = trajectory_top_output_directory / experiment_name / "diffusion_position_samples"

output_top_dir = trajectory_data_directory.parent / "visualization"
output_dir = output_top_dir / f"visualise_sampling_trajectory_epoch_{epoch}_sample_{sample_idx}"
output_dir.mkdir(exist_ok=True, parents=True)

if __name__ == '__main__':
    pickle_path = trajectory_data_directory / f"diffusion_position_sample_epoch={epoch}.pt"
    sample_trajectory = PredictorCorrectorSampleTrajectory.read_from_pickle(pickle_path)

    pickle_path = trajectory_data_directory / f"diffusion_position_sample_epoch={epoch}.pt"
    sample_trajectory = PredictorCorrectorSampleTrajectory.read_from_pickle(pickle_path)

    basis_vectors = sample_trajectory.data['unit_cell'][sample_idx].numpy()
    lattice = Lattice(matrix=basis_vectors, pbc=(True, True, True))

    list_predictor_coordinates = sample_trajectory.data['predictor_x_i']

    for idx, sample_predictor_coordinates in enumerate(list_predictor_coordinates):
        coordinates = sample_predictor_coordinates[sample_idx].numpy()
        number_of_atoms = coordinates.shape[0]
        species = number_of_atoms * ['Si']

        structure = Structure(lattice=lattice,
                              species=species,
                              coords=coordinates,
                              coords_are_cartesian=False)

        file_path = str(output_dir / f"diffusion_positions_{idx}.cif")
        structure.to_file(file_path)
