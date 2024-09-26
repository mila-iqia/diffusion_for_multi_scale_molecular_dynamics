from crystal_diffusion.utils.ovito_utils import (create_cif_files,
                                                 create_ovito_session_state)
from sanity_checks.visualizing_fake_trajectories_with_ovito import \
    VISUALIZATION_SANITY_CHECK_DIRECTORY
from sanity_checks.visualizing_fake_trajectories_with_ovito.fake_data_utils import \
    generate_fake_trajectories_pickle

acell = 5
number_of_atoms = 8
number_of_frames = 101  # the 'time' dimension
number_of_trajectories = 4  # the 'batch' dimension
if __name__ == '__main__':

    pickle_path = VISUALIZATION_SANITY_CHECK_DIRECTORY / "trajectories.pt"

    generate_fake_trajectories_pickle(acell=acell,
                                      number_of_atoms=number_of_atoms,
                                      number_of_frames=number_of_frames,
                                      number_of_trajectories=number_of_trajectories,
                                      pickle_path=pickle_path)

    trajectory_directory = VISUALIZATION_SANITY_CHECK_DIRECTORY / "trajectories"
    for trj_idx in range(number_of_trajectories):
        print(f"Computing Ovito trajectory session state for trajectory index {trj_idx}")
        create_cif_files(visualization_artifacts_path=trajectory_directory,
                         trajectory_index=trj_idx,
                         ode_trajectory_pickle=pickle_path)

        create_ovito_session_state(visualization_artifacts_path=trajectory_directory,
                                   trajectory_index=trj_idx)
