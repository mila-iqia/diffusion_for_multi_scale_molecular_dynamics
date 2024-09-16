import logging
from pathlib import Path

from crystal_diffusion.utils.logging_utils import setup_analysis_logger
from crystal_diffusion.utils.ovito_utils import (create_cif_files,
                                                 create_ovito_session_state)

logger = logging.getLogger(__name__)

setup_analysis_logger()

results_dir = Path(
    "/Users/bruno/courtois/partial_trajectory_sampling_EGNN_sept_10/partial_samples_EGNN_Sept_10"
)
number_of_trajectories = 32  # the 'batch' dimension

reference_cif_file = results_dir / "reference_validation_structure_Hydrogen.cif"

list_sample_times = [1.0]
if __name__ == "__main__":
    for sample_time in list_sample_times:
        logger.info(f"Processing sample time = {sample_time}")
        pickle_path = (
            results_dir / f"diffusion_position_sample_time={sample_time:4.3f}.pt"
        )

        trajectory_directory = (
            results_dir / f"trajectories_sample_time={sample_time:4.3f}"
        )
        for trj_idx in range(number_of_trajectories):
            logger.info(
                f" - Computing Ovito trajectory session state for trajectory index {trj_idx}"
            )
            create_cif_files(
                visualization_artifacts_path=trajectory_directory,
                trajectory_index=trj_idx,
                ode_trajectory_pickle=pickle_path,
            )

            create_ovito_session_state(
                visualization_artifacts_path=trajectory_directory,
                trajectory_index=trj_idx,
                reference_cif_file=reference_cif_file,
            )
