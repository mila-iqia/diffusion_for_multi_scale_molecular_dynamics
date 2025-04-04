import logging

from utils import RESULTS_DIR
from utils.analysis_utils import (InputParameters, create_samples,
                                  get_checkpoint_path, get_vector_field_movie,
                                  plot_marginal_distribution, plot_samples,
                                  plot_samples_radial_distribution)

logger = logging.getLogger(__name__)


def analyse(experiment_name: str, run_name: str):
    """Conduct all the analyses for a given experiment."""
    logger.info(f"Starting analysis of experiment {experiment_name} / {run_name}")

    checkpoint_path = get_checkpoint_path(experiment_name, run_name)

    logger.info(f"  checkpoint is  {checkpoint_path}")

    input_parameters = InputParameters(algorithm="predictor_corrector",
                                       total_time_steps=50,
                                       number_of_corrector_steps=2,
                                       corrector_step_epsilon=2.5e-8,
                                       number_of_samples=10_000,
                                       record_samples=False)

    plot_marginal_distribution(input_parameters, output_directory=RESULTS_DIR)

    logger.info("  Creating samples...")
    output_samples_path = RESULTS_DIR / experiment_name / run_name / "samples"
    create_samples(input_parameters,
                   output_directory=str(output_samples_path),
                   checkpoint_path=checkpoint_path
                   )

    logger.info("  Plotting samples...")
    plot_samples(output_samples_path, experiment_name)

    plot_samples_radial_distribution(output_samples_path, experiment_name)

    logger.info("  Generate vector field videos...")
    output_video_path = RESULTS_DIR / experiment_name / run_name / "score_network_vector_field.mp4"
    get_vector_field_movie(input_parameters, checkpoint_path, output_video_path)

    logger.info("Done!")


# Choose which experiments to analyse. They must have been trained first!
list_experiment_names = ["analytical",
                         "no_regularizer",
                         "analytical_regression_regularizer",
                         "fokker_planck_regularizer",
                         "consistency_regularizer",
                         "consistency_with_analytical_guide_regularizer",
                         ]
run_name = "run1"


if __name__ == "__main__":
    for experiment_name in list_experiment_names:
        logger.info(f"Doing {experiment_name}")
        analyse(experiment_name, run_name)
