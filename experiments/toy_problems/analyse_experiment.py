import logging
import sys

from diffusion_for_multi_scale_molecular_dynamics import TOP_DIR
from diffusion_for_multi_scale_molecular_dynamics.utils.logging_utils import \
    setup_analysis_logger

sys.path.append(str(TOP_DIR / "experiments")) # noqa

from toy_problems import RESULTS_DIR  # noqa
from toy_problems.utils.analysis_utils import InputParameters  # noqa
from toy_problems.utils.analysis_utils import create_samples  # noqa
from toy_problems.utils.analysis_utils import get_checkpoint_path  # noqa
from toy_problems.utils.analysis_utils import get_vector_field_movie  # noqa
from toy_problems.utils.analysis_utils import \
    plot_marginal_distribution  # noqa
from toy_problems.utils.analysis_utils import plot_samples  # noqa

logger = logging.getLogger(__name__)
setup_analysis_logger()


def analyse(experiment_name: str, run_name: str):
    """Conduct all the analyses for a given experiment."""
    logger.info(f"Starting analysis of experiment {experiment_name} / {run_name}")

    checkpoint_path = get_checkpoint_path(experiment_name, run_name)

    logger.info(f"  checkpoint is  {checkpoint_path}")

    input_parameters = InputParameters(algorithm="predictor_corrector",
                                       total_time_steps=100,
                                       number_of_corrector_steps=0,
                                       corrector_r=0.4,
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
                         "consistency_with_analytical_guide_regularizer"]
run_name = "run1"


if __name__ == "__main__":
    for experiment_name in list_experiment_names:
        logger.info(f"Doing {experiment_name}")
        analyse(experiment_name, run_name)
