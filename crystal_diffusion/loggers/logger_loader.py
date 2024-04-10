from typing import Any, AnyStr, Dict, List

from pytorch_lightning.loggers import (CometLogger, CSVLogger, Logger,
                                       TensorBoardLogger)


def create_all_loggers(hyper_params: Dict[AnyStr, Any], output_directory: str) -> List[Logger]:
    """Create all loggers.

    This method instantiates all machine learning loggers defined in `hyper_params`.

    Args:
        hyper_params : configuration parameters.
        output_directory: path to where outputs are to be written.
        verbose: if relevant, should the callback produce verbose output.

    Returns:
        all_loggers : a list of loggers to pass to the Trainer.
    """
    all_loggers = []

    for logger_name in hyper_params.get('logging', []):

        match logger_name:
            case "csv":
                logger = CSVLogger(save_dir=output_directory)
            case "tensorboard":
                logger = TensorBoardLogger(save_dir=output_directory,
                                           default_hp_metric=False,
                                           version=0,  # Necessary to resume tensorboard logging
                                           )
            case "comet":
                assert 'exp_name' in hyper_params, "Cannot log to Comet without an 'exp_name'."
                # The comet logger will read the API key from ~/.comet.logger.
                logger = CometLogger(project_name='institut-courtois',      # hardcoding the project.
                                     experiment_name=hyper_params["exp_name"])

            case other:
                raise ValueError(f"Logger {other} is not implemented. Review input")

        all_loggers.append(logger)

    return all_loggers
