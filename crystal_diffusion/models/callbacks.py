import dataclasses

from pytorch_lightning import Callback


class HPLoggingCallback(Callback):
    """This callback is responsible for logging hyperparameters."""
    def on_train_start(self, trainer, pl_module):
        """Log hyperparameters when training starts."""
        assert hasattr(pl_module, 'hyper_params'), \
            "The lightning module should have a hyper_params attribute for HP logging."
        hp_dict = dataclasses.asdict(pl_module.hyper_params)
        trainer.logger.log_hyperparams(hp_dict)
