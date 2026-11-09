import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping


def restart_from_checkpoint(trainer, new_patience, new_learning_rate):
    """Prepare a trainer to restart from a checkpoint.

    The problem is that trainer.fit will overwrite any variable we give to the model. As such, we create a
    callback just before the fitting starts that resets the following variables to their expected values:
        - Patience: set back to the given patience.
        - Best score: reset to its initial "no best yet" value (+/-inf, per EarlyStopping's own convention).
        - Number of epochs without improvement: reset back to 0.
        - (If applicable) Learning rate: set back to the given learning rate.
        - (If applicable) LR scheduler's own "epochs since improvement" bookkeeping: reset the same way.

    Args:
        trainer: the Trainer that will resume training.
        new_patience: patience to use from this run on.
        new_learning_rate: learning rate to use from this run on.
    """
    trainer.callbacks.append(_RestartOverrideCallback(new_patience, new_learning_rate))


class _RestartOverrideCallback(Callback):
    def __init__(self, new_patience, new_learning_rate):
        self.new_patience = new_patience
        self.new_learning_rate = new_learning_rate

    def on_train_start(self, trainer, pl_module):
        for callback in trainer.callbacks:
            if isinstance(callback, EarlyStopping):
                callback.patience = self.new_patience
                callback.wait_count = 0
                callback.best_score = (
                    torch.tensor(torch.inf) if callback.monitor_op == torch.lt else torch.tensor(-torch.inf)
                )

        for optimizer in trainer.optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.new_learning_rate

        for lr_scheduler_config in trainer.lr_scheduler_configs:
            scheduler = lr_scheduler_config.scheduler
            if hasattr(scheduler, "_reset"):
                scheduler._reset()
