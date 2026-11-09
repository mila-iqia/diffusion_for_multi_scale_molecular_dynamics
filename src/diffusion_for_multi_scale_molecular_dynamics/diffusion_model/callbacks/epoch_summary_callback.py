import time
from pathlib import Path
from typing import Union

from lightning import Callback
from lightning.pytorch.callbacks import EarlyStopping


class EpochSummaryLogger(Callback):
    """A per-epoch training log.

    Shows, for each epoch: train loss, bad frames loss (only when training with bad frames), validation loss, AXL loss
    (only those with weight>0), learning rate (optional), wall clock time, and early stopping progress (wait count /
    patience). Reports the early stopping outcome once training ends, if triggered.
    """

    def __init__(self, early_stopping_callback: EarlyStopping, log_file_path: Union[str, Path]):
        """Init method.

        Args:
            early_stopping_callback: the EarlyStopping callback also given to the Trainer; its wait_count,
                patience, best_score and stopped_epoch are read to report early stopping progress.
            log_file_path: path of the log file to append the per-epoch summary lines to.
        """
        self.early_stopping_callback = early_stopping_callback
        self.epoch_start_time = None

        log_file_path = Path(log_file_path)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(log_file_path, "a")

    def _emit(self, message: str):
        """Print a message and append it to the log file."""
        print(message)
        self.log_file.write(message + "\n")
        self.log_file.flush()

    def on_train_epoch_start(self, trainer, pl_module):
        """Record the epoch start time."""
        self.epoch_start_time = time.monotonic()

    def on_train_epoch_end(self, trainer, pl_module):
        """Print the per-epoch summary line."""
        metrics = trainer.callback_metrics

        def get(name):
            value = metrics.get(name)
            return float(value) if value is not None else float("nan")

        epoch_time = time.monotonic() - self.epoch_start_time if self.epoch_start_time is not None else float("nan")
        # LearningRateMonitor if present.
        lr = next((float(v) for k, v in metrics.items() if k.startswith("lr-")), None)

        # Fixed-width fields (nan pads to the same width as a normal float) so lines stay aligned across epochs.
        segments = [
            f"Epoch {trainer.current_epoch:4d}",
            f"train_loss={get('train_epoch_loss'):>10.6f}",
        ]
        tsm_parameters = pl_module.target_score_matching_parameters
        if tsm_parameters is not None and tsm_parameters.bad_frames_path is not None:
            segments.append(f"bad_frames_loss={get('train_epoch_bad_frames_loss'):>10.6f}")
        segments += [
            f"val_loss={get('validation_epoch_loss'):>10.6f}",
            f"X_loss={get('validation_epoch_relative_coordinates_loss'):>10.6f}",
        ]
        if pl_module.loss_weights.A != 0:
            segments.append(f"A_loss={get('validation_epoch_atom_types_loss'):>10.6f}")
        if pl_module.loss_weights.L != 0:
            segments.append(f"L_loss={get('validation_epoch_lattice_parameters_loss'):>10.6f}")
        if lr is not None:
            segments.append(f"lr={lr:>9.2e}")
        segments.append(f"time={epoch_time:>6.1f}s")
        segments.append(
            f"early_stop {self.early_stopping_callback.wait_count:>2d}/{self.early_stopping_callback.patience:<2d}"
        )

        self._emit(" | ".join(segments))

    def on_train_end(self, trainer, pl_module):
        """Report the early stopping outcome, if it triggered, and close the log file."""
        if self.early_stopping_callback.stopped_epoch > 0:
            best_epoch = self.early_stopping_callback.stopped_epoch - self.early_stopping_callback.patience
            self._emit(
                f"Early stopping: best validation loss of {float(self.early_stopping_callback.best_score):.6f} "
                f"reached at epoch {best_epoch}. Patience of {self.early_stopping_callback.patience} reached."
            )
        self.log_file.close()
