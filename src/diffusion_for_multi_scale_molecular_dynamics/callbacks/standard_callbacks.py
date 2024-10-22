import logging
import os
from typing import Any, AnyStr, Dict

from pytorch_lightning import Callback
from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint,
                                         RichProgressBar)

logger = logging.getLogger(__name__)

BEST_MODEL_NAME = "best_model"
LAST_MODEL_NAME = "last_model"


def instantiate_early_stopping_callback(
    callback_params: Dict[AnyStr, Any], output_directory: str, verbose: bool
) -> Dict[str, Callback]:
    """Instantiate early stopping callback."""
    early_stopping = EarlyStopping(
        callback_params["metric"],
        mode=callback_params["mode"],
        patience=callback_params["patience"],
        verbose=verbose,
    )
    return dict(early_stopping=early_stopping)


def instantiate_model_checkpoint_callbacks(
    callback_params: Dict[AnyStr, Any], output_directory: str, verbose: bool
) -> Dict[str, Callback]:
    """Instantiate best and last checkpoint callbacks."""
    best_model_path = os.path.join(output_directory, BEST_MODEL_NAME)
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=best_model_path,
        filename="best_model-{epoch:03d}-{step:06d}",
        save_top_k=1,
        verbose=verbose,
        monitor=callback_params["monitor"],
        mode=callback_params["mode"],
        every_n_epochs=1,
    )

    last_model_path = os.path.join(output_directory, LAST_MODEL_NAME)
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=last_model_path,
        filename="last_model-{epoch:03d}-{step:06d}",
        verbose=verbose,
        every_n_epochs=1,
    )
    return dict(
        best_checkpoint=best_checkpoint_callback,
        last_checkpoint=last_checkpoint_callback,
    )


class CustomProgressBar(RichProgressBar):
    """A custom progress bar based on Rich that doesn't log the v_num stuff."""

    def get_metrics(self, *args, **kwargs):
        """Get metrics."""
        # don't show the version number
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        return items
