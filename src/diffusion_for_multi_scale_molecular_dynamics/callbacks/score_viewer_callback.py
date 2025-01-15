from dataclasses import dataclass
from typing import Any, AnyStr, Dict

from matplotlib import pyplot as plt
from pytorch_lightning import Callback, LightningModule, Trainer

from diffusion_for_multi_scale_molecular_dynamics.analysis.score_viewer import (
    ScoreViewer, ScoreViewerParameters)
from diffusion_for_multi_scale_molecular_dynamics.loggers.logger_loader import \
    log_figure
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.analytical_score_network import \
    AnalyticalScoreNetworkParameters


@dataclass(kw_only=True)
class ScoreViewerCallbackParameters:
    """Parameters to decide what to plot and write to disk."""

    record_every_n_epochs: int = 1

    score_viewer_parameters: ScoreViewerParameters
    analytical_score_network_parameters: AnalyticalScoreNetworkParameters


def instantiate_score_viewer_callback(
    callback_params: Dict[AnyStr, Any], output_directory: str, verbose: bool
) -> Dict[str, Callback]:
    """Instantiate the Diffusion Sampling callback."""
    analytical_score_network_parameters = (
        AnalyticalScoreNetworkParameters(**callback_params['analytical_score_network']))

    score_viewer_parameters = ScoreViewerParameters(**callback_params['score_viewer_parameters'])

    score_viewer_callback_parameters = ScoreViewerCallbackParameters(
        record_every_n_epochs=callback_params['record_every_n_epochs'],
        score_viewer_parameters=score_viewer_parameters,
        analytical_score_network_parameters=analytical_score_network_parameters)

    callback = ScoreViewerCallback(
        score_viewer_callback_parameters, output_directory
    )

    return dict(score_viewer=callback)


class ScoreViewerCallback(Callback):
    """Score Viewer Callback."""

    def __init__(self, score_viewer_callback_parameters: ScoreViewerCallbackParameters, output_directory: str):
        """Init method."""
        self.record_every_n_epochs = score_viewer_callback_parameters.record_every_n_epochs
        self.score_viewer = ScoreViewer(
            score_viewer_parameters=score_viewer_callback_parameters.score_viewer_parameters,
            analytical_score_network_parameters=score_viewer_callback_parameters.analytical_score_network_parameters)

    def _compute_results_at_this_epoch(self, current_epoch: int) -> bool:
        """Check if results should be computed at this epoch."""
        return current_epoch % self.record_every_n_epochs == 0

    def on_validation_end(self, trainer: Trainer, pl_model: LightningModule) -> None:
        """On validation epoch end."""
        if not self._compute_results_at_this_epoch(trainer.current_epoch):
            return

        figure = self.score_viewer.create_figure(score_network=pl_model.axl_network)
        figure.suptitle(f"Epoch {trainer.current_epoch}, Step {trainer.global_step}")
        # Set the DPI so we can actually see something in the logger window.
        figure.set_dpi(100)
        figure.tight_layout()

        for pl_logger in trainer.loggers:
            log_figure(
                figure=figure,
                global_step=trainer.current_epoch,
                dataset="validation",
                pl_logger=pl_logger,
                name="projected_normalized_scores",
            )
            plt.close(figure)
