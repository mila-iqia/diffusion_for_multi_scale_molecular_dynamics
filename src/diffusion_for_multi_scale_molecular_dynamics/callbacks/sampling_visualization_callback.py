import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AnyStr, Dict

import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from matplotlib import pyplot as plt

from diffusion_for_multi_scale_molecular_dynamics.analysis import (
    PLEASANT_FIG_SIZE, PLOT_STYLE_PATH)
from diffusion_for_multi_scale_molecular_dynamics.loggers.logger_loader import \
    log_figure

logger = logging.getLogger(__name__)

plt.style.use(PLOT_STYLE_PATH)


@dataclass(kw_only=True)
class SamplingVisualizationParameters:
    """Parameters to decide what to plot and write to disk."""

    record_every_n_epochs: int = 1
    first_record_epoch: int = 1
    record_trajectories: bool = True
    record_energies: bool = True
    record_structure: bool = True
    record_lattice_parameters: bool = True


def instantiate_sampling_visualization_callback(
    callback_params: Dict[AnyStr, Any], output_directory: str, verbose: bool
) -> Dict[str, Callback]:
    """Instantiate the Diffusion Sampling callback."""
    sampling_visualization_parameters = SamplingVisualizationParameters(
        **callback_params
    )

    callback = SamplingVisualizationCallback(
        sampling_visualization_parameters, output_directory
    )

    return dict(sampling_visualization=callback)


class SamplingVisualizationCallback(Callback):
    """Callback class to periodically generate samples and log their energies."""

    def __init__(
        self,
        sampling_visualization_parameters: SamplingVisualizationParameters,
        output_directory: str,
    ):
        """Init method."""
        self.parameters = sampling_visualization_parameters
        self.output_directory = output_directory

        if self.parameters.record_energies:
            self.sample_energies_output_directory = os.path.join(
                output_directory, "energy_samples"
            )
            Path(self.sample_energies_output_directory).mkdir(
                parents=True, exist_ok=True
            )

        if self.parameters.record_structure:
            self.sample_distances_output_directory = os.path.join(
                output_directory, "distance_samples"
            )
            Path(self.sample_distances_output_directory).mkdir(
                parents=True, exist_ok=True
            )

        if self.parameters.record_trajectories:
            self.sample_trajectories_output_directory = os.path.join(
                output_directory, "trajectory_samples"
            )
            Path(self.sample_trajectories_output_directory).mkdir(
                parents=True, exist_ok=True
            )

        if self.parameters.record_lattice_parameters:
            self.lattice_parameters_output_directory = os.path.join(
                output_directory, "lattice_parameters"
            )
            Path(self.lattice_parameters_output_directory).mkdir(
                parents=True, exist_ok=True
            )

    def on_validation_end(self, trainer: Trainer, pl_model: LightningModule) -> None:
        """On validation end."""
        if not self._compute_results_at_this_epoch(trainer.current_epoch):
            return

        if self.parameters.record_energies:
            assert (
                pl_model.energy_ks_metric is not None
            ), "The energy_ks_metric is absent. Energy calculation must be requested in order to be visualized!"
            reference_energies = (
                pl_model.energy_ks_metric.reference_samples_metric.compute()
            )
            sample_energies = (
                pl_model.energy_ks_metric.predicted_samples_metric.compute()
            )
            energy_output_path = os.path.join(
                self.sample_energies_output_directory,
                f"energies_sample_epoch={trainer.current_epoch}.pt",
            )
            torch.save(sample_energies, energy_output_path)

            sample_energies = sample_energies.cpu().numpy()
            reference_energies = reference_energies.cpu().numpy()

            fig1 = self._plot_energy_histogram(
                sample_energies, reference_energies, trainer.current_epoch
            )
            fig2 = self._plot_energy_quantiles(
                sample_energies, reference_energies, trainer.current_epoch
            )

            for pl_logger in trainer.loggers:
                log_figure(
                    figure=fig1,
                    global_step=trainer.global_step,
                    dataset="validation",
                    pl_logger=pl_logger,
                    name="energy_distribution",
                )
                log_figure(
                    figure=fig2,
                    global_step=trainer.global_step,
                    dataset="validation",
                    pl_logger=pl_logger,
                    name="energy_quantiles",
                )
                plt.close(fig1)
                plt.close(fig2)

        if self.parameters.record_structure:
            assert pl_model.structure_ks_metric is not None, (
                "The structure_ks_metric is absent. Structure factor calculation "
                "must be requested in order to be visualized!"
            )

            reference_distances = (
                pl_model.structure_ks_metric.reference_samples_metric.compute()
            )
            sample_distances = (
                pl_model.structure_ks_metric.predicted_samples_metric.compute()
            )

            distance_output_path = os.path.join(
                self.sample_distances_output_directory,
                f"distances_sample_epoch={trainer.current_epoch}.pt",
            )

            torch.save(sample_distances, distance_output_path)
            fig = self._plot_distance_histogram(
                sample_distances.numpy(),
                reference_distances.numpy(),
                trainer.current_epoch,
            )

            for pl_logger in trainer.loggers:
                log_figure(
                    figure=fig,
                    global_step=trainer.global_step,
                    dataset="validation",
                    pl_logger=pl_logger,
                    name="distances",
                )
                plt.close(fig)

        if self.parameters.record_trajectories:
            assert (
                pl_model.generator is not None
            ), "Cannot record trajectories if a generator has not be created."

            pickle_output_path = os.path.join(
                self.sample_trajectories_output_directory,
                f"trajectories_sample_epoch={trainer.current_epoch}.pt",
            )
            pl_model.generator.sample_trajectory_recorder.write_to_pickle(
                pickle_output_path
            )

        if self.parameters.record_lattice_parameters:
            assert pl_model.lattice_parameters_ks_metrics is not None, (
                "The lattice_parameter_ks_metric is absent. Lattice parameters calculation "
                "must be requested in order to be visualized!"
            )

            reference_lattice_parameters = [
                metric.reference_samples_metric.compute()
                for metric in pl_model.lattice_parameters_ks_metrics
                if metric is not None
            ]
            sample_lattice_parameters = [
                metric.predicted_samples_metric.compute()
                for metric in pl_model.lattice_parameters_ks_metrics
                if metric is not None
            ]

            lattice_parameters_output_path = os.path.join(
                self.lattice_parameters_output_directory,
                f"lattice_parameters_sample_epoch={trainer.current_epoch}.pt",
            )

            torch.save(sample_lattice_parameters, lattice_parameters_output_path)
            figs = [
                self._plot_lattice_parameters_histogram(
                    samples.numpy(),
                    references.numpy(),
                    lattice_index,
                    trainer.current_epoch,
                )
                for lattice_index, (references, samples) in enumerate(
                    zip(reference_lattice_parameters, sample_lattice_parameters)
                )
            ]

            for pl_logger in trainer.loggers:
                for i, fig in enumerate(figs):
                    log_figure(
                        figure=fig,
                        global_step=trainer.global_step,
                        dataset="validation",
                        pl_logger=pl_logger,
                        name=f"lattice_parameter_{i}",
                    )
                    plt.close(fig)

    def _compute_results_at_this_epoch(self, current_epoch: int) -> bool:
        """Check if results should be computed at this epoch."""
        if (
            current_epoch % self.parameters.record_every_n_epochs == 0
            and current_epoch >= self.parameters.first_record_epoch
        ):
            return True
        else:
            return False

    @staticmethod
    def _plot_energy_quantiles(
        sample_energies: np.ndarray, validation_dataset_energies: np.array, epoch: int
    ) -> plt.figure:
        """Generate a plot of the energy quantiles."""
        list_q = np.linspace(0, 1, 101)
        sample_quantiles = np.quantile(sample_energies, list_q)
        dataset_quantiles = np.quantile(validation_dataset_energies, list_q)

        fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
        fig.suptitle(f"Sampling Energy Quantiles\nEpoch {epoch}")
        ax = fig.add_subplot(111)

        label = f"Samples \n(total count = {len(sample_energies)})"
        ax.plot(100 * list_q, sample_quantiles, "-", lw=5, color="red", label=label)

        label = f"Validation Data \n(count = {len(validation_dataset_energies)})"
        ax.plot(
            100 * list_q, dataset_quantiles, "--", lw=10, color="green", label=label
        )
        ax.set_xlabel("Quantile (%)")
        ax.set_ylabel("Energy (eV)")
        ax.set_xlim(-0.1, 100.1)
        ax.legend(loc="upper right", fancybox=True, shadow=True, ncol=1, fontsize=6)
        fig.tight_layout()

        return fig

    @staticmethod
    def _plot_energy_histogram(
        sample_energies: np.ndarray, validation_dataset_energies: np.array, epoch: int
    ) -> plt.figure:
        """Generate a plot of the energy samples."""
        fig = plt.figure(figsize=PLEASANT_FIG_SIZE)

        minimum_energy = validation_dataset_energies.min()
        maximum_energy = validation_dataset_energies.max()
        energy_range = maximum_energy - minimum_energy

        emin = minimum_energy - 0.2 * energy_range
        emax = maximum_energy + 0.2 * energy_range
        bins = np.linspace(emin, emax, 101)

        number_of_samples_in_range = np.logical_and(
            sample_energies >= emin, sample_energies <= emax
        ).sum()

        fig.suptitle(f"Sampling Energy Distributions\nEpoch {epoch}")

        common_params = dict(density=True, bins=bins, histtype="stepfilled", alpha=0.25)

        ax1 = fig.add_subplot(111)

        ax1.hist(
            sample_energies,
            **common_params,
            label=f"Samples \n(total count = {len(sample_energies)}, in range = {number_of_samples_in_range})",
            color="red",
        )
        ax1.hist(
            validation_dataset_energies,
            **common_params,
            label=f"Validation Data \n(count = {len(validation_dataset_energies)})",
            color="green",
        )

        ax1.set_xlabel("Energy (eV)")
        ax1.set_ylabel("Density")
        ax1.legend(loc="upper right", fancybox=True, shadow=True, ncol=1, fontsize=6)
        fig.tight_layout()
        return fig

    @staticmethod
    def _plot_distance_histogram(
        sample_distances: np.ndarray, validation_dataset_distances: np.array, epoch: int
    ) -> plt.figure:
        """Generate a plot of the inter-atomic distances of the samples."""
        fig = plt.figure(figsize=PLEASANT_FIG_SIZE)

        maximum_distance = validation_dataset_distances.max()

        dmin = 0.0
        dmax = maximum_distance + 0.1
        bins = np.linspace(dmin, dmax, 251)

        fig.suptitle(f"Sampling Distances Distribution\nEpoch {epoch}")

        common_params = dict(density=True, bins=bins, histtype="stepfilled", alpha=0.25)

        ax1 = fig.add_subplot(111)

        ax1.hist(
            sample_distances,
            **common_params,
            label=f"Samples \n(total count = {len(sample_distances)})",
            color="red",
        )
        ax1.hist(
            validation_dataset_distances,
            **common_params,
            label=f"Validation Data \n(count = {len(validation_dataset_distances)})",
            color="green",
        )

        ax1.set_xlabel(r"Distance ($\AA$)")
        ax1.set_ylabel("Density")
        ax1.legend(loc="upper right", fancybox=True, shadow=True, ncol=1, fontsize=6)
        ax1.set_xlim(left=dmin, right=dmax)
        fig.tight_layout()
        return fig

    @staticmethod
    def _plot_lattice_parameters_histogram(
        sample_parameters: np.ndarray,
        validation_parameters: np.array,
        parameter_index: int,
        epoch: int,
    ) -> plt.figure:
        """Generate a plot of a lattice parameter of the samples."""
        fig = plt.figure(figsize=PLEASANT_FIG_SIZE)

        maximum_parameter = validation_parameters.max()

        dmin = 0.0
        dmax = maximum_parameter + 0.1
        bins = np.linspace(dmin, dmax, 251)

        fig.suptitle(
            f"Sampling Lattice Parameter {parameter_index} Distribution\nEpoch {epoch}"
        )

        common_params = dict(density=True, bins=bins, histtype="stepfilled", alpha=0.25)

        ax1 = fig.add_subplot(111)

        ax1.hist(
            sample_parameters,
            **common_params,
            label=f"Samples \n(total count = {len(sample_parameters)})",
            color="red",
        )
        ax1.hist(
            validation_parameters,
            **common_params,
            label=f"Validation Data \n(count = {len(validation_parameters)})",
            color="green",
        )

        ax1.set_xlabel(r"Lattice parameter ($\AA$)")
        ax1.set_ylabel("Density")
        ax1.legend(loc="upper right", fancybox=True, shadow=True, ncol=1, fontsize=6)
        ax1.set_xlim(left=dmin, right=dmax)
        fig.tight_layout()
        return fig
