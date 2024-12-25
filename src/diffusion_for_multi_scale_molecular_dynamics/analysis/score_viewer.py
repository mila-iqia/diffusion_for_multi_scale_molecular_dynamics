from dataclasses import dataclass
from typing import List

import einops
import torch
from matplotlib import pyplot as plt

from diffusion_for_multi_scale_molecular_dynamics.analysis import (
    PLEASANT_FIG_SIZE, PLOT_STYLE_PATH)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.analytical_score_network import (
    AnalyticalScoreNetwork, AnalyticalScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, CARTESIAN_FORCES, NOISE, NOISY_AXL_COMPOSITION, TIME, UNIT_CELL)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.exploding_variance import \
    VarianceScheduler
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell

plt.style.use(PLOT_STYLE_PATH)


@dataclass(kw_only=True)
class ScoreViewerParameters:
    """Parameters for the Score Viewer class."""
    sigma_min: float
    sigma_max: float

    number_of_space_steps: int = 1000

    # Starting and ending relative coordinates should be of shape [number of atoms, spatial dimension]
    starting_relative_coordinates: List[List[float]]
    ending_relative_coordinates: List[List[float]]


class ScoreViewer:
    """Score Viewer.

    This class drives the generation of figures that show the score along specified
    directions.
    """

    def __init__(self, score_viewer_parameters: ScoreViewerParameters,
                 analytical_score_network_parameters: AnalyticalScoreNetworkParameters):
        """Init method."""
        total_time_steps = 8

        noise_parameters = NoiseParameters(total_time_steps=total_time_steps,
                                           sigma_min=score_viewer_parameters.sigma_min,
                                           sigma_max=score_viewer_parameters.sigma_max)

        self.times = torch.tensor([0., 0.1, 0.2, 0.3, 0.4, 0.8, 0.9, 1.0])
        self.sigmas = VarianceScheduler(noise_parameters).get_sigma(self.times).numpy()

        self.analytical_score_network = AnalyticalScoreNetwork(analytical_score_network_parameters)

        self.natoms = analytical_score_network_parameters.number_of_atoms

        self.start = torch.tensor(score_viewer_parameters.starting_relative_coordinates)
        self.end = torch.tensor(score_viewer_parameters.ending_relative_coordinates)

        self.number_of_space_steps = score_viewer_parameters.number_of_space_steps

        self.relative_coordinates, self.displacements = self.get_relative_coordinates_and_displacement()
        self.direction_vector = self.get_direction_vector()

    def get_relative_coordinates_and_displacement(self):
        """Get the relative coordinates and the displacement."""
        direction = (self.end - self.start) / self.number_of_space_steps
        steps = torch.arange(self.number_of_space_steps + 1)
        relative_coordinates = self.start.unsqueeze(0) + steps.view(-1, 1, 1) * direction.unsqueeze(0)
        relative_coordinates = map_relative_coordinates_to_unit_cell(relative_coordinates)

        displacements = steps * direction.norm()
        return relative_coordinates, displacements

    def get_direction_vector(self):
        """Get direction vector."""
        direction_vector = einops.rearrange(self.end - self.start, "natoms space -> (natoms space)")
        return direction_vector / direction_vector.norm()

    def get_batch(self, time: float, sigma: float):
        """Get batch."""
        batch_size = self.relative_coordinates.shape[0]

        sigmas_t = sigma * torch.ones(batch_size, 1)
        times = time * torch.ones(batch_size, 1)
        unit_cell = torch.ones(batch_size, 1, 1)
        forces = torch.zeros_like(self.relative_coordinates)
        atom_types = torch.zeros(batch_size, self.natoms, dtype=torch.int64)

        composition = AXL(A=atom_types,
                          X=self.relative_coordinates,
                          L=torch.zeros_like(self.relative_coordinates))

        batch = {NOISY_AXL_COMPOSITION: composition,
                 NOISE: sigmas_t,
                 TIME: times,
                 UNIT_CELL: unit_cell,
                 CARTESIAN_FORCES: forces}
        return batch

    def create_figure(self, score_network: ScoreNetwork):
        """Create a matplotlib figure."""
        figsize = (2 * PLEASANT_FIG_SIZE[0], PLEASANT_FIG_SIZE[0])
        fig = plt.figure(figsize=figsize)

        ax1 = fig.add_subplot(241)
        ax2 = fig.add_subplot(242)
        ax3 = fig.add_subplot(243)
        ax4 = fig.add_subplot(244)
        ax5 = fig.add_subplot(245)
        ax6 = fig.add_subplot(246)
        ax7 = fig.add_subplot(247)
        ax8 = fig.add_subplot(248)

        list_ax = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]

        list_params = [dict(color='green', lw=4, label='Analytical Normalized Score'),
                       dict(color='red', lw=2, label='Model Normalized Score')]

        list_score_networks = [self.analytical_score_network, score_network]

        for time, sigma, ax in zip(self.times, self.sigmas, list_ax):
            batch = self.get_batch(time, sigma)

            for model, params in zip(list_score_networks, list_params):
                sigma_normalized_scores = model(batch).X.detach()
                vectors = einops.rearrange(sigma_normalized_scores,
                                           "batch natoms space -> batch (natoms space)")
                projected_sigma_normalized_scores = torch.matmul(vectors, self.direction_vector)
                ax.plot(self.displacements, projected_sigma_normalized_scores, **params)

            ax.set_title(f"t = {time: 3.2f}," + r" $\sigma(t)$ = " + f"{sigma:5.3f}")
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)

            ymin, ymax = ax.set_ylim()
            ax.set_ylim(-ymax, ymax)
            ax.set_xlim(self.displacements[0] - 0.01, self.displacements[-1] + 0.01)

        ax8.legend(loc=0)
        fig.tight_layout()

        return fig


if __name__ == '__main__':
    analytical_score_network_parameters = (
        AnalyticalScoreNetworkParameters(number_of_atoms=2,
                                         spatial_dimension=1,
                                         num_atom_types=1,
                                         kmax=5,
                                         sigma_d=0.01,
                                         equilibrium_relative_coordinates=[[0.25], [0.75]],
                                         use_permutation_invariance=True))

    score_viewer_parameters = ScoreViewerParameters(sigma_min=0.001,
                                                    sigma_max=0.2,
                                                    starting_relative_coordinates=[[0.], [1.]],
                                                    ending_relative_coordinates=[[1.], [0.]])

    score_viewer = ScoreViewer(score_viewer_parameters, analytical_score_network_parameters)

    score_network = AnalyticalScoreNetwork(analytical_score_network_parameters)

    fig = score_viewer.create_figure(score_network)

    plt.show()
