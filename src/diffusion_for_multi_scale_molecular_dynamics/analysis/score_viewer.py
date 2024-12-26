from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

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

    number_of_space_steps: int = 100

    # Starting and ending relative coordinates should be of shape [number of atoms, spatial dimension]
    starting_relative_coordinates: List[List[float]]
    ending_relative_coordinates: List[List[float]]


class ScoreViewer:
    """Score Viewer.

    This class drives the generation of figures that show the score along specified
    directions. The figure is composed of 8 panes (matplotlib "axes") that show
    the projected normalized score and various baselines along the specified 1D direction.
    The projection of the score is on the tangent to the 1D line going from starting to ending
    relative coordinates.
    """

    def __init__(
        self,
        score_viewer_parameters: ScoreViewerParameters,
        analytical_score_network_parameters: AnalyticalScoreNetworkParameters,
    ):
        """Init method."""
        total_time_steps = 8

        noise_parameters = NoiseParameters(
            total_time_steps=total_time_steps,
            sigma_min=score_viewer_parameters.sigma_min,
            sigma_max=score_viewer_parameters.sigma_max,
        )

        self.times = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.8, 0.9, 1.0])
        self.number_of_axes = 8

        self.sigmas = VarianceScheduler(noise_parameters).get_sigma(self.times).numpy()

        self.analytical_score_network = AnalyticalScoreNetwork(
            analytical_score_network_parameters
        )

        number_of_equilibrium_points = len(self.analytical_score_network.all_x0)

        self.natoms = analytical_score_network_parameters.number_of_atoms

        self.start = torch.tensor(score_viewer_parameters.starting_relative_coordinates)
        self.end = torch.tensor(score_viewer_parameters.ending_relative_coordinates)

        self.number_of_space_steps = score_viewer_parameters.number_of_space_steps

        self.relative_coordinates, self.displacements = (
            self._get_relative_coordinates_and_displacement()
        )
        self.direction_vector = self._get_direction_vector()

        # Compute the various references and baselines once and for all, keeping them in memory.
        self.projected_analytical_scores = self._compute_projected_scores(
            self.analytical_score_network
        )

        self.projected_gaussian_scores_dict = (
            self._get_naive_projected_gaussian_scores()
        )

        self.all_in_distribution_bands = self._get_in_distribution_bands()

        self.plot_style_parameters = self._get_plot_style_parameters(
            number_of_equilibrium_points
        )

    def _get_relative_coordinates_and_displacement(self):
        """Get the relative coordinates and the displacement."""
        direction = (self.end - self.start) / (self.number_of_space_steps + 1)
        # Avoid the first point, which tends to klank because of periodicity.
        steps = torch.arange(self.number_of_space_steps + 1)[1:]
        relative_coordinates = self.start.unsqueeze(0) + steps.view(
            -1, 1, 1
        ) * direction.unsqueeze(0)
        relative_coordinates = map_relative_coordinates_to_unit_cell(
            relative_coordinates
        )

        displacements = steps * direction.norm()
        return relative_coordinates, displacements

    def _get_direction_vector(self):
        """Get direction vector."""
        direction_vector = einops.rearrange(
            self.end - self.start, "natoms space -> (natoms space)"
        )
        return direction_vector / direction_vector.norm()

    def _get_batch(self, time: float, sigma: float):
        """Get batch."""
        batch_size = self.relative_coordinates.shape[0]

        sigmas_t = sigma * torch.ones(batch_size, 1)
        times = time * torch.ones(batch_size, 1)
        unit_cell = torch.ones(batch_size, 1, 1)
        forces = torch.zeros_like(self.relative_coordinates)
        atom_types = torch.zeros(batch_size, self.natoms, dtype=torch.int64)

        composition = AXL(
            A=atom_types,
            X=self.relative_coordinates,
            L=torch.zeros_like(self.relative_coordinates),
        )

        batch = {
            NOISY_AXL_COMPOSITION: composition,
            NOISE: sigmas_t,
            TIME: times,
            UNIT_CELL: unit_cell,
            CARTESIAN_FORCES: forces,
        }
        return batch

    def _compute_projected_scores(self, score_network: ScoreNetwork):
        """Compute projected scores."""
        list_projected_scores = []
        for time, sigma in zip(self.times, self.sigmas):
            batch = self._get_batch(time, sigma)

            sigma_normalized_scores = score_network(batch).X.detach()
            vectors = einops.rearrange(
                sigma_normalized_scores, "batch natoms space -> batch (natoms space)"
            )
            projected_sigma_normalized_scores = torch.matmul(
                vectors, self.direction_vector
            )
            list_projected_scores.append(projected_sigma_normalized_scores)

        return list_projected_scores

    def _get_naive_projected_gaussian_scores(self) -> Dict:
        """Compute the scores as if coming from a simple, single Gaussian."""
        projected_gaussian_scores_dict = defaultdict(list)
        for time, sigma in zip(self.times, self.sigmas):

            prefactor = -sigma / (
                sigma**2 + self.analytical_score_network.sigma_d_square
            )

            for idx, x0 in enumerate(self.analytical_score_network.all_x0):

                equilibrium_relative_coordinates = einops.repeat(
                    x0,
                    "natoms space -> batch natoms space",
                    batch=self.number_of_space_steps,
                )

                directions = (
                    self.relative_coordinates - equilibrium_relative_coordinates
                )
                vectors = einops.rearrange(
                    directions, "batch natoms space -> batch (natoms space)"
                )
                projected_directions = torch.matmul(vectors, self.direction_vector)
                gaussian_normalized_score = prefactor * projected_directions
                projected_gaussian_scores_dict[idx].append(gaussian_normalized_score)

        return projected_gaussian_scores_dict

    def _get_in_distribution_bands(self) -> List[List[Tuple]]:
        """Create the limits where the relative coordinates are within sigma_eff of an equilibrium point."""
        # Start of the 1D visualization line
        origin = einops.rearrange(self.start, "natoms space -> (natoms space)")

        # Tangent vector to the 1D visualization line
        d_hat = einops.rearrange(
            self.end - self.start, "natoms space -> (natoms space)"
        )
        d_hat = d_hat / d_hat.norm()

        list_bands = []
        for sigma in self.sigmas:
            effective_sigma_square = torch.tensor(
                sigma**2 + self.analytical_score_network.sigma_d_square
            )

            bands = []
            for idx, x0 in enumerate(self.analytical_score_network.all_x0):
                center = einops.rearrange(x0, "natoms space -> (natoms space)")

                # We can build a simple quadratic equation to identify where the visualization line
                # intersects the sphere centered at 'center' with radius effective_sigma.
                v = origin - center
                a = 1.0
                b = 2.0 * torch.dot(d_hat, v)
                c = torch.dot(v, v) - effective_sigma_square

                discriminant = b**2 - 4 * a * c

                if discriminant < 0:
                    band = None
                else:
                    dmin = (-b - torch.sqrt(discriminant)) / (2 * a)
                    dmax = (-b + torch.sqrt(discriminant)) / (2 * a)
                    band = (dmin, dmax)
                bands.append(band)
            list_bands.append(bands)
        return list_bands

    def _get_plot_style_parameters(self, number_of_equilibrium_points: int):
        """Create the linestyles for each item in the plots."""
        gaussian_linestyle = dict(ls="--", color="black", lw=2, alpha=0.5)

        list_params = [
            dict(ls="-", color="green", lw=4, label="Analytical Normalized Score"),
            dict(ls="-", color="red", lw=2, label="Model Normalized Score"),
        ]

        gaussian_params = dict(gaussian_linestyle)
        gaussian_params["label"] = "Gaussian Approximation"
        list_params.append(gaussian_params)

        for _ in range(number_of_equilibrium_points - 1):
            gaussian_params = dict(gaussian_linestyle)
            gaussian_params["label"] = "__nolegend__"
            list_params.append(gaussian_params)

        return list_params

    def create_figure(self, score_network: ScoreNetwork):
        """Create Figure.

        Create a matplotlib figure showing the projected normalized scores for the model
        along with various baselines.
        """
        model_projected_scores = self._compute_projected_scores(score_network)

        figsize = (2 * PLEASANT_FIG_SIZE[0], PLEASANT_FIG_SIZE[0])
        fig = plt.figure(figsize=figsize)

        ax_id = 240
        for idx, (time, sigma) in enumerate(zip(self.times, self.sigmas)):
            ax_id += 1
            ax = fig.add_subplot(ax_id)

            ax.set_title(f"t = {time: 3.2f}," + r" $\sigma(t)$ = " + f"{sigma:5.3f}")
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)

            list_projected_scores = [
                self.projected_analytical_scores[idx],
                model_projected_scores[idx],
            ]

            maximum_projected_score_values = (
                torch.stack(list_projected_scores).abs().max()
            )

            for (
                all_gaussian_projected_scores
            ) in self.projected_gaussian_scores_dict.values():
                list_projected_scores.append(all_gaussian_projected_scores[idx])

            for projected_scores, params in zip(
                list_projected_scores, self.plot_style_parameters
            ):
                ax.plot(self.displacements, projected_scores, **params)

            ymax = 1.2 * maximum_projected_score_values
            ax.set_ylim(-ymax, ymax)
            ax.set_xlim(self.displacements[0] - 0.01, self.displacements[-1] + 0.01)

            bands = self.all_in_distribution_bands[idx]
            label = r"$x_0 \pm \sigma_{eff}$"
            for band in bands:
                if band is None:
                    continue
                ax.fill_betweenx(
                    y=[-ymax, ymax],
                    x1=band[0],
                    x2=band[1],
                    color="green",
                    alpha=0.10,
                    label=label,
                )
                label = "__nolegend__"

        # The last ax gets the legend.
        ax.legend(loc=0)
        fig.tight_layout()

        return fig


if __name__ == "__main__":
    # A simple demonstration of how the Score Viewer works. We naively use an analytical score network
    # as the external score network, such that the 'model' results will overlap with the analytical score baseline.
    analytical_score_network_parameters = AnalyticalScoreNetworkParameters(
        number_of_atoms=2,
        spatial_dimension=1,
        num_atom_types=1,
        kmax=5,
        sigma_d=0.01,
        equilibrium_relative_coordinates=[[0.25], [0.75]],
        use_permutation_invariance=True,
    )

    score_viewer_parameters = ScoreViewerParameters(
        sigma_min=0.001,
        sigma_max=0.2,
        starting_relative_coordinates=[[0.0], [1.0]],
        ending_relative_coordinates=[[1.0], [0.0]],
    )

    score_viewer = ScoreViewer(
        score_viewer_parameters, analytical_score_network_parameters
    )

    score_network = AnalyticalScoreNetwork(analytical_score_network_parameters)

    fig = score_viewer.create_figure(score_network)
    fig.suptitle("Demonstration")
    fig.tight_layout()

    plt.show()
