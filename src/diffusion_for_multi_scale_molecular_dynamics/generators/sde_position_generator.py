import dataclasses
import logging
from dataclasses import dataclass

import einops
import torch
import torchsde

from diffusion_for_multi_scale_molecular_dynamics.generators.axl_generator import (
    AXLGenerator, SamplingParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks import \
    ScoreNetwork
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, CARTESIAN_FORCES, NOISE, NOISY_AXL_COMPOSITION, TIME, UNIT_CELL)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.exploding_variance import \
    VarianceScheduler
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import (
    map_axl_composition_to_unit_cell, map_relative_coordinates_to_unit_cell)
from diffusion_for_multi_scale_molecular_dynamics.utils.sample_trajectory import \
    SampleTrajectory

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SDESamplingParameters(SamplingParameters):
    """Hyper-parameters for diffusion sampling with the sde algorithm."""

    algorithm: str = "sde"
    sde_type: str = "ito"
    method: str = "euler"
    adaptive: bool = False
    absolute_solver_tolerance: float = (
        1.0e-7  # the absolute error tolerance passed to the SDE solver.
    )
    relative_solver_tolerance: float = (
        1.0e-5  # the relative error tolerance passed to the SDE solver.
    )


class SDE(torch.nn.Module):
    """SDE.

    This class computes the drift and the diffusion coefficients in order to be consistent with the expectations
    of the torchsde library.
    """

    noise_type = "diagonal"  # we assume that there is a distinct Wiener process for each component.
    sde_type = "ito"

    def __init__(
        self,
        noise_parameters: NoiseParameters,
        sampling_parameters: SDESamplingParameters,
        axl_network: ScoreNetwork,
        atom_types: torch.LongTensor,  # TODO review formalism - this is treated as constant through the SDE solver
        unit_cells: torch.Tensor,  # TODO replace with AXL-L
        initial_diffusion_time: torch.Tensor,
        final_diffusion_time: torch.Tensor,
    ):
        """Init method.

        This class will provide drift and diffusion for the torchsde solver. The SDE will be solved
        "backwards" in diffusion time, such that the sde time will start where the diffusion ends, and vice-versa.

        Args:
            noise_parameters: parameters defining the noise schedule.
            sampling_parameters : parameters defining the sampling procedure.
            axl_network : the model to use for drawing samples that predicts an AXL:
                atom types: predicts p(a_0 | a_t)
                relative coordinates: predicts the sigma normalized score
                lattice: placeholder  # TODO
            atom_types: atom type indices. Tensor of dimensions [number_of_samples, natoms]
            unit_cells: unit cell definition in Angstrom.
                Tensor of dimensions [number_of_samples, spatial_dimension, spatial_dimension]
            initial_diffusion_time : initial diffusion time. Dimensionless tensor.
            final_diffusion_time : final diffusion time. Dimensionless tensor.
        """
        super().__init__()
        self.sde_type = sampling_parameters.sde_type
        self.noise_parameters = noise_parameters
        self.exploding_variance = VarianceScheduler(noise_parameters)
        self.axl_network = axl_network
        self.atom_types = atom_types
        self.unit_cells = unit_cells  # TODO replace with AXL-L
        self.number_of_atoms = sampling_parameters.number_of_atoms
        self.spatial_dimension = sampling_parameters.spatial_dimension
        self.initial_diffusion_time = initial_diffusion_time
        self.final_diffusion_time = final_diffusion_time

    def _get_diffusion_coefficient_g_squared(
        self, diffusion_time: torch.Tensor
    ) -> torch.Tensor:
        """Get diffusion coefficient g(t)^2.

        The noise is given by sigma(t) = sigma_{min} (sigma_{max} / sigma_{min})^t
        and g(t)^2 = d/dt sigma(t)^2.

        Args:
            diffusion_time : Diffusion time.

        Returns:
            coefficient_g : the coefficient g(t)
        """
        return self.exploding_variance.get_g_squared(diffusion_time)

    def _get_diffusion_time(self, sde_time: torch.Tensor) -> torch.Tensor:
        """Get diffusion time.

        Args:
            sde_time : SDE time.

        Returns:
            diffusion_time: the time for the diffusion process.
        """
        return self.final_diffusion_time - sde_time

    def f(
        self, sde_time: torch.Tensor, flat_relative_coordinates: torch.Tensor
    ) -> torch.Tensor:
        """Drift function.

        Args:
            sde_time : time for the SDE. Dimensionless tensor.
            flat_relative_coordinates : time-dependent state. This corresponds to flattened atomic coordinates.
                Dimension: [batch_size (number of samples), natoms x spatial_dimension]

        Returns:
            f : the drift.
        """
        diffusion_time = self._get_diffusion_time(sde_time)

        sigma_normalized_scores = self.get_model_predictions(
            diffusion_time, flat_relative_coordinates, self.atom_types
        ).X  # we are only using the sigma normalized score for the relative coordinates diffusion
        flat_sigma_normalized_scores = einops.rearrange(
            sigma_normalized_scores, "batch natom space -> batch (natom space)"
        )

        g_squared = self._get_diffusion_coefficient_g_squared(diffusion_time)
        sigma = self.exploding_variance.get_sigma(diffusion_time)
        # Careful! The prefactor must account for the following facts:
        #   -  the SDE time is NEGATIVE the diffusion time; this introduces a minus sign dt_{diff} = -dt_{sde}
        #   -  what our model calculates is the NORMALIZED score (ie, Score x sigma). We must thus divide by sigma.
        prefactor = g_squared / sigma

        return prefactor * flat_sigma_normalized_scores

    def get_model_predictions(
        self,
        diffusion_time: torch.Tensor,
        flat_relative_coordinates: torch.Tensor,
        atom_types: torch.Tensor,
    ) -> AXL:
        """Get sigma normalized score.

        This is a utility method to wrap around the computation of the sigma normalized score in this context,
        dealing with dimensions and tensor reshapes as needed.

        Args:
            diffusion_time : the diffusion time. Dimensionless tensor.
            flat_relative_coordinates : the flat relative coordinates.
                Dimension [batch_size, natoms x spatial_dimensions]
            atom_types: indices for the atom types. Dimension [batch_size, natoms]

        Returns:
            model predictions: AXL with
                A: estimate of p(a_0|a_t). Dimension [batch_size, natoms, num_classes]
                X: sigma normalized score. Dimension [batch_size, natoms, spatial_dimensions]
                L: placeholder  # TODO
        """
        batch_size = flat_relative_coordinates.shape[0]
        sigma = self.exploding_variance.get_sigma(diffusion_time)
        sigmas = einops.repeat(sigma.unsqueeze(0), "1 -> batch 1", batch=batch_size)
        times = einops.repeat(
            diffusion_time.unsqueeze(0), "1 -> batch 1", batch=batch_size
        )

        relative_coordinates = einops.rearrange(
            flat_relative_coordinates,
            "batch (natom space) -> batch natom space",
            natom=self.number_of_atoms,
            space=self.spatial_dimension,
        )
        batch = {
            NOISY_AXL_COMPOSITION: AXL(
                A=atom_types,
                X=map_relative_coordinates_to_unit_cell(relative_coordinates),
                L=self.unit_cells,  # TODO
            ),
            NOISE: sigmas,
            TIME: times,
            UNIT_CELL: self.unit_cells,
            CARTESIAN_FORCES: torch.zeros_like(
                relative_coordinates
            ),  # TODO: handle forces correctly.
        }
        # Shape for the coordinates scores [batch_size, number of atoms, spatial dimension]
        model_predictions = self.axl_network(batch)
        return model_predictions

    def g(self, sde_time, y):
        """Diffusion function."""
        diffusion_time = self._get_diffusion_time(sde_time)
        g_squared = self._get_diffusion_coefficient_g_squared(diffusion_time)
        g_of_t = torch.sqrt(g_squared)

        return g_of_t * torch.ones_like(y)


class ExplodingVarianceSDEPositionGenerator(AXLGenerator):
    """Exploding Variance SDE Position Generator.

    This class generates position samples by solving a stochastic differential equation (SDE).
    It assumes that the diffusion noise is parameterized in the 'Exploding Variance' scheme.
    """

    def __init__(
        self,
        noise_parameters: NoiseParameters,
        sampling_parameters: SDESamplingParameters,
        axl_network: ScoreNetwork,
    ):
        """Init method.

        Args:
            noise_parameters : the diffusion noise parameters.
            sampling_parameters: the parameters needed for sampling.
            axl_network: the score network to use for drawing samples.
        """
        self.initial_diffusion_time = torch.tensor(0.0)
        self.final_diffusion_time = torch.tensor(1.0)

        self.noise_parameters = noise_parameters
        self.axl_network = axl_network
        self.sampling_parameters = sampling_parameters

        self.number_of_atoms = sampling_parameters.number_of_atoms
        self.spatial_dimension = sampling_parameters.spatial_dimension
        self.absolute_solver_tolerance = sampling_parameters.absolute_solver_tolerance
        self.relative_solver_tolerance = sampling_parameters.relative_solver_tolerance
        self.record = sampling_parameters.record_samples
        if self.record:
            self.sample_trajectory_recorder = SampleTrajectory()
            self.sample_trajectory_recorder.record(key="noise_parameters",
                                                   entry=dataclasses.asdict(noise_parameters))
            self.sample_trajectory_recorder.record(key="sampling_parameters",
                                                   entry=dataclasses.asdict(sampling_parameters))

    def get_sde(self, unit_cells: torch.Tensor, atom_types: torch.LongTensor) -> SDE:
        """Get SDE."""
        return SDE(
            noise_parameters=self.noise_parameters,
            sampling_parameters=self.sampling_parameters,
            axl_network=self.axl_network,
            atom_types=atom_types,
            unit_cells=unit_cells,
            initial_diffusion_time=self.initial_diffusion_time,
            final_diffusion_time=self.final_diffusion_time,
        )

    def initialize(
        self, number_of_samples: int, device: torch.device = torch.device("cpu")
    ):
        """This method must initialize the samples from the fully noised distribution."""
        relative_coordinates = torch.rand(
            number_of_samples, self.number_of_atoms, self.spatial_dimension
        ).to(device)
        atom_types = (
            torch.zeros(number_of_samples, self.number_of_atoms).long().to(device)
        )
        lattice_vectors = torch.zeros(
            number_of_samples, self.spatial_dimension * (self.spatial_dimension - 1)
        ).to(
            device
        )  # TODO placeholder
        init_composition = AXL(A=atom_types, X=relative_coordinates, L=lattice_vectors)
        return init_composition

    def sample(
        self, number_of_samples: int, device: torch.device, unit_cell: torch.Tensor
    ) -> AXL:
        """Sample.

        This method draws an AXL sample.

        Args:
            number_of_samples : number of samples to draw.
            device: device to use (cpu, cuda, etc.). Should match the PL model location.
            unit_cell: unit cell definition in Angstrom.
                Tensor of dimensions [number_of_samples, spatial_dimension, spatial_dimension]

        Returns:
            samples: samples as AXL composition.
        """
        initial_composition = map_axl_composition_to_unit_cell(
            self.initialize(number_of_samples), device
        )

        sde = self.get_sde(unit_cell, atom_types=initial_composition.A)
        sde.to(device)

        y0 = einops.rearrange(
            initial_composition.X, "batch natom space -> batch (natom space)"
        )

        sde_times = torch.linspace(
            self.initial_diffusion_time,
            self.final_diffusion_time,
            self.noise_parameters.total_time_steps,
        ).to(device)

        dt = (self.final_diffusion_time - self.initial_diffusion_time) / (
            self.noise_parameters.total_time_steps - 1
        )

        with torch.no_grad():
            # Dimensions [number of time steps, number of samples, natom x spatial_dimension]
            logger.info("Starting SDE solver...")
            ys = torchsde.sdeint(
                sde,
                y0,
                sde_times,
                method=self.sampling_parameters.method,
                dt=dt,
                adaptive=self.sampling_parameters.adaptive,
                atol=self.sampling_parameters.absolute_solver_tolerance,
                rtol=self.sampling_parameters.relative_solver_tolerance,
            )
            logger.info("SDE solver Finished.")

        if self.record:
            self.record_sample(sde, ys, sde_times)

        # only the final sde time (ie, diffusion time t0) is the real sample.
        flat_relative_coordinates = ys[-1, :, :]

        relative_coordinates = einops.rearrange(
            flat_relative_coordinates,
            "batch (natom space) -> batch natom space",
            natom=self.number_of_atoms,
            space=self.spatial_dimension,
        )

        return map_relative_coordinates_to_unit_cell(relative_coordinates)

    def record_sample(self, sde: SDE, ys: torch.Tensor, sde_times: torch.Tensor):
        """Record sample.

        This  takes care of recomputing the normalized score on the solution trajectory and record it to the
        sample trajectory object.
        Args:
            sde: the SDE object that provides drift and diffusion.
            ys: the SDE solutions.
                Dimensions [number of time steps, number of samples, natom x spatial_dimension]
            sde_times : times along the sde trajectory.

        Returns:
            None
        """
        list_normalized_scores = []
        sigmas = []
        evaluation_times = []
        # Reverse the times since the SDE times are inverted compared to the diffusion times.
        for sde_time, flat_relative_coordinates in zip(
            sde_times.flip(dims=(0,)), ys.flip(dims=(0,))
        ):
            diffusion_time = sde._get_diffusion_time(sde_time)
            sigma = sde.exploding_variance.get_sigma(diffusion_time)
            sigmas.append(sigma)
            evaluation_times.append(diffusion_time)

            with torch.no_grad():
                normalized_scores = sde.get_model_predictions(
                    diffusion_time,
                    flat_relative_coordinates,
                    sde.atom_types,
                ).X
            list_normalized_scores.append(normalized_scores)

        sigmas = torch.tensor(sigmas)
        evaluation_times = torch.tensor(evaluation_times)

        record_normalized_scores = einops.rearrange(
            torch.stack(list_normalized_scores),
            "time batch natom space  -> batch time natom space",
        )

        record_relative_coordinates = einops.rearrange(
            ys.flip(dims=(0,)),
            "times batch (natom space) -> batch times natom space",
            natom=self.number_of_atoms,
            space=self.spatial_dimension,
        )

        entry = dict(unit_cell=sde.unit_cells,
                     times=evaluation_times,
                     sigmas=sigmas,
                     relative_coordinates=record_relative_coordinates,
                     normalized_scores=record_normalized_scores
                     )

        self.sample_trajectory_recorder.record(key='sde', entry=entry)
