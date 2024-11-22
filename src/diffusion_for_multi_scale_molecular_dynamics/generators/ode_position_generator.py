import dataclasses
import logging
from dataclasses import dataclass
from typing import Callable

import einops
import torch
import torchode as to
from torchode import Solution

from diffusion_for_multi_scale_molecular_dynamics.generators.axl_generator import (
    AXLGenerator, SamplingParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.score_network import \
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
class ODESamplingParameters(SamplingParameters):
    """Hyper-parameters for diffusion sampling with the ode algorithm."""

    algorithm: str = "ode"
    absolute_solver_tolerance: float = (
        1.0e-3  # the absolute error tolerance passed to the ODE solver.
    )
    relative_solver_tolerance: float = (
        1.0e-2  # the relative error tolerance passed to the ODE solver.
    )


class ExplodingVarianceODEAXLGenerator(AXLGenerator):
    """Exploding Variance ODE Position Generator.

    This class generates position samples by solving an ordinary differential equation (ODE).
    It assumes that the diffusion noise is parameterized in the 'Exploding Variance' scheme.
    """

    def __init__(
        self,
        noise_parameters: NoiseParameters,
        sampling_parameters: ODESamplingParameters,
        axl_network: ScoreNetwork,
    ):
        """Init method.

        Args:
            noise_parameters : the diffusion noise parameters.
            sampling_parameters: the parameters needed for sampling.
            axl_network : the model to use for drawing samples that predicts an AXL:
                atom types: predicts p(a_0 | a_t)
                relative coordinates: predicts the sigma normalized score
                lattice: placeholder  # TODO
        """
        self.t0 = 0.0  # The "initial diffusion time", corresponding to the physical distribution.
        self.tf = 1.0  # The "final diffusion time", corresponding to the uniform distribution.

        self.noise_parameters = noise_parameters
        self.exploding_variance = VarianceScheduler(noise_parameters)

        self.axl_network = axl_network

        assert (
            self.noise_parameters.total_time_steps >= 2
        ), "There must at least be two time steps in the noise parameters to define the limits t0 and tf."
        self.number_of_atoms = sampling_parameters.number_of_atoms
        self.spatial_dimension = sampling_parameters.spatial_dimension
        self.num_classes = (
            sampling_parameters.num_atom_types + 1
        )  # add 1 for the MASK class
        self.absolute_solver_tolerance = sampling_parameters.absolute_solver_tolerance
        self.relative_solver_tolerance = sampling_parameters.relative_solver_tolerance
        self.record = sampling_parameters.record_samples

        if self.record:
            self.sample_trajectory_recorder = SampleTrajectory()
            self.sample_trajectory_recorder.record(key="noise_parameters",
                                                   entry=dataclasses.asdict(noise_parameters))
            self.sample_trajectory_recorder.record(key="sampling_parameters",
                                                   entry=dataclasses.asdict(sampling_parameters))

    def _get_ode_prefactor(self, times):
        """Get ODE prefactor.

        The ODE for the relative coordinates is given by
            dx = [-1/2 g(t)^2 x Score] dt
        with
            g(t)^2 = d sigma(t)^2 / dt

        We can rearrange the ODE to:

            dx = -[1/2 g(t)^2 / sigma] x sigma Score
                  --------v-----------
                       Prefactor.

        The prefactor is then given by

            Prefactor = d sigma(t) / dt

        Args:
            times: the values of the time.

        Returns:
            ode prefactor: the prefactor in the ODE.
        """
        return self.exploding_variance.get_sigma_time_derivative(times)

    def generate_ode_term(
        self, unit_cell: torch.Tensor, atom_types: torch.LongTensor
    ) -> Callable:
        """Generate the ode_term needed to compute the ODE solution."""

        def ode_term(
            times: torch.Tensor,
            flat_relative_coordinates: torch.Tensor,
        ) -> torch.Tensor:
            """ODE term.

            This function is in the format required by the ODE solver.

            The ODE solver expect the features to be bi-dimensional, ie [batch, feature size].

            Args:
                times : ODE times, dimension [batch_size]
                flat_relative_coordinates : relative coordinates features for every time step, dimension
                    [batch_size, number of features].

            Returns:
                rhs: the right-hand-side of the corresponding ODE.
            """
            sigmas = self.exploding_variance.get_sigma(times)
            ode_prefactor = self._get_ode_prefactor(times)

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
                    L=unit_cell,  # TODO
                ),
                NOISE: sigmas.unsqueeze(-1),
                TIME: times.unsqueeze(-1),
                UNIT_CELL: unit_cell,  # TODO replace with AXL-L
                CARTESIAN_FORCES: torch.zeros_like(
                    relative_coordinates
                ),  # TODO: handle forces correctly.
            }

            # Shape [batch_size, number of atoms, spatial dimension]
            sigma_normalized_scores = self.axl_network(batch).X
            flat_sigma_normalized_scores = einops.rearrange(
                sigma_normalized_scores, "batch natom space -> batch (natom space)"
            )

            return -ode_prefactor.unsqueeze(-1) * flat_sigma_normalized_scores

        return ode_term

    def sample(
        self, number_of_samples: int, device: torch.device, unit_cell: torch.Tensor
    ) -> AXL:
        """Sample.

        This method draws an AXL sample.

        Args:
            number_of_samples : number of samples to draw.
            device: device to use (cpu, cuda, etc.). Should match the PL model location.
            unit_cell: unit cell definition in Angstrom.  # TODO replace with AXL-L
                Tensor of dimensions [number_of_samples, spatial_dimension, spatial_dimension]

        Returns:
            samples: samples as AXL composition
        """
        initial_composition = map_axl_composition_to_unit_cell(
            self.initialize(number_of_samples, device), device
        )

        ode_term = self.generate_ode_term(unit_cell, atom_types=initial_composition.A)

        y0 = einops.rearrange(
            initial_composition.X, "batch natom space -> batch (natom space)"
        )

        evaluation_times = torch.linspace(
            self.tf, self.t0, self.noise_parameters.total_time_steps
        ).to(device)

        t_eval = einops.repeat(
            evaluation_times, "t -> batch t", batch=number_of_samples
        )

        term = to.ODETerm(ode_term)
        step_method = to.Dopri5(term=term)

        step_size_controller = to.IntegralController(
            atol=self.absolute_solver_tolerance,
            rtol=self.relative_solver_tolerance,
            term=term,
        )
        solver = to.AutoDiffAdjoint(step_method, step_size_controller)
        # jit_solver = torch.compile(solver) # Compilation is not necessary, and breaks on the cluster...
        jit_solver = solver

        logger.info("Starting ODE solver...")
        sol = jit_solver.solve(to.InitialValueProblem(y0=y0, t_eval=t_eval))
        logger.info("ODE solver Finished.")

        if self.record:
            self.record_sample(ode_term, sol, evaluation_times, unit_cell)

        # sol.ys has dimensions [number of samples, number of times, number of features]
        # only the final time (ie, t0) is the real sample.
        flat_relative_coordinates = sol.ys[:, -1, :]

        relative_coordinates = einops.rearrange(
            flat_relative_coordinates,
            "batch (natom space) -> batch natom space",
            natom=self.number_of_atoms,
            space=self.spatial_dimension,
        )

        updated_composition = AXL(
            A=initial_composition.A, X=relative_coordinates, L=initial_composition.L
        )

        return map_axl_composition_to_unit_cell(updated_composition, device)

    def record_sample(
        self,
        ode_term: Callable,
        sol: Solution,
        evaluation_times: torch.Tensor,
        unit_cell: torch.Tensor,
    ):
        """Record sample.

        This method takes care of recomputing the normalized score on the solution trajectory and record it to the
        sample trajectory object.
        Args:
            ode_term : the Callable that is used to compute the rhs of the solved ODE.
            sol : the solution object obtained when solving the ODE.
            evaluation_times : times along the trajectory.
            unit_cell : unit cell definition in Angstrom.

        Returns:
            None
        """
        number_of_samples = sol.ys.shape[0]

        record_relative_coordinates = einops.rearrange(
            sol.ys,
            "batch times (natom space) -> batch times natom space",
            natom=self.number_of_atoms,
            space=self.spatial_dimension,
        )
        sigmas = self.exploding_variance.get_sigma(evaluation_times)
        ode_prefactor = self._get_ode_prefactor(evaluation_times)
        list_flat_normalized_scores = []
        for time_idx, (time, gamma) in enumerate(zip(evaluation_times, ode_prefactor)):
            times = time * torch.ones(number_of_samples).to(sol.ys)
            # The score network must be called again to get scores at intermediate times
            flat_normalized_score = (
                -ode_term(times=times, flat_relative_coordinates=sol.ys[:, time_idx])
                / gamma
            )
            list_flat_normalized_scores.append(flat_normalized_score)
        record_normalized_scores = einops.rearrange(
            torch.stack(list_flat_normalized_scores),
            "time batch (natom space) -> batch time natom space",
            natom=self.number_of_atoms,
            space=self.spatial_dimension,
        )

        entry = dict(times=evaluation_times,
                     sigmas=sigmas,
                     relative_coordinates=record_relative_coordinates,
                     normalized_scores=record_normalized_scores,
                     unit_cell=unit_cell,
                     stats=sol.stats,
                     status=sol.status)
        self.sample_trajectory_recorder.record(key='ode', entry=entry)

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
