import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.axl_generator import \
    SamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, NOISY_AXL_COMPOSITION)
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    get_number_of_lattice_parameters


@dataclass(kw_only=True)
class TrajectoryInitializerParameters:
    """Parameters for trajectory initialization."""

    spatial_dimension: int = 3  # the dimension of Euclidean space where atoms live.
    num_atom_types: int  # number of atom types excluding MASK

    use_fixed_lattice_parameters: bool = False
    fixed_lattice_parameters: Optional[torch.Tensor] = None

    # the number of atoms that must be generated in a sampled configuration.
    number_of_atoms: int

    # Path to a pickle file that contains starting configuration information.
    path_to_constraint_data_pickle: Optional[str] = None

    def __post_init__(self):
        """Post init."""
        if self.use_fixed_lattice_parameters:
            assert (
                self.fixed_lattice_parameters is not None
            ), "If use_fixed_lattice_parameters is True, then fixed_lattice_parameters must be provided."
            assert self.fixed_lattice_parameters.shape[
                0
            ] == get_number_of_lattice_parameters(self.spatial_dimension), (
                f"The fixed_lattice_parameters tensor must have shape"
                f"[spatial_dimension * (spatial_dimension + 1) / 2]."
                f"Got {self.fixed_lattice_parameters.shape}."
            )

        else:
            assert self.fixed_lattice_parameters is None, (
                "fixed_lattice_parameters must be None if "
                "use_fixed_lattice_parameters is False."
            )


class TrajectoryInitializer(ABC):
    """Trajectory Initializer.

    This class is responsible for initializing a sampling trajectory, as well as
    describing its starting and end times.
    """

    def __init__(
        self, trajectory_initializer_parameters: TrajectoryInitializerParameters
    ) -> None:
        """Init method."""
        self.trajectory_initializer_parameters = trajectory_initializer_parameters
        self.spatial_dimension = trajectory_initializer_parameters.spatial_dimension
        self.number_of_atoms = trajectory_initializer_parameters.number_of_atoms
        self.masked_atom_type_index = trajectory_initializer_parameters.num_atom_types
        self.num_lattice_parameters = get_number_of_lattice_parameters(
            trajectory_initializer_parameters.spatial_dimension
        )
        self.use_fixed_lattice_parameters = (
            trajectory_initializer_parameters.use_fixed_lattice_parameters
        )
        self.fixed_lattice_parameters = (
            trajectory_initializer_parameters.fixed_lattice_parameters
        )

    @abstractmethod
    def initialize(self, number_of_samples: int, device: torch.device) -> AXL:
        """This method must initialize the samples."""
        pass

    @abstractmethod
    def create_start_time_step_index(self, number_of_discretization_steps: int) -> int:
        """This method determines the first time step index."""
        pass

    @abstractmethod
    def create_end_time_step_index(self) -> int:
        """This method determines the last time step index."""
        pass


class FullRandomTrajectoryInitializer(TrajectoryInitializer):
    """Full Random Trajectory Initializer.

    This class initializes a trajectory with a random configuration.
    It represents a full trajectory over all time steps.
    """

    def initialize(self, number_of_samples: int, device: torch.device) -> AXL:
        """This method must initialize the samples."""
        # all atoms are initialized as masked
        atom_types = (
            torch.ones(number_of_samples, self.number_of_atoms).long().to(device)
            * self.masked_atom_type_index
        )
        # relative coordinates are sampled from the uniform distribution
        relative_coordinates = torch.rand(
            number_of_samples, self.number_of_atoms, self.spatial_dimension
        ).to(device)
        if self.use_fixed_lattice_parameters:
            lattice_parameters = self.fixed_lattice_parameters.repeat(
                number_of_samples, 1
            ).to(device)
        else:
            lattice_parameters = torch.randn(
                number_of_samples, self.num_lattice_parameters
            ).to(device)
        init_composition = AXL(
            A=atom_types, X=relative_coordinates, L=lattice_parameters
        )
        return init_composition

    def create_start_time_step_index(self, number_of_discretization_steps: int) -> int:
        """This method determines the first time step index."""
        return number_of_discretization_steps

    def create_end_time_step_index(self) -> int:
        """This method determines the last time step index."""
        return 0


class StartFromConstraintTrajectoryInitializer(TrajectoryInitializer):
    """Full Random Trajectory Initializer.

    This class initializes a trajectory with a random configuration.
    It represents a full trajectory over all time steps.
    """

    def __init__(
        self, trajectory_initializer_parameters: TrajectoryInitializerParameters
    ) -> None:
        """Init method."""
        super().__init__(trajectory_initializer_parameters)
        self.start_time_step_index, self.noisy_starting_composition = (
            self._read_pickle_data(
                trajectory_initializer_parameters.path_to_constraint_data_pickle
            )
        )

    def _read_pickle_data(self, path_to_constraint_data_pickle: str):
        """Read a pickle data file that contains the starting index and the starting composition."""
        assert os.path.isfile(
            path_to_constraint_data_pickle
        ), f"The file {path_to_constraint_data_pickle} does not exist. Review input."

        data = torch.load(path_to_constraint_data_pickle)

        noisy_starting_composition = data[NOISY_AXL_COMPOSITION]
        start_time_step_index = data["start_time_step_index"]
        return start_time_step_index, noisy_starting_composition

    def initialize(self, number_of_samples: int, device: torch.device) -> AXL:
        """This method must initialize the samples."""
        batch_size = self.noisy_starting_composition.X.shape[0]
        assert number_of_samples == batch_size, (
            "The number of samples requested is inconsistent with the number of "
            "constrained configurations in the data pickle. "
            "Something is probably inconsistent: stopping here, review inputs."
        )

        atom_types = self.noisy_starting_composition.A.to(device)
        relative_coordinates = self.noisy_starting_composition.X.to(device)
        lattice_vectors = self.noisy_starting_composition.L.to(device)

        init_composition = AXL(A=atom_types, X=relative_coordinates, L=lattice_vectors)

        return init_composition

    def create_start_time_step_index(self, number_of_discretization_steps: int) -> int:
        """This method determines the first time step index."""
        return self.start_time_step_index

    def create_end_time_step_index(self) -> int:
        """This method determines the last time step index."""
        return 0


def instantiate_trajectory_initializer(
    sampling_parameters: SamplingParameters,
    path_to_constraint_data_pickle: Union[str, None] = None,
) -> TrajectoryInitializer:
    """Instantiate a trajectory initializer.

    Args:
        sampling_parameters: Sampling parameters
        path_to_constraint_data_pickle: path to constraint data pickle

    Returns:
        TrajectoryInitializer: a trajectory initializer object.
    """
    params = TrajectoryInitializerParameters(
        spatial_dimension=sampling_parameters.spatial_dimension,
        num_atom_types=sampling_parameters.num_atom_types,
        number_of_atoms=sampling_parameters.number_of_atoms,
        use_fixed_lattice_parameters=sampling_parameters.use_fixed_lattice_parameters,
        fixed_lattice_parameters=sampling_parameters.fixed_lattice_parameters,
        path_to_constraint_data_pickle=path_to_constraint_data_pickle,
    )

    if path_to_constraint_data_pickle:
        return StartFromConstraintTrajectoryInitializer(params)
    else:
        return FullRandomTrajectoryInitializer(params)
