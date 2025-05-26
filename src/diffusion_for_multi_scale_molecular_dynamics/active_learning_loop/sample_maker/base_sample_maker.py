from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.base_excisor import \
    BaseEnvironmentExcision
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import (
    get_positions_from_coordinates, get_reciprocal_basis_vectors,
    get_relative_coordinates_from_cartesian_positions,
    map_lattice_parameters_to_unit_cell_vectors,
    map_numpy_unit_cell_to_lattice_parameters)


@dataclass(kw_only=True)
class BaseSampleMakerArguments:
    """Parameters controlling the sample maker method."""

    algorithm: str
    sample_box_strategy: str = "fixed"
    sample_box_size: Optional[float] = None

    element_list: List[str]

    def __post_init__(self):
        """Post init."""
        # TODO lattice parameters could be determined by the diffusion model - starting with a reasonable value
        assert self.sample_box_strategy in [
            "fixed",
            "noop",
        ], f"Sample box making strategy {self.sample_box_strategy} is not implemented."
        if self.sample_box_strategy == "fixed":
            assert self.sample_box_size is not None
            box_size = np.array(self.sample_box_size)
            unit_cell = np.diag(box_size) if box_size.ndim == 1 else box_size
            self.new_box_lattice_parameters = map_numpy_unit_cell_to_lattice_parameters(
                unit_cell
            )


class BaseSampleMaker(ABC):
    """Base class for the method making new samples."""

    def __init__(
        self, sample_maker_arguments: BaseSampleMakerArguments, device: str = "cpu"
    ):
        """Init method.

        Args:
            sample_maker_arguments: arguments defining the sample maker method
            device: device for the score network model. Defaults to cpu.
        """
        self.arguments = sample_maker_arguments
        self.sample_box_strategy = sample_maker_arguments.sample_box_strategy
        self.device = torch.device(device)

    @abstractmethod
    def make_samples(
        self,
        structure: AXL,
        uncertainty_per_atom: np.array,
    ) -> List[AXL]:
        """Create samples based on the provided structure.

        Args:
            structure: initial atomic configuration as an AXL object
            uncertainty_per_atom: uncertainty for each atom in the structure AXL.

        Returns:
            list of generated structure
        """
        pass

    @abstractmethod
    def filter_made_samples(self, structures: List[AXL]) -> List[AXL]:
        """Rules for rejecting samples.

        Args:
            structures: list of generates samples as AXL

        Returns:
            filtered list of samples as AXL
        """
        pass

    def make_filtered_samples(
        self,
        structure: AXL,
        uncertainty_per_atom: np.array,
    ) -> List[AXL]:
        """Generate a list of samples and filter them.

        This combines the make_samples and filter_made_samples in a single call.

        Args:
            structure: initial atomic configuration as an AXL object
            uncertainty_per_atom: uncertainty for each atom in the structure AXL.

        Returns:
             filtered list of samples as AXL
        """
        unfiltered_samples = self.make_samples(structure, uncertainty_per_atom)
        filtered_samples = self.filter_made_samples(unfiltered_samples)
        return filtered_samples

    def make_new_lattice_parameters(self, structure: AXL) -> np.array:
        """Get the lattice parameters for a generated structure given the initial one.

        Args:
            structure: initial atomic configuration as an AXL object

        Returns:
            lattice parameters for the generated structure
        """
        match self.arguments.sample_box_strategy:
            case "noop":
                return structure.L
            case "fixed":
                return self.arguments.new_box_lattice_parameters
            case _:  # something went wrong, an invalid box making strategy is used
                return None


@dataclass(kw_only=True)
class NoOpSampleMakerArguments(BaseSampleMakerArguments):
    """Parameters for a trivial sample maker method."""

    algorithm = "NoOpSampleMaker"
    sample_box_strategy = "noop"


class NoOpSampleMaker(BaseSampleMaker):
    """Trivial sample maker that reproduces the excised environment without modifications."""

    def make_samples(
        self,
        structure: AXL,
        uncertainty_per_atom: np.array,
    ) -> List[AXL]:
        """Noop make samples."""
        return [structure]

    def filter_made_samples(self, structures: List[AXL]) -> List[AXL]:
        """Noop filter samples."""
        return structures


@dataclass(kw_only=True)
class BaseExciseSampleMakerArguments(BaseSampleMakerArguments):
    """Parameters for a sample maker relying on an excision method to find the constrained atoms."""

    max_constrained_substructure: int = (
        -1
    )  # max number of problematic environment to consider. -1 means no limit.
    number_of_samples_per_substructure: int = (
        1  # number of samples to make for each sub-environment excised
    )


class BaseExciseSampleMaker(BaseSampleMaker):
    """Base class for a sample maker relying on an excisor to extract atomic environments with high uncertainties."""

    def __init__(
        self,
        sample_maker_arguments: BaseExciseSampleMakerArguments,
        environment_excisor: BaseEnvironmentExcision,
    ):
        """Init method."""
        super().__init__(sample_maker_arguments)
        self.environment_excisor = environment_excisor

    @abstractmethod
    def make_samples_from_constrained_substructure(
        self,
        substructure: AXL,
        num_samples: int = 1,
    ) -> List[AXL]:
        """Create new samples using a constrained structure.

        Args:
            substructure: constrained atoms described as an AXL
            num_samples: number of samples to make. Defaults to 1.

        Returns:
            list of samples created. The length of the list should match num_samples.
        """
        pass

    @staticmethod
    def embed_structure_in_new_box(
        structure_with_centered_atoms: AXL,
        new_lattice_parameters: np.array,
    ) -> AXL:
        """Replace the lattice parameters of a structure with new ones.

        This replaces the L component of the AXL object, and also rescale the relative coordinates X.
        Typically, replace a large box with a smaller one. Note that doing so translates the original atoms.

        Args:
            structure_with_centered_atoms: structure as an AXL object with atoms centered
            new_lattice_parameters: lattice parameters

        Returns:
            rescaled_structure: AXL with the new lattice parameters
        """
        # get the basis vectors for the large box
        original_basis_vectors = map_lattice_parameters_to_unit_cell_vectors(
            torch.tensor(structure_with_centered_atoms.L).float()
        )
        # basis vectors for the small box
        new_basis_vectors = map_lattice_parameters_to_unit_cell_vectors(
            torch.tensor(new_lattice_parameters).float()
        )

        # atoms are centered in the box - so the central atom coordinates should be (0.5, 0.5, ...) by definition
        reduced_coordinates_in_large_box = structure_with_centered_atoms.X

        # we can redefine the coordinates of an atom as a translation vector from the center of the box
        reduced_coordinates_as_vector_translation = (
            reduced_coordinates_in_large_box
            - np.ones_like(reduced_coordinates_in_large_box) * 0.5
        )
        # we can make those vector to vectors in the cartesian positions space
        cartesian_positions_as_vector_translation = get_positions_from_coordinates(
            torch.tensor(
                reduced_coordinates_as_vector_translation
            ).float(),  # have to cast to float explicitly
            basis_vectors=original_basis_vectors,
        )

        # we will place those atoms around the center of the new box
        spatial_dimension = reduced_coordinates_in_large_box.shape[-1]
        new_box_center_point_reduced_coordinates = (
            torch.ones(1, spatial_dimension) * 0.5
        )
        new_box_center_point_cartesian_positions = get_positions_from_coordinates(
            new_box_center_point_reduced_coordinates, basis_vectors=new_basis_vectors
        )

        # place the atoms around the center of the new box
        new_box_atoms_cartesian_positions = (
            cartesian_positions_as_vector_translation
            + new_box_center_point_cartesian_positions.numpy()
        )

        # TODO we assume the box is orthogonal here
        # check that the atoms cartesian positions are inside the new box
        new_box_size = torch.diag(new_basis_vectors)
        for d in range(spatial_dimension):
            assert (
                torch.max(new_box_atoms_cartesian_positions[:, d]) < new_box_size[d]
            ) and (
                torch.min(new_box_atoms_cartesian_positions[:, d]) > 0
            ), "Excised atoms are outside the new box. Use a larger box or smaller cutoff size for the excision."

        # convert the cartesian positions to reduce coordinates in the new box
        reciprocal_basis_vectors = get_reciprocal_basis_vectors(new_basis_vectors)
        new_box_atoms_relative_coordinates = (
            get_relative_coordinates_from_cartesian_positions(
                new_box_atoms_cartesian_positions, reciprocal_basis_vectors
            )
        )

        new_structure = AXL(
            A=structure_with_centered_atoms.A,
            X=new_box_atoms_relative_coordinates.numpy(),
            L=new_lattice_parameters,
        )

        return new_structure

    def make_samples(
        self,
        structure: AXL,
        uncertainty_per_atom: np.array,
    ) -> List[AXL]:
        """Make new samples based on the excised substructures created using the uncertainties.

        Args:
            structure: crystal structure, including atomic species, relative coordinates and lattice parameters
            uncertainty_per_atom: uncertainty associated to each atom. The order is assumed to be the same as those in
                the structure variable.

        Returns:
            created_samples: list of generated samples as AXL structures
        """
        constrained_environments_after_excision = (
            self.environment_excisor.excise_environments(
                structure, uncertainty_per_atom, center_atoms=True
            )
        )
        if (
            0
            < self.arguments.max_constrained_substructure
            < len(constrained_environments_after_excision)
        ):
            constrained_environments_after_excision = (
                constrained_environments_after_excision[
                    : self.arguments.max_constrained_substructure
                ]
            )
        created_samples = []
        for constrained_environment in constrained_environments_after_excision:
            if self.sample_box_strategy == "fixed":
                constrained_environment_in_new_box = self.embed_structure_in_new_box(
                    constrained_environment, self.arguments.new_box_lattice_parameters
                )
            # TODO use the L diffusion to get a new box
            else:  # noop
                constrained_environment_in_new_box = constrained_environment
            new_samples = self.make_samples_from_constrained_substructure(
                constrained_environment_in_new_box,
                self.arguments.number_of_samples_per_substructure,
            )
            created_samples += new_samples
        return created_samples


@dataclass(kw_only=True)
class NoOpExciseSampleMakerArguments(BaseExciseSampleMakerArguments):
    """Parameters for a trivial sample maker method."""

    algorithm: str = "NoOpSampleMaker"


class NoOpExciseSampleMaker(BaseExciseSampleMaker):
    """Trivial sample maker that reproduces the excised environment without modifications."""

    def make_samples_from_constrained_substructure(
        self,
        substructure: AXL,
        num_samples: int = 1,
    ) -> List[AXL]:
        """Create new samples using a constrained structure.

        Args:
            substructure: constrained atoms described as an AXL
            num_samples: number of samples to make. Defaults to 1.

        Returns:
            list of samples created. The length of the list should match num_samples.
        """
        return [substructure] * num_samples

    def filter_made_samples(self, structures: List[AXL]) -> List[AXL]:
        """Return identical structures."""
        return structures
