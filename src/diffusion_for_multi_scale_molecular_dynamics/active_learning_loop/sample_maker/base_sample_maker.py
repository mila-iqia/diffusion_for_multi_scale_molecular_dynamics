from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.atom_selector.base_atom_selector import \
    BaseAtomSelector
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.base_excisor import \
    BaseEnvironmentExcision
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.namespace import (
    AXL_STRUCTURE_IN_NEW_BOX, AXL_STRUCTURE_IN_ORIGINAL_BOX)
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import (
    get_positions_from_coordinates, get_reciprocal_basis_vectors,
    get_relative_coordinates_from_cartesian_positions,
    map_lattice_parameters_to_unit_cell_vectors,
    map_numpy_unit_cell_to_lattice_parameters)

_UNLIMITED_CONSTRAINED_STRUCTURE = -1


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

    def __init__(self, sample_maker_arguments: BaseSampleMakerArguments, atom_selector: BaseAtomSelector, **kwargs):
        """Init method.

        Args:
            sample_maker_arguments: arguments defining the sample maker method
            atom_selector: an atom selector object that can identify interesting atoms.
            kwargs: optional arguments.
        """
        self.arguments = sample_maker_arguments
        self.atom_selector = atom_selector
        self.sample_box_strategy = sample_maker_arguments.sample_box_strategy

    @abstractmethod
    def make_samples(
        self,
        structure: AXL,
        uncertainty_per_atom: np.array,
    ) -> Tuple[List[AXL], List[np.array], List[Dict[str, Any]]]:
        """Create samples based on the provided structure.

        Args:
            structure: initial atomic configuration as an AXL object
            uncertainty_per_atom: uncertainty for each atom in the structure AXL.

        Returns:
            list_sample_structures: list of generated structures
            list_active_environment_indices: list of arrays of atom indices, one for each sample structure, identifying
                the "active environments", namely the central atoms around which the structure are generated.
            list_extra_info: a list of dictionaries containing extra information.
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
                raise NotImplementedError(
                    f"{self.arguments.sample_box_strategy} is an invalid box making strategy."
                )

    def _create_sample_info_dictionary(self, axl_structure: AXL) -> Dict[str, Any]:
        """Create sample info dictionary.

        This utility method will create an "info" dictionary that captures the indices
        describing the atoms in the input structure.

        The usefulness of this method is in distinguishing the indices of the atoms in
        the initial structure with an uncertain atom from the atoms that are then (potentially)
        generated.

        This assumes that any new atoms added to a structure by "repaint" will always be
        appended to existing "constrained" atoms.

        Args:
            axl_structure: structure as an AXL object.

        Returns:
            sample_info_dict: A Dictionary containing useful information about the input structure.
        """
        number_of_atoms = len(axl_structure.X)
        sample_info_dict = dict(constrained_atom_indices=list(range(number_of_atoms)))
        return sample_info_dict


@dataclass(kw_only=True)
class BaseExciseSampleMakerArguments(BaseSampleMakerArguments):
    """Parameters for a sample maker relying on an excision method to find the constrained atoms."""

    max_constrained_substructure: int = (
        _UNLIMITED_CONSTRAINED_STRUCTURE  # max number of problematic environment to consider. If the value is
        # _UNLIMITED_CONSTRAINED_STRUCTURE (-1), then no limits are assumed.
    )
    number_of_samples_per_substructure: int = (
        1  # number of samples to make for each sub-environment excised
    )

    def __post_init__(self):
        """Post init checks."""
        super().__post_init__()
        assert (
            self.max_constrained_substructure == _UNLIMITED_CONSTRAINED_STRUCTURE
            or self.max_constrained_substructure > 0
        ), (
            "max_constrained_substructure should be greater than 0 or be "
            f"equal to {_UNLIMITED_CONSTRAINED_STRUCTURE}."
            f"Got {self.max_constrained_substructure}"
        )


class BaseExciseSampleMaker(BaseSampleMaker):
    """Base class for a sample maker relying on an excisor to extract atomic environments with high uncertainties."""

    def __init__(
        self,
        sample_maker_arguments: BaseExciseSampleMakerArguments,
        atom_selector: BaseAtomSelector,
        environment_excisor: BaseEnvironmentExcision,
    ):
        """Init method."""
        super().__init__(sample_maker_arguments, atom_selector)
        self.environment_excisor = environment_excisor

    @abstractmethod
    def make_samples_from_constrained_substructure(
        self,
        substructure: AXL,
        active_atom_index: int,
        num_samples: int = 1,
    ) -> Tuple[List[AXL], List[int], List[Dict[str, Any]]]:
        """Create new samples using a constrained structure.

        Args:
            substructure: constrained atoms described as an AXL.
            active_atom_index: index of the "active atom" in the input substructure.
            num_samples: number of samples to make. Defaults to 1.

        Returns:
            list_samples: the list of created samples. The length of the list should match num_samples.
            list_active_atom_indices: for each created sample, the index of the "active atom", ie the
                atom at the center of the excised region.
            list_info: list of samples additional information.
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
            torch.tensor(structure_with_centered_atoms.L)
        )
        # basis vectors for the small box
        new_basis_vectors = map_lattice_parameters_to_unit_cell_vectors(
            torch.tensor(new_lattice_parameters)
        )

        # atoms are centered in the box - so the central atom coordinates should be (0.5, 0.5, ...) by definition
        relative_coordinates_in_large_box = structure_with_centered_atoms.X

        # we can redefine the coordinates of an atom as a translation vector from the center of the box
        relative_coordinates_as_vector_translation = (
            relative_coordinates_in_large_box
            - np.ones_like(relative_coordinates_in_large_box) * 0.5
        )
        # we can map those vector as translation vectors in the cartesian positions space
        cartesian_positions_as_vector_translation = get_positions_from_coordinates(
            torch.tensor(relative_coordinates_as_vector_translation),
            basis_vectors=original_basis_vectors,
        )

        # we will place those atoms around the center of the new box
        spatial_dimension = relative_coordinates_in_large_box.shape[-1]
        new_box_center_point_relative_coordinates = (
            torch.ones(1, spatial_dimension) * 0.5
        ).to(new_basis_vectors)
        new_box_center_point_cartesian_positions = get_positions_from_coordinates(
            new_box_center_point_relative_coordinates, basis_vectors=new_basis_vectors
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
    ) -> Tuple[List[AXL], List[np.array], List[Dict[str, Any]]]:
        """Make new samples based on the excised substructures created using the uncertainties.

        Args:
            structure: crystal structure, including atomic species, relative coordinates and lattice parameters.
                The AXL elements are expected to be numpy arrays.
            uncertainty_per_atom: uncertainty associated to each atom. The order is assumed to be the same as those in
                the structure variable.

        Returns:
            list_sample_structures: list of generated samples as AXL structures
            list_active_environment_indices: list of arrays of atom indices, one for each sample structure, identifying
                the "active environments", namely the central atoms around which the structure are generated.
                Note that, since we are excising around a single atom, each "array" in the list will be composed of
                a single element.
            created_samples_info: list of dictionary with additional information on the created samples
        """
        # Start by excising the selected environments
        central_atom_indices = self.atom_selector.select_central_atoms(uncertainty_per_atom)
        constrained_environments_after_excision, excised_central_atoms_indices = (
            self.environment_excisor.excise_environments(
                structure, central_atom_indices, center_atoms=True
            )
        )
        assert len(constrained_environments_after_excision) == len(
            central_atom_indices
        ), "Number of excised environment do not match the number of central atom index. Something went wrong."

        if (
            self.arguments.max_constrained_substructure
            != _UNLIMITED_CONSTRAINED_STRUCTURE
        ) and (
            self.arguments.max_constrained_substructure
            < len(constrained_environments_after_excision)
        ):
            constrained_environments_after_excision = (
                constrained_environments_after_excision[
                    : self.arguments.max_constrained_substructure
                ]
            )
            excised_central_atoms_indices = excised_central_atoms_indices[
                : self.arguments.max_constrained_substructure
            ]

        # Build the samples based on excised regions
        list_created_samples = []
        list_samples_info = []
        list_active_environment_indices = []
        for constrained_environment, central_atom_index in zip(
            constrained_environments_after_excision, excised_central_atoms_indices
        ):
            # Box embedding does not affect the order of atoms in the structure, and so does not
            # change the index "central atom".
            if self.sample_box_strategy == "fixed":
                constrained_environment_in_new_box = self.embed_structure_in_new_box(
                    constrained_environment, self.arguments.new_box_lattice_parameters
                )
            # TODO use the L diffusion to get a new box
            else:  # noop
                constrained_environment_in_new_box = constrained_environment

            new_samples, active_atom_indices, new_samples_info = (
                self.make_samples_from_constrained_substructure(
                    substructure=constrained_environment_in_new_box,
                    active_atom_index=central_atom_index,
                    num_samples=self.arguments.number_of_samples_per_substructure,
                )
            )
            list_created_samples += new_samples
            list_active_environment_indices += [np.array([idx]) for idx in active_atom_indices]

            new_samples_info_updated = []
            for sample_info in new_samples_info:
                sample_info.update(
                    {
                        AXL_STRUCTURE_IN_ORIGINAL_BOX: constrained_environment,
                        AXL_STRUCTURE_IN_NEW_BOX: constrained_environment_in_new_box,
                    }
                )
                new_samples_info_updated.append(sample_info)
            list_samples_info += new_samples_info_updated
        return list_created_samples, list_active_environment_indices, list_samples_info
