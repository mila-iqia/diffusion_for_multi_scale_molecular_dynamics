import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.atom_selector.base_atom_selector import \
    BaseAtomSelector
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.base_excisor import \
    BaseEnvironmentExcision
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.base_sample_maker import (
    BaseExciseSampleMaker, BaseExciseSampleMakerArguments)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.utils import (
    get_distances_from_reference_point,
    partition_relative_coordinates_for_voxels, select_occupied_voxels)
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_lattice_parameters_to_unit_cell_vectors


@dataclass(kw_only=True)
class ExciseAndRandomSampleMakerArguments(BaseExciseSampleMakerArguments):
    """Arguments for a sample generator based on the excise and repaint approach."""

    algorithm: str = "excise_and_random"
    total_number_of_atoms: int  # number of atoms in total, including the excised atoms
    random_coordinates_algorithm: str = "true_random"
    max_attempts: int = 10
    minimal_interatomic_distance: float = 0.5  # in Angstrom

    def __post_init__(self):
        """Post init."""
        super().__post_init__()
        assert self.random_coordinates_algorithm in ["true_random", "voxel_random"], (
            "Random coordinates algorithm should be true_random or voxel_random."
            f"Got {self.random_coordinates_algorithm}"
        )


class ExciseAndRandomSampleMaker(BaseExciseSampleMaker):
    """Sample maker for the excise and random approach.

    An excisor extract atomic environments with high uncertainties and atoms are placed randomly around them.
    We define two methods for the random positioning of atoms.
        - true_random: place the atoms randomly in the unit cell
        - voxel_random: split the unit cell in sub-grid (voxels) and place 1 atom in each voxel at a random location

    In both cases, if two atoms are placed within a distance minimal_distance (in Angstrom),
    we allow the sample maker to retry up to max_attempts times
    """

    def __init__(
        self,
        sample_maker_arguments: ExciseAndRandomSampleMakerArguments,
        atom_selector: BaseAtomSelector,
        environment_excisor: BaseEnvironmentExcision,
    ):
        """Init method.

        Args:
            sample_maker_arguments: arguments for the excise and repaint sample maker
            atom_selector: atomic selector class instance
            environment_excisor: atomic environment excisor
        """
        super().__init__(sample_maker_arguments, atom_selector, environment_excisor)
        self.num_atom_types = len(sample_maker_arguments.element_list)

    @staticmethod
    def generate_random_relative_coordinates(
        n_atoms: int, spatial_dimension: int = 3
    ) -> np.ndarray:
        """Create a numpy array of relative coordinates for n_atoms without constraints.

        Args:
            n_atoms: number of total atoms (constrained or not)
            spatial_dimension: number of spatial dimensions

        Returns:
            random relative coordinates in a (n_atoms, spatial_dimension) array
        """
        return np.random.random((n_atoms, spatial_dimension))

    @staticmethod
    def generate_atom_types(n_atoms: int, num_atom_types: int) -> np.array:
        """Create a list of atom types chosen randomly.

        Atom types are chosen from a uniform distribution: this doesn't account for the actual atomic species
        distribution in any given dataset.

        Args:
            n_atoms: number of atoms
            num_atom_types: number of possible atomic types

        Returns:
           atomic species indices as a numpy array of size (n_atoms,)
        """
        return np.random.randint(0, num_atom_types, size=(n_atoms,))

    @staticmethod
    def sort_atoms_indices_by_distance(
        target_point: np.array,
        atom_relative_coordinates: np.ndarray,
        lattice_parameters: np.array,
    ) -> np.array:
        """Sort atoms according to their distance to a target point.

        Distance calculations take into account the periodicity.

        Args:
            target_point: relative coordinates of the target point
            atom_relative_coordinates: atomic relative coordinates
            lattice_parameters: lattice parameters as a 1D numpy array

        Returns:
            sorted indices from the atom closest to target_point to the most distant atomic.
        """
        distances_between_target_to_atoms = get_distances_from_reference_point(
            atom_relative_coordinates, target_point, lattice_parameters
        )
        return np.argsort(distances_between_target_to_atoms)

    def generate_relative_coordinates_true_random(
        self, spatial_dimension
    ) -> np.ndarray:
        """Generate random relative coordinates when using a true random algorithm."""
        return self.generate_random_relative_coordinates(
            self.arguments.total_number_of_atoms, spatial_dimension
        )

    def generate_relative_coordinates_voxel_random(
        self, lattice_parameters
    ) -> np.ndarray:
        """Generate random relative coordinates when using a voxel_random algorithm."""
        box_size = (
            map_lattice_parameters_to_unit_cell_vectors(
                torch.tensor(lattice_parameters)
            )
            .diag()
            .numpy()
        )

        box_partition, num_voxel_per_dimension = (
            partition_relative_coordinates_for_voxels(
                box_size, self.arguments.total_number_of_atoms
            )
        )

        spatial_dimension, n_voxels = box_partition.shape

        # generate random relative coordinates between 0 and 1
        new_relative_coordinates = self.generate_random_relative_coordinates(
            self.arguments.total_number_of_atoms, spatial_dimension
        )  # shape (natom, spatial_dimension)
        # rescale by the number of voxels along each dimension
        new_relative_coordinates /= num_voxel_per_dimension

        # choose the voxels occupied by the atoms
        voxel_occupancies = select_occupied_voxels(
            n_voxels, self.arguments.total_number_of_atoms
        )
        # this is a (natom,) array
        occupied_voxel_coordinates = box_partition[
            :, voxel_occupancies
        ].transpose()  # (num_atoms, spatial_dimension)
        new_relative_coordinates = occupied_voxel_coordinates + new_relative_coordinates
        return new_relative_coordinates

    def make_single_structure(self, constrained_structure: AXL, active_atom_index: int) -> Tuple[AXL, int]:
        """Make a structure placing adding atoms at random locations in addition to those in the constrained structure.

        Args:
            constrained_structure: fixed atoms as an AXL of numpy arrays
            active_atom_index: index of the "active atom" in the input constrained_structure.

        Returns:
            new_structure: new structure as an AXL of np.array
            new_active_atom_index: index of the "active atom" in the new structure.
        """
        constrained_relative_coordinates = constrained_structure.X
        constrained_atom_types = constrained_structure.A
        lattice_parameters = constrained_structure.L
        spatial_dimension = constrained_relative_coordinates.shape[-1]

        # generate random relative coordinates and atomic species
        match self.arguments.random_coordinates_algorithm:
            case "true_random":
                new_relative_coordinates = (
                    self.generate_relative_coordinates_true_random(spatial_dimension)
                )
            case "voxel_random":
                new_relative_coordinates = (
                    self.generate_relative_coordinates_voxel_random(
                        constrained_structure.L
                    )
                )
            case _:  # noop
                new_relative_coordinates = constrained_relative_coordinates
        new_atom_types = self.generate_atom_types(
            self.arguments.total_number_of_atoms, self.num_atom_types
        )
        # replace some of the generated atoms by the constrained atoms

        atom_indices_replaced = (
            []
        )  # the atoms at those coordinates will be replaced by the constrained atoms
        new_relative_coordinates_copy = new_relative_coordinates.copy()
        # replace some relative coordinates by those of the constrained atoms
        replacement_indices = []

        new_active_atom_index = None
        for constrained_structure_atom_index, (constrained_atom_x, constrained_atom_a) in enumerate(zip(
            constrained_relative_coordinates, constrained_atom_types
        )):
            if constrained_structure_atom_index == active_atom_index:
                # We need to track the index of this atom since it is the active one.
                active_atom = True
            else:
                active_atom = False

            nearest_atom_indices = self.sort_atoms_indices_by_distance(
                constrained_atom_x, new_relative_coordinates, lattice_parameters
            )
            for (
                atom_idx
            ) in (
                nearest_atom_indices
            ):  # avoid mapping two different constrained atoms to the same target index
                if atom_idx not in atom_indices_replaced:
                    atom_indices_replaced.append(atom_idx)
                    new_relative_coordinates_copy[atom_idx] = constrained_atom_x
                    new_atom_types[atom_idx] = constrained_atom_a
                    replacement_indices.append(atom_idx)
                    if active_atom:
                        assert new_active_atom_index is None, \
                            "Trying to set the new_active_index more than once! Something is wrong."
                        new_active_atom_index = atom_idx
                    break

        # Shuffle the atom order to ensure that the constrained atoms come first.
        # We use a MASK to extract all the generated atoms; their order doesn't matter.
        # However, we extract the constrained coordinates exactly in the order defined
        # by the index array "replacement_indices": using a MASK here might shuffle their order!
        generated_atoms_mask = np.ones(len(new_relative_coordinates), dtype=bool)
        generated_atoms_mask[replacement_indices] = False

        atom_types = np.concatenate([new_atom_types[replacement_indices],
                                     new_atom_types[generated_atoms_mask]])

        relative_coordinates = np.vstack([new_relative_coordinates_copy[replacement_indices],
                                          new_relative_coordinates_copy[generated_atoms_mask]])

        new_structure = AXL(A=atom_types, X=relative_coordinates, L=lattice_parameters)

        assert new_active_atom_index is not None, "The new active atom index is not set. Something went wrong."
        # The active atom index should be one of the replacement_indices since it is part of the
        # constrained set of atoms.
        active_atom_idx = replacement_indices.index(new_active_atom_index)

        return new_structure, active_atom_idx

    @staticmethod
    def get_shortest_distance_between_atoms(
        atom_relative_coordinates: np.ndarray, lattice_parameters: np.array
    ) -> float:
        """Find the shortest distance between any two atoms.

        Args:
            atom_relative_coordinates: relative coordinates as a (n_atoms, spatial_dimension) array
            lattice_parameters: lattice parameters as a 1D array

        Returns:
            shortest interatomic distance in the input structure, taking periodicity into account
        """
        atom_distances = []
        for atom_coordinate in atom_relative_coordinates:
            distances_to_other_atoms = get_distances_from_reference_point(
                atom_relative_coordinates, atom_coordinate, lattice_parameters
            )
            # find the 2nd smallest distance - 1st being 0 by construction
            # partition is more efficient than a full sort
            shortest_distance = np.partition(distances_to_other_atoms, 1)[1]
            atom_distances.append(shortest_distance)
        return min(atom_distances)

    def make_single_sample_from_constrained_substructure(
        self, constrained_structure: AXL, active_atom_index: int
    ) -> Tuple[AXL, int]:
        """Make a structure placing adding atoms at random locations.

        This calls make_single_structure_true_random or make_single_structure_voxel_random, checks if two atoms are on
        top of each other and retries if necessary.

        Args:
            constrained_structure: fixed atoms as an AXL of numpy arrays
            active_atom_index: index of the "active atom" in the input constrained_structure.

        Returns:
            new_structure: new AXL with additional atoms placed randomly
            new_active_atom_index: index of the "active atom" in the newly created structure.
        """
        new_structure = None
        new_active_index = None
        n_constraint_atoms = constrained_structure.X.shape[0]
        assert n_constraint_atoms <= self.arguments.total_number_of_atoms, (
            f"There are more constrained atoms {n_constraint_atoms} than total number of atoms "
            f"{self.arguments.total_number_of_atoms}."
        )
        success = False
        for _ in range(self.arguments.max_attempts):
            new_structure, new_active_index = self.make_single_structure(
                constrained_structure, active_atom_index
            )

            min_interatomic_distance = self.get_shortest_distance_between_atoms(
                new_structure.X, new_structure.L
            )
            if min_interatomic_distance > self.arguments.minimal_interatomic_distance:
                success = True
                break

        if not success:
            logging.warning(f"A sample structure with all inter-atomic distances larger "
                            f"than {self.arguments.minimal_interatomic_distance} could not be "
                            f"generated in {self.arguments.max_attempts} attempts. "
                            f"The last generated structure is returned.")

        return new_structure, new_active_index

    def make_samples_from_constrained_substructure(
        self,
        substructure: AXL,
        active_atom_index: int,
        num_samples: int = 1,
    ) -> Tuple[List[AXL], List[int], List[Dict[str, Any]]]:
        """Create new samples using a constrained structure using a random sampling to place non-constrained atoms.

        This method assumes the lattice parameters in the constrained structure are already rescaled
        (box size is reduced).

        Args:
            substructure: excised substructure
            active_atom_index: index of the "active atom" in the input substructure.
            num_samples: number of samples to generate with the substructure

        Returns:
            list_sample_structures: the list of created samples. The length of the list should match num_samples.
            list_active_atom_indices: for each created sample, the index of the "active atom", ie the
                atom at the center of the excised region.
            list_info: list of samples additional information.
        """
        list_sample_structures = []
        list_active_atom_indices = []
        additional_information_on_new_structures = []
        for _ in range(num_samples):
            new_structure, new_active_index = self.make_single_sample_from_constrained_substructure(substructure,
                                                                                                    active_atom_index)
            list_sample_structures.append(new_structure)
            list_active_atom_indices.append(new_active_index)
            # additional information on generated structures can be passed here
            additional_information_on_new_structures.append(self._create_sample_info_dictionary(substructure))

        return list_sample_structures, list_active_atom_indices, additional_information_on_new_structures

    def filter_made_samples(self, structures: List[AXL]) -> List[AXL]:
        """Return identical structures."""
        return structures
