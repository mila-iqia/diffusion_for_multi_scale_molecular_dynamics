from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import (
    get_positions_from_coordinates,
    map_lattice_parameters_to_unit_cell_vectors)


def get_structures_for_retraining(
    prediction_df: pd.DataFrame,
    criteria_threshold: Optional[float] = None,
    number_of_structures: Optional[int] = None,
    evaluation_criteria: str = "nbh_grades",
    structure_index: str = "structure_index",
) -> List[pd.DataFrame]:
    """Find the structures with the worst value of the evaluation criteria.

    Args:
        prediction_df: dataframe with the atom positions, forces, uncertainty criteria (e.g. MaxVol coefficient),
            indices and structure indices
        criteria_threshold: value above which the evaluation_criteria is considered bad. Either this or
            number_of_structures should be specified. number_of_structures has priority if both are specified.
            Defaults to None.
        number_of_structures: number of structures to return. The top number_of_structures with the highest value of
            evaluation_criteria are returned. Either this or criteria_threshold should be specified. Defaults to None.
        evaluation_criteria: name of the evaluation criteria. Defaults to nbh_grades (MaxVol coefficient in MTP)
        structure_index: name of the column in the dataframe with the index identifying the structure. Defaults to
            structure_index.

    Returns:
        list of the structures with a bad evaluation criteria. Length of the list depends on criteria_threhold and
            number_of_structures.
    """
    assert (
        criteria_threshold is not None or number_of_structures is not None
    ), "criteria_threshold or number_of_structures should be set."
    # get the highest evaluation_criteria for each structure i.e. only the worst atom counts for structure selection
    criteria_by_structure = (
        prediction_df[[evaluation_criteria, structure_index]]
        .groupby(structure_index)
        .max()
    )
    # find the top number_of_structures
    structures_indices = criteria_by_structure.sort_values(
        by=evaluation_criteria, ascending=False
    )
    if criteria_threshold is not None:
        structures_indices = structures_indices[
            structures_indices[evaluation_criteria] >= criteria_threshold
        ]
    structures_indices = structures_indices.index.to_list()
    if number_of_structures is not None:
        structures_indices = structures_indices[:number_of_structures]
    structures_to_retrain = []
    for idx in structures_indices:
        structures_to_retrain.append(
            prediction_df[prediction_df[structure_index] == idx]
        )
    return structures_to_retrain


def extract_target_region(
    structure_df: pd.DataFrame,
    extraction_radius: float,
    evaluation_criteria: str = "nbh_grades",
) -> pd.DataFrame:
    """Extract the atom with the worst evaluation criteria and all the atoms within a distance extraction_radious.

    This is obsolete. The excisor methods should be used instead.

    Args:
        structure_df: dataframe with the atomic positions and the evaluation criteria (e.g. MaxVol value)
        extraction_radius: include all atoms within this distance of the targeted atom
        evaluation_criteria: name of the evaluation criteria. Defaults to nbh_grades (maxvol in MTP)

    Returns:
        dataframe with the atomic coordinates in columns x, y, z
    """
    # extract the worst ato and a region around of radius extraction_radius
    # TODO better method to determine radius: number of atoms ?
    target_atom = structure_df[evaluation_criteria].idxmax()
    target_position = structure_df.loc[target_atom][["x", "y", "z"]]
    # TODO periodicity... and pd warnings about .loc
    structure_df.loc[:, "distance_squared"] = structure_df.apply(
        lambda x: sum([(x[i] - target_position[i]) ** 2 for i in ["x", "y", "z"]]),
        axis=1,
    )
    atom_positions = structure_df.loc[
        structure_df["distance_squared"] <= extraction_radius**2,
        ["x", "y", "z", "species"],
    ]
    return atom_positions


def get_distances_from_reference_point(
    atom_relative_coordinates: np.ndarray,
    reference_point_relative_coordinates: np.array,
    lattice_parameters: np.array,
) -> np.ndarray:
    """Find the distance in Angstrom between atoms positions and a reference point, taking into account periodicity.

    Args:
        atom_relative_coordinates: atom relative coordinates as a (natom, spatial dimension) array
        reference_point_relative_coordinates: reference point as a (spatial dimension, ) array
        lattice_parameters: lattice parameters. The lattice is assumed to be orthogonal. (spatial dimension, ) array

    Returns:
        distances as a (natom, ) array
    """
    basis_vectors = map_lattice_parameters_to_unit_cell_vectors(
        torch.tensor(lattice_parameters)
    )
    cartesian_positions = get_positions_from_coordinates(
        torch.tensor(atom_relative_coordinates), basis_vectors
    )

    reference_point_cartesian_positions = get_positions_from_coordinates(
        torch.tensor(reference_point_relative_coordinates).unsqueeze(0), basis_vectors
    )

    # TODO we assume an orthogonal box here
    box_distances_parameters = torch.diag(basis_vectors).numpy()
    distances = (
        cartesian_positions.numpy() - reference_point_cartesian_positions.numpy()
    )
    distances_squared = np.minimum(
        distances**2, (distances - box_distances_parameters) ** 2
    )
    distances_squared = np.minimum(
        distances_squared, (distances + box_distances_parameters) ** 2
    )
    return np.sqrt(distances_squared.sum(axis=-1))


def find_partition_sizes(box_size: np.array, n_voxel: int) -> np.array:
    """Find a partition scheme to fit n_voxel in a box assumed to be orthorhombic.

    We use a naive guess without trying to optimize that partition. We take the best estimate for the number of voxel
    in each direction (which could be a float) and round it to the nearest integer. This does not guarantee that the
    number of proposed voxels match the desired number of voxels.

    Args:
        box_size: array containing the size of the box of size (spatial_dimension,)
        n_voxel: desired number of voxels

    Returns:
        proposed_partition: integers describing in how many segments to split each axis
        difference between the number of proposed voxels and the desired number of voxels
    """
    assert box_size.ndim == 1
    assert np.all(box_size > 0)
    box_volume = np.prod(box_size)
    spatial_dimension = box_size.shape[0]
    scaling_factor = (n_voxel / box_volume) ** (1 / spatial_dimension)
    # find integers such that their product is close to n_voxel while retaining approximately the same ratio
    # x/y, x/z, y/z as the box_size array
    # impose a minimum of 1 voxel in each dimension and cast as integers
    proposed_partition_size = (
        np.round(box_size * scaling_factor).clip(min=1).astype(int)
    )
    return proposed_partition_size


def partition_relative_coordinates_for_voxels(
    box_size: np.array, n_voxel: int
) -> Tuple[np.ndarray, np.array]:
    """Split a box in voxels of similar volumes.

    This finds the closest number of voxels to n_voxel and returns the corner of each voxel in relative coordinates.

    Args:
        box_size: array containing the size of the box in Angstrom (height, width, depth in 3D for example)
        n_voxel: number of voxels desired

    Returns:
        stacked_meshes: corner of each voxel in an array of shape (spatial dimension, number of voxels)
        proposed_partition_size: number of voxel along each dimension in an array of shape (spatial dimension)
    """
    proposed_partition_size = find_partition_sizes(box_size, n_voxel)
    grid_points = [
        np.linspace(0, 1, p, endpoint=False) for p in proposed_partition_size
    ]
    meshes = np.meshgrid(*grid_points, indexing="ij")
    stacked_meshes = np.stack(meshes).reshape(len(meshes), -1)
    return stacked_meshes, proposed_partition_size


def select_occupied_voxels(num_voxels, num_atoms) -> np.array:
    """Select randomly which voxels are occupied by atoms.

    This algorithm minimizes the number of voxels populated by more than 1 atom as much as possible.

    Args:
        num_voxels: number of available voxels
        num_atoms: number of atoms to place

    Returns:
       array of voxels indices to use
    """
    if num_atoms == num_voxels:
        return np.arange(num_voxels)
    elif num_atoms < num_voxels:
        return np.random.choice(np.arange(num_voxels), size=num_atoms, replace=False)
    else:  # num_atoms > num_voxels
        list_voxels = np.arange(num_voxels)
        double_occupied_voxels = select_occupied_voxels(
            num_voxels, num_atoms - num_voxels
        )
        return np.concatenate((list_voxels, double_occupied_voxels))
