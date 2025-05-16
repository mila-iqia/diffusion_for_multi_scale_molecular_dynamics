from typing import List, Optional

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
    atom_relative_positions: np.ndarray,
    reference_point_relative_coordinates: np.array,
    lattice_parameters: np.array,
) -> np.ndarray:
    """Find the distance between a point and a reference point, taking into account periodicity.

    Args:
        atom_relative_positions: atom relative positions as a (natom, spatial dimension) array
        reference_point_relative_coordinates: reference point as a (spatial dimension, ) array
        lattice_parameters: lattice parameters. The lattice is assumed to be orthogonal. (spatial dimension, ) array

    Returns:
        distances as a (natom, ) array
    """
    basis_vectors = map_lattice_parameters_to_unit_cell_vectors(
        torch.tensor(lattice_parameters)
    )
    cartesian_positions = get_positions_from_coordinates(
        torch.tensor(atom_relative_positions), basis_vectors
    )

    reference_point_cartesian_coordinates = get_positions_from_coordinates(
        torch.tensor(reference_point_relative_coordinates).unsqueeze(0), basis_vectors
    )

    # TODO we assume an orthogonal box here
    box_distances_parameters = torch.diag(basis_vectors).numpy()
    distances = (
        cartesian_positions.numpy() - reference_point_cartesian_coordinates.numpy()
    )
    distances_squared = np.minimum(
        distances**2, (distances - box_distances_parameters) ** 2
    )
    distances_squared = np.minimum(
        distances_squared, (distances + box_distances_parameters) ** 2
    )
    return np.sqrt(distances_squared.sum(axis=-1))
