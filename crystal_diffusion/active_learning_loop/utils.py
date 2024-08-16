from typing import List, Optional

import pandas as pd


def get_structures_for_retraining(prediction_df: pd.DataFrame,
                                  criteria_threshold: Optional[float] = None,
                                  number_of_structures: Optional[int] = None,
                                  evaluation_criteria: str = 'nbh_grades',
                                  structure_index: str = 'structure_index'
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
    assert criteria_threshold is not None or number_of_structures is not None, \
        "criteria_threshold or number_of_structures should be set."
    # get the highest evaluation_criteria for each structure i.e. only the worst atom counts for structure selection
    criteria_by_structure = prediction_df[[evaluation_criteria, structure_index]].groupby(structure_index).max()
    # find the top number_of_structures
    structures_indices = criteria_by_structure.sort_values(by=evaluation_criteria, ascending=False)
    if criteria_threshold is not None:
        structures_indices = structures_indices[structures_indices[evaluation_criteria] >= criteria_threshold]
    structures_indices = structures_indices.index.to_list()
    if number_of_structures is not None:
        structures_indices = structures_indices[:number_of_structures]
    structures_to_retrain = []
    for idx in structures_indices:
        structures_to_retrain.append(prediction_df[prediction_df[structure_index] == idx])
    return structures_to_retrain


def extract_target_region(structure_df: pd.DataFrame,
                          extraction_radius: float,
                          evaluation_criteria: str ='nbh_grades') -> pd.DataFrame:
    """Extract the atom with the worst evaluation criteria and all the atoms within a distance extraction_radious.

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
    target_position = structure_df.loc[target_atom][['x', 'y', 'z']]
    # TODO periodicity...
    structure_df.loc[:, 'distance_squared'] = structure_df.apply(
        lambda x: sum([(x[i] - target_position[i]) ** 2 for i in ['x', 'y', 'z']]), axis=1)
    atom_positions = structure_df.loc[structure_df['distance_squared'] <= extraction_radius ** 2, ['x', 'y', 'z']]
    return atom_positions