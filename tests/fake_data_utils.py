from collections import namedtuple
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import yaml

from crystal_diffusion.namespace import (CARTESIAN_FORCES, CARTESIAN_POSITIONS,
                                         RELATIVE_COORDINATES)

Configuration = namedtuple("Configuration",
                           ["spatial_dimension", CARTESIAN_POSITIONS, CARTESIAN_FORCES, RELATIVE_COORDINATES,
                            "types", "ids", "cell_dimensions",
                            "potential_energy", "kinetic_energy", "energy"])


def generate_fake_configuration(spatial_dimension: int, number_of_atoms: int):
    """Generate fake configuration.

    Args:
        spatial_dimension : dimension of space. Should be 1, 2 or 3.
        number_of_atoms : how many atoms to generate.

    Returns:
        configuration: a configuration object with all the data describing a configuration.
    """

    relative_coordinates = np.random.rand(number_of_atoms, spatial_dimension)

    cell_dimensions = 5. + 5. * np.random.rand(spatial_dimension)  # make sure the cell is big.
    unit_cell_vectors = np.diag(cell_dimensions)
    positions = np.dot(relative_coordinates, unit_cell_vectors)
    potential_energy = np.random.rand()
    kinetic_energy = np.random.rand()
    energy = potential_energy + kinetic_energy

    return Configuration(spatial_dimension=spatial_dimension,
                         relative_coordinates=relative_coordinates,
                         cartesian_positions=positions,
                         cartesian_forces=np.random.rand(number_of_atoms, spatial_dimension),
                         types=np.random.randint(1, 10, number_of_atoms),
                         ids=np.arange(1, number_of_atoms + 1),
                         cell_dimensions=cell_dimensions,
                         potential_energy=potential_energy,
                         kinetic_energy=kinetic_energy,
                         energy=energy)


def get_configuration_runs(number_of_runs, spatial_dimension, number_of_atoms):
    """Generate multiple random configuration runs, each composed of many different configurations."""
    list_configurations = []
    for _ in range(number_of_runs):
        number_of_configs = np.random.randint(1, 16)
        configurations = [generate_fake_configuration(spatial_dimension=spatial_dimension,
                                                      number_of_atoms=number_of_atoms)
                          for _ in range(number_of_configs)]
        list_configurations.append(configurations)

    return list_configurations


def generate_parse_dump_output_dataframe(configurations: List[Configuration]) -> pd.DataFrame:
    """Generate parse lammps run

    Args:
        configurations : a list of configuration objects.

    Returns:
        df: the expected output of parse_lammps_output.
    """
    rows = []
    for configuration in configurations:
        row = dict(box=configuration.cell_dimensions, id=list(configuration.ids), type=list(configuration.types))
        for coordinates, name in zip(configuration.cartesian_positions.transpose(), ['x', 'y', 'z']):
            row[name] = list(coordinates)

        for coordinate_forces, name in zip(configuration.cartesian_forces.transpose(), ['fx', 'fy', 'fz']):
            row[name] = list(coordinate_forces)

        rows.append(row)
    return pd.DataFrame(rows)


def create_dump_single_record(configuration: Configuration, timestep: int) -> Dict[str, Any]:

    spatial_dimension = configuration.spatial_dimension

    box = [[0, float(dimension)] for dimension in configuration.cell_dimensions]

    # keywords should be of the form : [id, type, x, y, z, fx, fy, fz, ]
    keywords = ['id', 'type']

    for direction, _ in zip(['x', 'y', 'z'], range(spatial_dimension)):
        keywords.append(direction)

    for force_direction, _ in zip(['fx', 'fy', 'fz'], range(spatial_dimension)):
        keywords.append(force_direction)

    # Each row of data should be a list in the same order as the keywords
    data = []

    for id, type, position, force in (
            zip(configuration.ids, configuration.types, configuration.cartesian_positions,
                configuration.cartesian_forces)):
        row = [int(id), int(type)] + [float(p) for p in position] + [float(f) for f in force]
        data.append(row)

    document = dict(creator='fake LAMMPS for tests',
                    timestep=timestep,
                    natoms=len(configuration.ids),
                    boundary=2 * spatial_dimension * ['p'],
                    box=box,
                    keywords=keywords,
                    data=data)
    return document


def create_dump_yaml_documents(configurations: List[Configuration]) -> List[Dict[str, Any]]:
    docs = []
    for timestep, configuration in enumerate(configurations):
        docs.append(create_dump_single_record(configuration, timestep))

    return docs


def create_thermo_yaml_documents(configurations: List[Configuration]) -> List[Dict[str, Any]]:

    keywords = ['Step', 'Temp', 'KinEng', 'PotEng', 'E_bond', 'E_angle',
                'E_dihed', 'E_impro', 'E_vdwl', 'E_coul', 'E_long', 'Press']

    number_of_keywords = len(keywords)

    data = []
    for timestep, configuration in enumerate(configurations):
        row = ([timestep]
               + [float(np.random.rand())]
               + [float(configuration.kinetic_energy), float(configuration.potential_energy)]
               + [float(np.random.rand()) for _ in range(number_of_keywords - 4)])
        data.append(row)

    document = dict(keywords=keywords, data=data)
    return [document]


def write_to_yaml(documents: List[Dict[str, Any]], output_file_path: str):
    """Write to yaml."""
    with open(output_file_path, "w") as fd:
        # This output format is not perfectly identical to what LAMMPS output, but very similar.
        yaml.dump_all(documents, fd, sort_keys=False, default_flow_style=None, width=1000)


def generate_parquet_dataframe(configurations: List[Configuration]) -> pd.DataFrame:
    rows = []
    for configuration in configurations:
        # There are two possible flattening orders; C-style (c) and fortran-style (f). For an array of the form
        #   A = [ v1, v2] --> A.flatten(order=c) = [v1, v2, v3, v4],   A.flatten(order=f) = [v1, v3, v2, v4].
        #       [ v3, v4]
        # C-style flattening is the correct convention to interoperate with pytorch reshaping operations.
        relative_positions = configuration.relative_coordinates.flatten(order='c')
        positions = configuration.cartesian_positions.flatten(order='c')
        forces = configuration.cartesian_forces.flatten(order='c')
        number_of_atoms = len(configuration.ids)
        box = configuration.cell_dimensions
        row = dict(natom=number_of_atoms,
                   box=box,
                   type=configuration.types,
                   potential_energy=configuration.potential_energy,
                   cartesian_positions=positions,
                   relative_coordinates=relative_positions,
                   cartesian_forces=forces,
                   )

        rows.append(row)
    return pd.DataFrame(rows)


def find_aligning_permutation(first_2d_array: torch.Tensor, second_2d_array: torch.Tensor, tol=1e-6) -> torch.Tensor:
    """Find aligning permutation, assuming the input two arrays contain the same information.

    This function computes and stores all distances. This scales quadratically, which is not good, but it should
    be ok for testing purposes.
    """
    assert first_2d_array.shape == second_2d_array.shape, "Incompatible shapes."
    assert len(first_2d_array.shape) == 2, "Unexpected shapes."

    number_of_vectors = first_2d_array.shape[0]

    distances = torch.sum((first_2d_array[:, None, :] - second_2d_array[None, :, :])**2, dim=-1)

    matching_indices = (distances < tol).nonzero()

    assert torch.allclose(matching_indices[:, 0], torch.arange(number_of_vectors)), \
        "There isn't exactly a one-to-one match between the two arrays"

    permutation_indices = matching_indices[:, 1]

    return permutation_indices
