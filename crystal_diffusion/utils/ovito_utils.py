"""Ovito utils.

The methods in this module make it easy to create an 'ovito session state'
file, which can then be loaded in the free version of Ovito. This
session state file will already be prepopulated with some common pipeline
elements.
"""
from pathlib import Path

import numpy as np
import ovito
import torch
from ovito.io import import_file
from ovito.modifiers import (AffineTransformationModifier,
                             CombineDatasetsModifier, CreateBondsModifier)
from pymatgen.core import Lattice, Structure
from tqdm import tqdm

_cif_directory_template = "cif_files_trajectory_{trajectory_index}"
_cif_file_name_template = "diffusion_positions_step_{time_index}.cif"


def create_cif_files(
    visualization_artifacts_path: Path,
    trajectory_index: int,
    ode_trajectory_pickle: Path,
):
    """Create cif files.

    Args:
        visualization_artifacts_path : where the various visualization artifacts should be written to disk.
        trajectory_index : the index of the trajectory to be loaded.
        ode_trajectory_pickle : Path to the data pickle written by ODESampleTrajectory.

    Returns:
        None
    """
    data = torch.load(ode_trajectory_pickle, map_location=torch.device("cpu"))

    cif_directory = visualization_artifacts_path / _cif_directory_template.format(
        trajectory_index=trajectory_index
    )
    cif_directory.mkdir(exist_ok=True, parents=True)

    basis_vectors = data["unit_cell"][trajectory_index].numpy()
    lattice = Lattice(matrix=basis_vectors, pbc=(True, True, True))
    trajectory_relative_coordinates = data["relative_coordinates"][
        trajectory_index
    ].numpy()

    for time_idx, relative_coordinates in tqdm(
        enumerate(trajectory_relative_coordinates), "Write CIFs"
    ):
        number_of_atoms = relative_coordinates.shape[0]
        species = number_of_atoms * ["Si"]

        structure = Structure(
            lattice=lattice,
            species=species,
            coords=relative_coordinates,
            coords_are_cartesian=False,
        )

        structure.to_file(
            str(cif_directory / _cif_file_name_template.format(time_index=time_idx))
        )


def create_ovito_session_state(
    visualization_artifacts_path: Path,
    trajectory_index: int,
    cell_scale_factor: int = 2,
    reference_cif_file: Path = None,
    cutoff_dict={"Si": 3.2, "H": 3.2},
):
    """Create Ovito session state.

    Write a 'session state' file that can be loaded into the free version of Ovito.

    Args:
        visualization_artifacts_path : where the various visualization artifacts should be written to disk.
        trajectory_index : the index of the trajectory to be loaded.
        cell_scale_factor : factor by which the cell will be modified. This is to mimic smaller atom size.
        reference_cif_file [Optional]: path to a cif file that should be added as a reference data source.
        cutoff_dict: Same particle cutoff used in bond creation.

    Returns:
        None
    """
    cif_directory = (
        visualization_artifacts_path / f"cif_files_trajectory_{trajectory_index}"
    )

    # Read the first structure to get the cell shape.
    structure = Structure.from_file(
        cif_directory / _cif_file_name_template.format(time_index=0)
    )

    # It is impossible to programmatically control the size of the atomic spheres from a python script.
    # By artificially making the cell larger, the effective size of the spheres appears smaller.

    # The lattice.matrix has the A, B, C vectors as rows; the target_cell should have vectors as columns.
    target_cell = (
        cell_scale_factor
        * np.vstack([structure.lattice.matrix, np.array([0.0, 0.0, 0.0])]).transpose()
    )

    cif_directory_template = str(
        cif_directory / _cif_file_name_template.format(time_index="*")
    )

    # Create the Ovito pipeline
    pipeline = import_file(cif_directory_template)
    if reference_cif_file is not None:
        # Insert the particles from a second file into the dataset.
        modifier = CombineDatasetsModifier()
        modifier.source.load(str(reference_cif_file))
        pipeline.modifiers.append(modifier)

    pipeline.modifiers.append(
        AffineTransformationModifier(
            operate_on={"particles", "cell"},
            relative_mode=False,
            target_cell=target_cell,
        )
    )
    bond_modifier = CreateBondsModifier()
    bond_modifier.cutoff *= cell_scale_factor
    bond_modifier.vis.width = 0.25
    bond_modifier.vis.color = (0.5, 0.5, 0.5)
    bond_modifier.vis.coloring_mode = ovito.vis.BondsVis.ColoringMode.ByParticle

    bond_modifier.mode = ovito.modifiers.CreateBondsModifier.Mode.Pairwise
    if reference_cif_file is not None:
        for type_a, cutoff in cutoff_dict.items():
            bond_modifier.set_pairwise_cutoff(
                type_a, type_a, cutoff=cell_scale_factor * cutoff
            )

    pipeline.modifiers.append(bond_modifier)

    pipeline.add_to_scene()
    ovito.scene.save(
        str(visualization_artifacts_path / f"trajectory_{trajectory_index}.ovito")
    )
    pipeline.remove_from_scene()  # remove or else subsequent calls superimposes pipelines in the same file.