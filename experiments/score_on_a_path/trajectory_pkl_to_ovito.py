from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from pymatgen.core import Lattice, Structure
from tqdm import tqdm

from diffusion_for_multi_scale_molecular_dynamics.analysis.sample_trajectory_analyser import (
    SampleTrajectoryAnalyser,
)
from diffusion_for_multi_scale_molecular_dynamics.data.element_types import ElementTypes
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL

TRAJECTORY_PATH = (
    "/Users/simonblackburn/projects/courtois2024/experiments/double_linear_debug/samples/baseline_ac/"
    + "bad_baseline"
)
TRAJECTORY_PATH = Path(TRAJECTORY_PATH)
OUTPUT_PATH = TRAJECTORY_PATH / "ovito_visualization_with_jacobian"
TRAJECTORY_INDEX = list(range(0, 5))

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
UNKNOWN_ATOM_TYPE = "X"


def main():
    trajectory_analyser = SampleTrajectoryAnalyser(
        TRAJECTORY_PATH / "trajectories.pt", num_classes=2
    )

    time_indices, axls = trajectory_analyser.extract_axl("composition_i")

    with open(TRAJECTORY_PATH / "jacobians_0_64.pt", "rb") as f:
        jacobians = torch.load(f, map_location="cpu")

    jacobians = torch.diagonal(jacobians["jacobians"], dim1=-2, dim2=-1)
    spatial_dimension = axls.X.shape[-1]
    jacobians = jacobians.view(
        jacobians.shape[0], jacobians.shape[1], -1, spatial_dimension
    )
    jacobians = jacobians.sum(dim=-1)

    jacobians_time_size = jacobians.shape[0]
    time_down_res = int(axls.X.shape[1] / jacobians_time_size)
    axls = AXL(
        X=axls.X[:, ::time_down_res].flip(dims=(1,)),
        A=axls.A[:, ::time_down_res].flip(dims=(1,)),
        L=axls.L[:, ::time_down_res].flip(dims=(1,)),
    )

    padded_jacobians = torch.zeros_like(axls.X[..., 0])
    for i, t_idx in enumerate(TRAJECTORY_INDEX):
        # padded_jacobians[t_idx] = jacobians[:, i]
        padded_jacobians[t_idx] = jacobians[:, t_idx]
    atomic_properties = dict(atomic_divergence=padded_jacobians)

    divergence_diff_from_mean = padded_jacobians - padded_jacobians.mean(
        dim=-1, keepdim=True
    )
    atomic_properties["difference_from_mean"] = divergence_diff_from_mean

    for i in TRAJECTORY_INDEX:
        create_cif_files(
            elements=["Si"],
            visualization_artifacts_path=OUTPUT_PATH,
            trajectory_index=i,
            trajectory_axl_compositions=axls,
            atomic_properties=atomic_properties,
        )


def create_cif_files(
    elements: list[str],
    visualization_artifacts_path: Path,
    trajectory_index: int,
    trajectory_axl_compositions: AXL,
    atomic_properties: Optional[Dict[str, torch.Tensor]] = None,
):
    """Create cif files.

    Args:
        elements: list of unique elements present in the samples
        visualization_artifacts_path : where the various visualization artifacts should be written to disk.
        trajectory_index : the index of the trajectory to be loaded.
        trajectory_axl_compositions: AXL that contains the trajectories, where each field
            has dimension [samples, time, ...]
        atomic_properties: scalar properties for each atom as a [samples, time, n_atom] tensor

    Returns:
        None
    """
    _cif_directory_template = "xyz_files_trajectory_{trajectory_index}"
    _cif_file_name_template = "diffusion_positions_step_{time_index}.xyz"

    element_types = ElementTypes(elements)
    atom_type_map = dict()
    for element in elements:
        id = element_types.get_element_id(element)
        atom_type_map[id] = element

    mask_id = np.max(element_types.element_ids) + 1
    atom_type_map[mask_id] = UNKNOWN_ATOM_TYPE

    cif_directory = visualization_artifacts_path / _cif_directory_template.format(
        trajectory_index=trajectory_index
    )
    cif_directory.mkdir(exist_ok=True, parents=True)

    trajectory_atom_types = trajectory_axl_compositions.A[trajectory_index].numpy()
    trajectory_relative_coordinates = trajectory_axl_compositions.X[
        trajectory_index
    ].numpy()
    trajectory_lattices = trajectory_axl_compositions.L[trajectory_index].numpy()

    if atomic_properties is not None:
        atomic_properties = {
            k: v[trajectory_index].numpy() for k, v in atomic_properties.items()
        }

    for time_idx, (atom_types, relative_coordinates, basis_vectors) in tqdm(
        enumerate(
            zip(
                trajectory_atom_types,
                trajectory_relative_coordinates,
                trajectory_lattices,
            )
        ),
        "Write CIFs",
    ):

        lattice = Lattice(matrix=basis_vectors, pbc=(True, True, True))
        species = list(map(atom_type_map.get, atom_types))

        if atomic_properties is not None:
            site_properties = {
                k: v[time_idx].tolist() for k, v in atomic_properties.items()
            }
        else:
            site_properties = None

        structure = Structure(
            lattice=lattice,
            species=species,
            coords=relative_coordinates,
            coords_are_cartesian=False,
            site_properties=site_properties,
        )

        structure_to_ovito(
            structure,
            str(cif_directory / _cif_file_name_template.format(time_index=time_idx)),
            properties=site_properties.keys(),
        )

        # structure.to_file(
        #  str(cif_directory / _cif_file_name_template.format(time_index=time_idx))
        # )

        # xyz = XYZ(structure)
        # xyz.write_file(
        #    str(cif_directory / _cif_file_name_template.format(time_index=time_idx))
        # )

        # writer = CifWriter(structure, write_site_properties=True)
        # writer.write_file(
        #     str(cif_directory / _cif_file_name_template.format(time_index=time_idx))
        # )


def structure_to_ovito(
    structure: Structure,
    output_name: str,
    properties: Optional[Union[str, List[str]]] = None,
):
    """Convert pymatgen structure to ovito readable

    Args:
        structure: pymatgen structure to convert
        lattice: lattice parameters in a 3x3 numpy array
        output_name: name of resulting file. An .xyz extension is added if not already in the name.
    """
    lattice = structure.lattice._matrix
    lattice = list(
        map(str, lattice.flatten())
    )  # flatten and convert to string for formatting
    lattice_str = 'Lattice="' + " ".join(lattice) + '" Origin="0 0 0" pbc="T T T"'

    n_atom = len(structure.sites)
    if properties is None:
        properties = []
    elif properties is not None and isinstance(properties, str):
        properties = [properties]

    xyz_txt = f"{n_atom}\n"
    xyz_txt += lattice_str + " Properties=pos:R:3"
    for prop in properties:
        xyz_txt += f":{prop}:R:1"
    xyz_txt += "\n"
    for i in range(n_atom):
        positions_values = structure.sites[i].coords
        xyz_txt += " ".join(map(str, positions_values))
        for prop in properties:
            prop_value = structure.sites[i].properties.get(prop, 0)
            xyz_txt += f" {prop_value}"
        xyz_txt += "\n"

    if not output_name.endswith(".xyz"):
        output_name += ".xyz"

    with open(output_name, "w") as f:
        f.write(xyz_txt)


if __name__ == "__main__":
    main()
