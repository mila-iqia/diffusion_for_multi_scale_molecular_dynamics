"""From a .pt file with a list of AXL to .xyz files readable by ovito."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from pymatgen.core import Lattice, Structure
from tqdm import tqdm

from diffusion_for_multi_scale_molecular_dynamics.data.element_types import \
    ElementTypes
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL

TRAJECTORY_PATH = (
    "/Users/simonblackburn/projects/courtois2024/experiments/score_on_a_path"
)
TRAJECTORY_PATH = Path(TRAJECTORY_PATH)

UNKNOWN_ATOM_TYPE = "X"

SIGMA_INDEX = 0


def main():
    trajectory_file = TRAJECTORY_PATH / "interpolated_positions.pt"

    axls = torch.load(trajectory_file, map_location="cpu")

    axls_tensors = AXL(
        A=torch.stack([axl.A for axl in axls]),
        X=torch.stack([axl.X for axl in axls]),
        L=torch.stack([axl.L for axl in axls]),
    )

    model_predictions_file = TRAJECTORY_PATH / "model_predictions.pt"
    model_predictions = torch.load(model_predictions_file, map_location="cpu")
    selected_sigma = model_predictions["sigma"][SIGMA_INDEX]
    score_axls = model_predictions["model_predictions"]
    sigma_normalized_score = torch.stack(
        [axl.X[SIGMA_INDEX, :, :] for axl in score_axls]
    )  # time, n_atom, spatial_dimension

    atom_divergence = model_predictions["jacobians"][:, SIGMA_INDEX, :, :]
    atom_divergence = (
        torch.diagonal(atom_divergence, dim1=1, dim2=2)
        .view(atom_divergence.shape[0], -1, sigma_normalized_score.shape[-1])
        .sum(dim=-1, keepdim=True)
    )  # num_sample, num_atom, 1

    atomic_properties = dict(
        sigma_normalized_score=sigma_normalized_score,
        atomic_divergence=atom_divergence,
    )

    output_path = TRAJECTORY_PATH / f"ovito_visualization_sigma_{selected_sigma:0.4f}"
    output_path.mkdir(parents=True, exist_ok=True)

    create_xyz_files(
        elements=["Si"],
        visualization_artifacts_path=output_path,
        trajectory_index=None,
        trajectory_axl_compositions=axls_tensors,
        atomic_properties=atomic_properties,
    )


def create_xyz_files(
    elements: list[str],
    visualization_artifacts_path: Path,
    trajectory_index: Optional[int],
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
    _xyz_directory_template = "xyz_files_trajectory_{trajectory_index}"
    _xyz_file_name_template = "diffusion_positions_step_{time_index}.xyz"

    element_types = ElementTypes(elements)
    atom_type_map = dict()
    for element in elements:
        id = element_types.get_element_id(element)
        atom_type_map[id] = element

    mask_id = np.max(element_types.element_ids) + 1
    atom_type_map[mask_id] = UNKNOWN_ATOM_TYPE

    xyz_directory = visualization_artifacts_path / _xyz_directory_template.format(
        trajectory_index=trajectory_index if trajectory_index is not None else 0
    )
    xyz_directory.mkdir(exist_ok=True, parents=True)

    if trajectory_index is not None:
        trajectory_atom_types = trajectory_axl_compositions.A[trajectory_index].numpy()
        trajectory_relative_coordinates = trajectory_axl_compositions.X[
            trajectory_index
        ].numpy()
        trajectory_lattices = trajectory_axl_compositions.L[trajectory_index].numpy()
    else:
        trajectory_atom_types = trajectory_axl_compositions.A.numpy()
        trajectory_relative_coordinates = trajectory_axl_compositions.X.numpy()
        trajectory_lattices = trajectory_axl_compositions.L.numpy()

    if atomic_properties is not None and trajectory_index is not None:
        atomic_properties = {
            k: v[trajectory_index].numpy() for k, v in atomic_properties.items()
        }
        atomic_properties_dim = {k: v.shape[-1] for k, v in atomic_properties.items()}
    elif atomic_properties is not None:
        atomic_properties = {k: v.numpy() for k, v in atomic_properties.items()}
        atomic_properties_dim = {k: v.shape[-1] for k, v in atomic_properties.items()}

    for time_idx, (atom_types, relative_coordinates, basis_vectors) in tqdm(
        enumerate(
            zip(
                trajectory_atom_types,
                trajectory_relative_coordinates,
                trajectory_lattices,
            )
        ),
        "Write XYZs",
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
            str(xyz_directory / _xyz_file_name_template.format(time_index=time_idx)),
            properties=site_properties.keys() if site_properties is not None else None,
            properties_dim=atomic_properties_dim,
        )


def structure_to_ovito(
    structure: Structure,
    output_name: str,
    properties: Optional[Union[str, List[str]]] = None,
    properties_dim: Optional[Dict[str, int]] = None,
):
    """Convert pymatgen structure to ovito readable

    Args:
        structure: pymatgen structure to convert
        lattice: lattice parameters in a 3x3 numpy array
        output_name: name of resulting file. An .xyz extension is added if not already in the name.
        properties: atomic properties names
        properties_dim: atomic properties dimensions
    """
    lattice = structure.lattice._matrix
    lattice = list(
        map(str, lattice.flatten())
    )  # flatten and convert to string for formatting
    lattice_str = 'Lattice="' + " ".join(lattice) + '" Origin="0 0 0" pbc="T T T"'

    n_atom = len(structure.sites)
    if properties is None:
        properties = []
        properties_dim = []
    elif properties is not None and isinstance(properties, str):
        properties = [properties]
        assert (
            properties_dim is not None
        ), "site properties are defined, but dimensionalities are not."

    if properties_dim is not None:
        properties_dim = [properties_dim[k] for k in properties]

    assert len(properties_dim) == len(
        properties
    ), "mismatch between number of site properties names and dimensions"

    xyz_txt = f"{n_atom}\n"
    xyz_txt += lattice_str + " Properties=pos:R:3"
    for prop, prop_dim in zip(properties, properties_dim):
        xyz_txt += f":{prop}:R:{prop_dim}"
    xyz_txt += "\n"
    for i in range(n_atom):
        positions_values = structure.sites[i].coords
        xyz_txt += " ".join(map(str, positions_values))
        for prop in properties:
            prop_value = structure.sites[i].properties.get(prop, 0)
            xyz_txt += f" {' '.join(map(str, prop_value))}"
        xyz_txt += "\n"

    if not output_name.endswith(".xyz"):
        output_name += ".xyz"

    with open(output_name, "w") as f:
        f.write(xyz_txt)


if __name__ == "__main__":
    main()
