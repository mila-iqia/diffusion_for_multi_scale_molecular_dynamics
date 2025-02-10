from pathlib import Path
from typing import Any, AnyStr, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from pymatgen.core import Lattice, Structure
from tqdm import tqdm

from diffusion_for_multi_scale_molecular_dynamics.analysis.ovito_utilities.ovito_utils import \
    UNKNOWN_ATOM_TYPE
from diffusion_for_multi_scale_molecular_dynamics.data.element_types import \
    ElementTypes
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL

CIF_DIRECTORY_TEMPLATE = "cif_files_trajectory_{trajectory_index}"
CIF_FILENAME_TEMPLATE = "diffusion_positions_step_{time_index}.cif"

XYZ_DIRECTORY_TEMPLATE = "xyz_files_trajectory_{trajectory_index}"
XYZ_FILENAME_TEMPLATE = "diffusion_positions_step_{time_index}.xyz"


def create_cif_files(
    elements: list[str],
    visualization_artifacts_path: Path,
    trajectory_index: int,
    trajectory_axl_compositions: AXL,
):
    """Create cif files.

    Args:
        elements: list of unique elements present in the samples
        visualization_artifacts_path : where the various visualization artifacts should be written to disk.
        trajectory_index : the index of the trajectory to be loaded.
        trajectory_axl_compositions: AXL that contains the trajectories, where each field
            has dimension [samples, time, ...]

    Returns:
        None
    """
    create_io_files(
        elements,
        visualization_artifacts_path,
        trajectory_index,
        trajectory_axl_compositions,
        atomic_properties=None,
        format="cif",
    )


def create_xyz_files(
    elements: list[str],
    visualization_artifacts_path: Path,
    trajectory_index: Optional[int],
    trajectory_axl_compositions: AXL,
    atomic_properties: Optional[Dict[str, torch.Tensor]],
):
    """Create xyz files.

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
    create_io_files(
        elements,
        visualization_artifacts_path,
        trajectory_index,
        trajectory_axl_compositions,
        atomic_properties,
        format="xyz",
    )


def create_io_files(
    elements: list[str],
    visualization_artifacts_path: Path,
    trajectory_index: Optional[int],
    trajectory_axl_compositions: AXL,
    atomic_properties: Optional[Dict[str, torch.Tensor]],
    format: str,
):
    """Create cif files.

    Args:
        elements: list of unique elements present in the samples
        visualization_artifacts_path : where the various visualization artifacts should be written to disk.
        trajectory_index : the index of the trajectory to be loaded.
        trajectory_axl_compositions: AXL that contains the trajectories, where each field
            has dimension [samples, time, ...]
        atomic_properties: scalar properties for each atom as a [samples, time, n_atom] tensor
        format: format of output file: should be cif or xyz.

    Returns:
        None
    """
    atom_type_map = _get_atom_type_map(elements)

    list_axl = get_list_trajectory_AXLs(trajectory_axl_compositions, trajectory_index)

    number_of_time_steps = len(list_axl)
    list_site_properties, atomic_properties_dim = (
        get_list_site_properties_and_atomic_properties_dim(
            atomic_properties, number_of_time_steps, trajectory_index
        )
    )

    match format:
        case "cif":
            writer = CifStructureWriter(
                visualization_artifacts_path, trajectory_index, atomic_properties_dim
            )
        case "xyz":
            writer = XyzStructureWriter(
                visualization_artifacts_path, trajectory_index, atomic_properties_dim
            )
        case _:
            raise NotImplementedError(f"no such format {format}")

    for time_idx, (axl, site_properties) in tqdm(
        enumerate(zip(list_axl, list_site_properties)), "Write XYZs"
    ):
        structure = Structure(
            lattice=Lattice(matrix=axl.L, pbc=(True, True, True)),
            species=list(map(atom_type_map.get, axl.A)),
            coords=axl.X,
            coords_are_cartesian=False,
            site_properties=site_properties,
        )
        writer.write(structure, time_idx, site_properties)


def get_list_site_properties_and_atomic_properties_dim(
    atomic_properties: Dict[AnyStr, Any],
    number_of_time_steps: int,
    trajectory_index: Optional[int],
) -> Tuple[List[Optional[Dict[AnyStr, Any]]], Optional[Dict[AnyStr, Any]]]:
    """Get list site properties and atomic properties dimensions.

    Extract the relevant information from input dictionaries.

    Args:
        atomic_properties: dictionary of atomic properties. If None, the return will be arrays of None with the
            right length.
        number_of_time_steps: the expected number of time steps that should be present in the
            atomic properties dictionary.
        trajectory_index: the trajectory index to extract. If absent, it is assumed that a single trajectory is present.

    Returns:
        list_site_properties: a list of site properties, each element corresponding to a different time step.
        atomic_properties_dim: the spatial dimensions of the site properties.
    """
    if atomic_properties is None:
        list_site_properties = number_of_time_steps * [None]
        atomic_properties_dim = None
        return list_site_properties, atomic_properties_dim

    atomic_properties_dim = {k: v.shape[-1] for k, v in atomic_properties.items()}

    if trajectory_index is None:
        atomic_properties = {k: v.numpy() for k, v in atomic_properties.items()}
    else:
        atomic_properties = {
            k: v[trajectory_index].numpy() for k, v in atomic_properties.items()
        }

    for key, values in atomic_properties.items():
        assert (
            len(values) == number_of_time_steps
        ), f"The number of time steps in property {key} is inconsistent with expectation."

    list_site_properties = []
    for time_idx in range(number_of_time_steps):
        site_properties = {
            k: v[time_idx].tolist() for k, v in atomic_properties.items()
        }
        list_site_properties.append(site_properties)

    return list_site_properties, atomic_properties_dim


def _get_atom_type_map(elements):
    """Get atom type map."""
    element_types = ElementTypes(elements)
    atom_type_map = dict()
    for element in elements:
        id = element_types.get_element_id(element)
        atom_type_map[id] = element
    mask_id = np.max(element_types.element_ids) + 1
    atom_type_map[mask_id] = UNKNOWN_ATOM_TYPE
    return atom_type_map


def get_list_trajectory_AXLs(
    trajectory_axl_compositions: AXL, trajectory_index: Optional[int]
):
    """Get list trajectory AXLs.

    This method extracts numpy arrays from the AXL compositions and creates a list of single time step AXLs.

    Args:
        trajectory_axl_compositions: AXL composition that contains multiple time steps, and potentially multiple
            trajectories.
        trajectory_index: If present, which trajectory should be extracted. If absent, it is assumed that
            trajectory_axl_compositions contains a single trajectory.

    Returns:
        list_axl: the selected trajectory, as a list of single time step AXLs. These are composed of
            io-ready numpy objects.
    """
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

    list_axl = []
    for atom_types, relative_coordinates, basis_vectors in zip(
        trajectory_atom_types, trajectory_relative_coordinates, trajectory_lattices
    ):
        list_axl.append(AXL(A=atom_types, X=relative_coordinates, L=basis_vectors))

    return list_axl


class StructureWriter:
    """Class to write trajectories to file to be used by Ovito."""

    def __init__(
        self,
        visualization_artifacts_path: Path,
        trajectory_index: Optional[int],
        properties_dim: Optional[Dict[str, int]],
    ):
        """Init method."""
        self.visualization_artifacts_path = visualization_artifacts_path
        self.visualization_artifacts_path.mkdir(parents=True, exist_ok=True)
        self.properties_dim = properties_dim
        self.trajectory_index = trajectory_index
        self.results_directory = self._create_results_directory()

    def _create_results_directory(self):
        raise NotImplementedError("Must be implemented in derived class")

    def write(
        self,
        structure: Structure,
        time_index: int,
        site_properties: Optional[Union[str, List[str]]],
    ):
        """Write structure to disk."""
        raise NotImplementedError("Must be implemented in derived class")


class CifStructureWriter(StructureWriter):
    """Class to write cif files."""

    def _create_results_directory(self):
        cif_directory = (
            self.visualization_artifacts_path
            / CIF_DIRECTORY_TEMPLATE.format(
                trajectory_index=(
                    self.trajectory_index if self.trajectory_index is not None else 0
                )
            )
        )
        cif_directory.mkdir(exist_ok=True, parents=True)
        return cif_directory

    def write(
        self,
        structure: Structure,
        time_index: int,
        site_properties: Optional[Union[str, List[str]]],
    ):
        """Convert pymatgen structure to ovito readable format.

        NOTE: the site properties are ignored by the CIF writer.

        Args:
            structure: pymatgen structure to convert
            time_index: the time index of the structure being written.
            site_properties: atomic properties names
        """
        structure.to_file(
            str(
                self.results_directory
                / CIF_FILENAME_TEMPLATE.format(time_index=time_index)
            )
        )


class XyzStructureWriter(StructureWriter):
    """Class to write xyz files."""

    def _create_results_directory(self):
        xyz_directory = (
            self.visualization_artifacts_path
            / XYZ_DIRECTORY_TEMPLATE.format(
                trajectory_index=(
                    self.trajectory_index if self.trajectory_index is not None else 0
                )
            )
        )
        xyz_directory.mkdir(exist_ok=True, parents=True)
        return xyz_directory

    def write(
        self,
        structure: Structure,
        time_index: int,
        site_properties: Optional[Union[str, List[str]]],
    ):
        """Convert pymatgen structure to ovito readable format.

        Args:
            structure: pymatgen structure to convert
            time_index: the time index of the structure being written.
            site_properties: atomic properties names
        """
        output_name = str(
            self.results_directory / XYZ_FILENAME_TEMPLATE.format(time_index=time_index)
        )

        lattice = structure.lattice._matrix
        lattice = list(
            map(str, lattice.flatten())
        )  # flatten and convert to string for formatting
        lattice_str = 'Lattice="' + " ".join(lattice) + '" Origin="0 0 0" pbc="T T T"'

        n_atom = len(structure.sites)
        if site_properties is None:
            site_properties = []
            properties_dim = []
        elif site_properties is not None and isinstance(site_properties, str):
            site_properties = [site_properties]
            assert (
                self.properties_dim is not None
            ), "site properties are defined, but dimensionalities are not."

        if self.properties_dim is not None:
            properties_dim = [self.properties_dim[k] for k in site_properties]

        assert len(properties_dim) == len(
            site_properties
        ), "mismatch between number of site properties names and dimensions"

        xyz_txt = f"{n_atom}\n"
        xyz_txt += lattice_str + " Properties=pos:R:3"
        for prop, prop_dim in zip(site_properties, properties_dim):
            xyz_txt += f":{prop}:R:{prop_dim}"
        xyz_txt += "\n"
        for i in range(n_atom):
            positions_values = structure.sites[i].coords
            xyz_txt += " ".join(map(str, positions_values))
            for prop in site_properties:
                prop_value = structure.sites[i].properties.get(prop, 0)
                xyz_txt += f" {' '.join(map(str, prop_value))}"
            xyz_txt += "\n"

        if not output_name.endswith(".xyz"):
            output_name += ".xyz"

        with open(output_name, "w") as f:
            f.write(xyz_txt)
