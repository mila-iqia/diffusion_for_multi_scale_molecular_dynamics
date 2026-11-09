from typing import List, Union

import numpy as np
import torch
from ase import Atoms
from ase.geometry import find_mic
from pykeops.torch import LazyTensor
from pymatgen.core import Lattice, Structure

from diffusion_for_multi_scale_molecular_dynamics.oracle.base_single_point_calculator import \
    BaseSinglePointCalculator
from diffusion_for_multi_scale_molecular_dynamics.utils.lattice_utils import \
    get_relative_coordinates_lattice_vectors
from diffusion_for_multi_scale_molecular_dynamics.utils.neighbors import (
    _get_shifted_positions, get_periodic_adjacency_information,
    get_positions_from_coordinates)


def create_structure(
    basis_vectors: np.ndarray, relative_coordinates: np.ndarray, species: List[str]
) -> Structure:
    """Create structure.

    A utility method to convert various arrays in to a pymatgen structure.

    Args:
        basis_vectors: vectors that define the unit cell.
        relative_coordinates: atomic relative coordinates.
        species : the species. In the same order as relative coordinates.

    Returns:
        structure: a pymatgen structure.
    """
    lattice = Lattice(matrix=basis_vectors, pbc=(True, True, True))

    structure = Structure(
        lattice=lattice,
        species=species,
        coords=relative_coordinates,
        coords_are_cartesian=False,
    )
    return structure


def compute_distances_in_batch(
    cartesian_positions: torch.Tensor,
    unit_cell: torch.Tensor,
    max_distance: float,
) -> torch.Tensor:
    """Compute distances between atoms in a batch up to a cutoff distance.

    Args:
        cartesian_positions: atomic positions in Angstrom. (batch_size, n_atoms, spatial_dimension)
        unit_cell: lattice vectors. (batch_size, spatial_dimension, spatial_dimension)
        max_distance: cutoff distance

    Returns:
        tensor with all the distances larger than 0 and lower than max_distance
    """
    # cartesian_positions: batch_size, n_atoms, spatial_dimension tensor - in Angstrom
    # unit_cell : batch_size, spatial_dimension x spatial_dimension tensor
    # this is a similar implementation as the computation of the adjacency matrix in
    device = cartesian_positions.device
    batch_size, max_natom, spatial_dimension = cartesian_positions.shape
    radial_cutoff = torch.tensor(max_distance).to(device)
    zero = torch.tensor(0.0).to(device)

    # The relative coordinates lattice vectors have dimensions [number of lattice vectors, spatial_dimension]
    relative_lattice_vectors = get_relative_coordinates_lattice_vectors(
        number_of_shells=1, spatial_dimension=spatial_dimension
    ).to(device)
    number_of_relative_lattice_vectors = len(relative_lattice_vectors)

    # Repeat the relative lattice vectors along the batch dimension; the basis vectors could potentially be
    # different for every batch element.
    batched_relative_lattice_vectors = relative_lattice_vectors.repeat(batch_size, 1, 1)
    lattice_vectors = get_positions_from_coordinates(
        batched_relative_lattice_vectors, unit_cell
    )
    # The shifted_positions are composed of the positions, which are located within the unit cell, shifted by
    # the various lattice vectors.
    #  Dimension [batch_size, number of relative lattice vectors, max_number_of_atoms, spatial_dimension].
    shifted_positions = _get_shifted_positions(cartesian_positions, lattice_vectors)

    # KeOps will be used to compute the distance matrix, |p_i - p_j |^2, without overflowing memory.
    x_i = LazyTensor(
        cartesian_positions.view(batch_size, 1, max_natom, 1, spatial_dimension)
    )
    x_j = LazyTensor(
        shifted_positions.view(
            batch_size,
            number_of_relative_lattice_vectors,
            1,
            max_natom,
            spatial_dimension,
        )
    )

    # Symbolic matrix of squared distances
    d_ij = ((x_i - x_j) ** 2).sum(dim=4)  # sum on the spatial_dimension variable.

    # Identify the number of neighbors within the cutoff distance for every atom.
    # This triggers a real computation, which involves a 'compilation' the first time around.
    # This compilation time is only paid once per code execution.
    max_k_array = (d_ij <= radial_cutoff**2).sum_reduction(
        dim=3
    )  # sum on "j", the second 'virtual' dimension.

    # This is the maximum number of neighbors for any atom in any structure in the batch.
    # Going forward, there is no need to look beyond this number of neighbors.
    max_number_of_neighbors = int(max_k_array.max())

    # Use KeOps KNN functionalities to find neighbors and their indices.
    squared_distances, dst_indices = d_ij.Kmin_argKmin(
        K=max_number_of_neighbors, dim=3
    )  # find neighbors along "j"
    # Dimensions: [batch_size, number_of_relative_lattice_vectors, max_natom, max_number_of_neighbors]
    # The 'dst_indices' array corresponds to KeOps first 'virtual' dimension (the "i" dimension). This goes from
    # 0 to max_atom - 1 and correspond to atom indices (specifically, destination indices!).
    distances = torch.sqrt(squared_distances.flatten())
    # Identify neighbors within the radial_cutoff, but avoiding self.
    valid_neighbor_mask = torch.logical_and(
        zero < distances, distances <= radial_cutoff
    )
    return distances[valid_neighbor_mask]


def get_orthogonal_basis_vectors(
    batch_size: int, cell_dimensions: List[float]
) -> torch.Tensor:
    """Get orthogonal basis vectors.

    Args:
        batch_size: number of required repetitions of the basis vectors.
        cell_dimensions : list of dimensions that correspond to the sides of the unit cell.

    Returns:
        basis_vectors: a diagonal matrix with the dimensions along the diagonal.
    """
    basis_vectors = (
        torch.diag(torch.Tensor(cell_dimensions)).unsqueeze(0).repeat(batch_size, 1, 1)
    )
    return basis_vectors


def compute_distances(
    cartesian_positions: torch.Tensor, basis_vectors: torch.Tensor, max_distance: float
):
    """Compute distances."""
    adj_info = get_periodic_adjacency_information(
        cartesian_positions, basis_vectors, radial_cutoff=max_distance
    )

    # The following are 1D arrays of length equal to the total number of neighbors for all batch elements
    # and all atoms.
    #   bch: which batch does an edge belong to
    #   src: at which atom does an edge start
    #   dst: at which atom does an edge end
    bch = adj_info.edge_batch_indices
    src, dst = adj_info.adjacency_matrix

    cartesian_displacements = (
        cartesian_positions[bch, dst] - cartesian_positions[bch, src] + adj_info.shifts
    )
    distances = torch.linalg.norm(cartesian_displacements, dim=-1)
    # Identify neighbors within the radial_cutoff, but avoiding self.
    return distances[distances > 0.0]


def atoms_per_element(configuration: Atoms, element_order: List[str]) -> np.ndarray:
    """Count the atoms of each element in a configuration, ordered by element_order."""
    chemical_symbols = configuration.get_chemical_symbols()
    return np.array([chemical_symbols.count(element) for element in element_order], dtype=float)


def configurations_are_equivalent(
    atoms_a: Atoms, atoms_b: Atoms, position_tolerance: float = 0.1
) -> bool:
    """Whether two configurations (same atom ordering) match within a per-atom position and cell tolerance.

    Compares the cells and the minimum-image per-atom displacement; True when the cells match and the largest
    displacement is below position_tolerance (Ang).
    """
    if not np.allclose(atoms_a.cell.array, atoms_b.cell.array, atol=position_tolerance):
        return False
    displacement = atoms_b.get_positions() - atoms_a.get_positions()
    _, distances = find_mic(displacement, atoms_a.cell, atoms_a.pbc)
    return float(distances.max()) < position_tolerance


def create_perturbed_structures(
    atoms: Union[Atoms, List[Atoms]], standard_deviation: float, number_of_configurations: int
) -> List[Atoms]:
    """Create slightly perturbed copies of each input structure.

    Each structure's atoms are displaced by small Gaussian noise (ASE's rattle), number_of_configurations
    times, keeping the same cell and species. This augments a training set around known configurations; it is
    not a fully-random placement of atoms.

    Args:
        atoms: an ase.Atoms (or list of them) to perturb.
        standard_deviation: standard deviation (Angstrom) of the Gaussian displacements.
        number_of_configurations: number of perturbed configurations to generate per input structure.

    Returns:
        perturbed_structures: the perturbed configurations.
    """
    if isinstance(atoms, Atoms):
        atoms = [atoms]
    random_generator = np.random.default_rng()
    perturbed_structures = []
    for structure in atoms:
        for _ in range(number_of_configurations):
            perturbed_structure = structure.copy()
            perturbed_structure.rattle(stdev=standard_deviation, rng=random_generator)
            perturbed_structures.append(perturbed_structure)
    return perturbed_structures


def label_configurations(
    configurations: List[Atoms], single_point_calculator: BaseSinglePointCalculator
) -> List[Atoms]:
    """Compute the energy and forces of each configuration with the single-point calculator.

    Args:
        configurations: the ase.Atoms to label.
        single_point_calculator: the oracle used to evaluate them.

    Returns:
        the configurations as labelled ase.Atoms (energy and forces on their calculator).
    """
    calculations = single_point_calculator.calculate_many(configurations)
    return [calculation.to_atoms(list(range(len(calculation.atoms)))) for calculation in calculations]


def assert_orthogonal_cell(atoms: Atoms, angle_tolerance: float = 1e-3) -> None:
    """Raise a ValueError unless the configuration's cell is orthogonal (all angles are 90 degrees).

    The LAMMPS dump reader assumes an orthogonal box; a tilted (triclinic) cell would be silently misread.

    Args:
        atoms: the ase.Atoms configuration to check.
        angle_tolerance: maximum allowed deviation (degrees) of each cell angle from 90.
    """
    angles = atoms.cell.angles()
    if not np.allclose(angles, 90.0, atol=angle_tolerance):
        raise ValueError(
            f"The configuration's cell is not orthogonal (angles {angles} degrees); only orthogonal cells "
            "are supported. Build a cubic/orthorhombic cell, e.g. ase.build.bulk(..., cubic=True)."
        )
