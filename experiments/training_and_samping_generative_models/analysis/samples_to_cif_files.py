from pathlib import Path

import torch
from pymatgen.core import Lattice, Structure
from tqdm import tqdm

from diffusion_for_multi_scale_molecular_dynamics.data.element_types import \
    ElementTypes
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, AXL_COMPOSITION)
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_lattice_parameters_to_unit_cell_vectors

samples_top_directory = Path(__file__).parent / "cif_samples"
samples_top_directory.mkdir(exist_ok=True)

base_experiment_dir = Path("/Users/brunorousseau/courtois/july26/")

list_work_directories = ["july26_si_egnn_1x1x1",
                         "july26_si_egnn_2x2x2",
                         "july26_si_egnn_3x3x3",
                         "july26_sige_egnn_1x1x1",
                         "july26_sige_egnn_2x2x2",
                         "july26_sige_egnn_3x3x3"]

list_dataset_names = ["Si_1x1x1", "Si_2x2x2", "Si_3x3x3", "SiGe_1x1x1", "SiGe_2x2x2", "SiGe_3x3x3"]

list_elements = [['Si'], ['Si'], ['Si'], ['Si', 'Ge'], ['Si', 'Ge'], ['Si', 'Ge']]

list_T = [1000, 10000]

if __name__ == "__main__":

    for elements, work_dir, dataset_name in zip(list_elements, list_work_directories, list_dataset_names):
        element_types = ElementTypes(elements)

        experiment_dir = base_experiment_dir / work_dir / "samples_from_run1/"

        for T in list_T:
            output_dir = samples_top_directory / f"{dataset_name}" / f"T={T}/"
            output_dir.mkdir(parents=True, exist_ok=True)

            energies_file_path = experiment_dir / f"sample_output_T={T}/energies.pt"
            energies = torch.load(energies_file_path)
            order = torch.argsort(energies)

            samples_file_path = experiment_dir / f"sample_output_T={T}/samples.pt"

            data = torch.load(samples_file_path, map_location="cpu")
            unordered_axl_composition = data[AXL_COMPOSITION]
            axl_composition = AXL(A=unordered_axl_composition.A[order],
                                  X=unordered_axl_composition.X[order],
                                  L=unordered_axl_composition.L[order])

            batch_lattice_parameters = axl_composition.L

            batch_basis_vectors = map_lattice_parameters_to_unit_cell_vectors(batch_lattice_parameters)

            batch_size = batch_basis_vectors.shape[0]

            atom_type_map = dict()
            for element in elements:
                id = element_types.get_element_id(element)
                atom_type_map[id] = element

            for batch_idx in tqdm(range(batch_size), "CIF : "):
                basis_vectors = batch_basis_vectors[batch_idx]
                lattice = Lattice(matrix=basis_vectors, pbc=(True, True, True))

                relative_coordinates = axl_composition.X[batch_idx].numpy()
                atom_types = axl_composition.A[batch_idx].numpy()
                species = list(map(atom_type_map.get, atom_types))

                structure = Structure(
                    lattice=lattice,
                    species=species,
                    coords=relative_coordinates,
                    coords_are_cartesian=False,
                )

                file_path = str(output_dir / f"sample_{batch_idx}.cif")
                structure.to_file(file_path)
