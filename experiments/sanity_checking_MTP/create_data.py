import pickle
from pathlib import Path

from src.diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.data.lammps import extract_structures_forces_and_energies_from_dump

lammps_dump_path = Path("/home/user/diffusion_for_multi_scale_molecular_dynamics/experiments/sanity_checking_MTP/artn_trajectory/dump.yaml")

output_dir = Path("/home/user/diffusion_for_multi_scale_molecular_dynamics/experiments/sanity_checking_MTP/mtp_experiments/ground_truth_data")
output_dir.mkdir(exist_ok=True)

if __name__ == '__main__':

    list_structures, list_forces, list_energies  = extract_structures_forces_and_energies_from_dump(lammps_dump_path)

    for idx, (structure, forces, energy) in enumerate(zip(list_structures, list_forces, list_energies)):

        data = {'structure': structure, 'forces': forces, 'energy': energy}

        output_file_path = str(output_dir / f"data_{idx}.pkl")

        with open(output_file_path, 'wb') as fd:
            # Use pickle.dump() to serialize and write the data to the file
            pickle.dump(data, fd)