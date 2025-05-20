import pickle
from pathlib import Path

from maml.apps.pes import MTPotential

lammps_dump_path = Path("/home/user/diffusion_for_multi_scale_molecular_dynamics/experiments/sanity_checking_MTP/artn_trajectory/dump.yaml")

mtp_experiment_path = Path("/home/user/diffusion_for_multi_scale_molecular_dynamics/experiments/sanity_checking_MTP/mtp_experiments")
mtp_experiment_path.mkdir(exist_ok=True)

ground_truth_data_directory = Path("/home/user/diffusion_for_multi_scale_molecular_dynamics/experiments/sanity_checking_MTP/mtp_experiments/ground_truth_data")

if __name__ == '__main__':

    all_lines = []
    for idx in range(10):
        print(f"Writing data for round {idx}")

        data_file_path = str(ground_truth_data_directory / f"data_{idx}.pkl")

        # To verify, you can also load it back
        with open(data_file_path, 'rb') as f:
            loaded_data = pickle.load(f)

        structure = loaded_data['structure']
        forces = loaded_data['forces']
        energy = loaded_data['energy']

        # The MTP Potential object from MAML implements the parsing of the MTP config files. We don't want to
        # Use the full functionality of the class, just the parsing. We have to hack around to get it to work.
        dummy_virial_stress = 6 * [0]

        mtp_potential = MTPotential()

        mtp_potential.elements = [str(element) for element in structure.elements]
        lines = mtp_potential._line_up(structure, energy, forces, virial_stress=dummy_virial_stress)

        output_file = str(mtp_experiment_path / f"config_{idx}.cfg")
        with open(output_file,"w") as fd:
            fd.write(lines)


        all_lines.append(lines)
        output_file = str(mtp_experiment_path / f"train_cumulative_configs_{idx}.cfg")
        with open(output_file,"w") as fd:
            fd.write("\n".join(all_lines))
