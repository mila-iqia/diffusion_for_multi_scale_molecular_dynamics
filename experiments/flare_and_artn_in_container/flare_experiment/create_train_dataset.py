import pickle
from pathlib import Path

import numpy as np

from flare_experiment.utilities.labelled_structure import LabelledStructure
from flare_experiment.utilities.utils import parse_lammps_dump

lammps_dump_path  = "/home/user/experiments/Si_2x2x2_lammps/dump.yaml"

flare_training_data = Path("/home/user/experiments/flare_experiment/training_data/md_training_data")
flare_training_data.mkdir(exist_ok=True, parents=True)


if __name__ == '__main__':
    data_dict = parse_lammps_dump(lammps_dump_path)

    list_atoms = data_dict['atoms']
    list_forces = data_dict['forces']
    list_energies = data_dict['energy']

    list_train_indices = np.arange(1, 10)
    list_test_indices = np.arange(61, 70)

    for list_indices, dataset in zip([list_train_indices, list_test_indices], ["train", "test"]):

        for idx in list_indices:
            atoms = list_atoms[idx]
            number_of_atoms = len(atoms)
            labelled_structure = LabelledStructure(atoms=atoms,
                                                   forces=list_forces[idx],
                                                   energy=list_energies[idx],
                                                   active_set_indices=np.arange(number_of_atoms))

            file_path = flare_training_data / f"md_{dataset}_structure_{idx}.pkl"

            with open(file_path, "wb") as fd:
                pickle.dump(labelled_structure, fd)

