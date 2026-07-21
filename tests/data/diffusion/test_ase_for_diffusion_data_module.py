import ase
import ase.io
import numpy as np
import pytest
import torch
from ase.calculators.singlepoint import SinglePointCalculator

from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.ase_for_diffusion_data_module import (
    ASEForDiffusionDataModule, ASEForDiffusionDataModuleParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    NOISY_ATOM_TYPES, NOISY_RELATIVE_COORDINATES, NUMBER_OF_ATOMS,
    PADDED_ATOM_TYPE)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters


class TestASEForDiffusionDataModule:

    @pytest.fixture()
    def elements(self):
        return ["Si"]

    @pytest.fixture()
    def natoms_per_structure(self):
        return [3, 5, 4]

    @pytest.fixture()
    def trajectory_path(self, tmp_path, natoms_per_structure):
        traj_path = tmp_path / "test.traj"
        rng = np.random.default_rng(42)
        with ase.io.Trajectory(str(traj_path), 'w') as traj:
            for n in natoms_per_structure:
                positions = rng.random((n, 3)) * 5.43
                cell = np.diag([5.43, 5.43, 5.43])
                atoms = ase.Atoms(symbols=['Si'] * n, positions=positions, cell=cell, pbc=True)
                forces = np.zeros((n, 3))
                calc = SinglePointCalculator(atoms, energy=-1.0, forces=forces)
                atoms.calc = calc
                traj.write(atoms)
        return str(traj_path)

    @pytest.fixture()
    def data_module(self, trajectory_path, natoms_per_structure, elements, tmp_path):
        hyper_params = ASEForDiffusionDataModuleParameters(
            data_source="test",
            elements=elements,
            batch_size=8,
            num_workers=0,
            max_atom=max(natoms_per_structure),
            noise_parameters=NoiseParameters(total_time_steps=10),
            use_fixed_lattice_parameters=True,
        )
        dm = ASEForDiffusionDataModule(
            processed_dataset_dir=str(tmp_path / "processed"),
            hyper_params=hyper_params,
            train_trajectory_list=[trajectory_path],
            validation_trajectory_list=[trajectory_path],
            working_cache_dir=str(tmp_path / "cache"),
        )
        dm.setup()
        return dm

    def test_padding(self, data_module, natoms_per_structure):
        dataset = data_module.train_dataset[:]
        noisy_atom_types = dataset[NOISY_ATOM_TYPES]        # [n_structures, max_atom]
        noisy_coords = dataset[NOISY_RELATIVE_COORDINATES]  # [n_structures, max_atom, 3]
        natoms = dataset[NUMBER_OF_ATOMS]                   # [n_structures]

        max_atom = max(natoms_per_structure)

        for i in range(len(natoms)):
            n = int(natoms[i])
            if n < max_atom:
                assert (noisy_atom_types[i, n:] == PADDED_ATOM_TYPE).all(), (
                    f"Sample {i} with {n} atoms: padded atom types at positions [{n}:] should be {PADDED_ATOM_TYPE}"
                )
                assert torch.isnan(noisy_coords[i, n:, :]).all(), (
                    f"Sample {i} with {n} atoms: padded coordinates at positions [{n}:] should be NaN"
                )
