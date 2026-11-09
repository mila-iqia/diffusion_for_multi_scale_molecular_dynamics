import pickle

import ase
import ase.io
import numpy as np
import pytest
import torch
from ase.calculators.singlepoint import SinglePointCalculator
from torch.utils.data import DataLoader

from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.data_module.diffusion.ase_for_diffusion_data_module import (  # noqa
    ASEForDiffusionDataModule, ASEForDiffusionDataModuleParameters)
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    NOISY_ATOM_TYPES, NOISY_RELATIVE_COORDINATES, NUMBER_OF_ATOMS,
    PADDED_ATOM_TYPE)


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


class TestComposedTransformMultiprocessing:
    """Regression test: the composed transform must survive real DataLoader multiprocessing.

    multiprocessing_context="spawn" is forced explicitly rather than relying on the OS default, so this catches
    an unpicklable transform on any platform: Linux's default ("fork") copies process memory directly and never
    pickles anything, so it would silently miss the exact bug that only broke on Mac/Windows (default "spawn").
    """

    @pytest.fixture()
    def elements(self):
        return ["Si"]

    @pytest.fixture()
    def natoms_per_structure(self):
        return [4] * 12

    @pytest.fixture()
    def trajectory_path(self, tmp_path, natoms_per_structure):
        traj_path = tmp_path / "test.traj"
        rng = np.random.default_rng(7)
        with ase.io.Trajectory(str(traj_path), "w") as traj:
            for n in natoms_per_structure:
                positions = rng.random((n, 3)) * 5.43
                cell = np.diag([5.43, 5.43, 5.43])
                traj.write(ase.Atoms(symbols=["Si"] * n, positions=positions, cell=cell, pbc=True))
        return str(traj_path)

    @pytest.fixture()
    def data_module(self, trajectory_path, natoms_per_structure, elements, tmp_path):
        hyper_params = ASEForDiffusionDataModuleParameters(
            data_source="test",
            elements=elements,
            batch_size=1,
            num_workers=2,
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

    def test_pickling_and_worker_rng_independence(self, data_module):
        # 1. The transform must be picklable on its own: a fast, precise check before touching multiprocessing.
        composed_transform = data_module.create_composed_transform()
        pickle.loads(pickle.dumps(composed_transform))

        # 2. Force "spawn" so the check is platform-independent (see class docstring).
        num_workers = 2
        loader = DataLoader(
            data_module.train_dataset, batch_size=1, num_workers=num_workers, multiprocessing_context="spawn",
        )
        # total_time_steps=10 means only 10 possible noise levels (Gaussian probability distributions) to pick
        # from, but the atomic displacement drawn from a given noise level is continuous, so displacements
        # should never overlap.
        displacement_values = [tuple(batch[NOISY_RELATIVE_COORDINATES].flatten().tolist()) for batch in loader]

        # Indices are dispatched to workers round-robin (index i -> worker i % num_workers), so grouping this
        # way shows a shared-RNG regression at a glance.
        values_by_worker = {w: displacement_values[w::num_workers] for w in range(num_workers)}

        assert len(displacement_values) == len(set(displacement_values)), (
            "Duplicate noised displacements across DataLoader workers:\n"
            + "\n".join(f"  Worker {w}: {values}" for w, values in values_by_worker.items())
        )
