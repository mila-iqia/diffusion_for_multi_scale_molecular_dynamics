"""Tests for the filesystem-backed TrainingDatabase (no MLIP/LAMMPS).

Restarts are notoriously fragile, so the resume/reset logic below is covered exhaustively to make sure
every crash-and-resume combination behaves exactly as expected.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from diffusion_for_multi_scale_molecular_dynamics.io.training_database import (
    Stage, TrainingDatabase)
from diffusion_for_multi_scale_molecular_dynamics.io.utils import \
    write_atoms_trajectory


def _labelled_atoms(energy: float = -1.0, with_forces: bool = True) -> Atoms:
    """A labelled configuration; omit forces to simulate unlabelled data."""
    atoms = Atoms("Si2", positions=[[0.0, 0.0, 0.0], [1.1, 1.1, 1.1]], cell=[5.0, 5.0, 5.0], pbc=True)
    results = dict(energy=energy)
    if with_forces:
        results["forces"] = np.zeros((2, 3))
    atoms.calc = SinglePointCalculator(atoms, **results)
    return atoms


def _uncertain_atoms() -> Atoms:
    atoms = Atoms("Si2", positions=[[0.0, 0.0, 0.0], [1.2, 1.2, 1.2]], cell=[5.0, 5.0, 5.0], pbc=True)
    atoms.info["uncertainty"] = np.array([0.7, 0.9])
    return atoms


def _generated_atoms() -> Atoms:
    atoms = Atoms("Si2", positions=[[0.0, 0.0, 0.0], [1.2, 1.2, 1.2]], cell=[5.0, 5.0, 5.0], pbc=True)
    atoms.info["active_indices"] = np.array([0])
    atoms.info["constrained_atom_indices"] = np.array([1])
    return atoms


def _mark_model_committed(database: TrainingDatabase, epoch: int) -> None:
    (database.model_directory(epoch) / "checkpoint").write_text("model")


@pytest.fixture
def database(tmp_path):
    return TrainingDatabase(tmp_path / "database")


class TestPathsAndDiscovery:
    def test_epoch_is_zero_when_empty(self, database):
        """A fresh database reports epoch 0 (no epochs yet)."""
        assert database.epoch == 0

    def test_epoch_returns_highest_existing(self, database):
        """The epoch property is the highest existing epoch folder."""
        database.epoch_directory(1)
        database.epoch_directory(2)
        assert database.epoch == 2

    def test_stage_directories_are_created_on_demand(self, database):
        """Each stage accessor creates its folder on the first call (the fixture created only the root)."""
        epoch_root = database._root / "epoch_1"
        dynamic_path = epoch_root / "dynamic"
        oracle_path = epoch_root / "oracle"
        model_path = epoch_root / "model"

        assert not dynamic_path.exists()
        assert database.dynamic_directory(1) == dynamic_path and dynamic_path.is_dir()

        assert not oracle_path.exists()
        assert database.oracle_directory(1) == oracle_path and oracle_path.is_dir()

        assert not model_path.exists()
        assert database.model_directory(1) == model_path and model_path.is_dir()

    def test_labelled_atoms_ignore_provided_configurations_and_other_trajectories(self, database):
        """Training set is training_configurations.traj + epoch oracles; provided_configurations/other .traj are not."""
        database.append_training_configurations([_labelled_atoms(energy=-1.0)])
        database.write_provided_configurations([_labelled_atoms(energy=-9.0)])              # seed, not trained on
        write_atoms_trajectory([_labelled_atoms(energy=-8.0)], database._root / "other.traj")  # ignored
        database.write_oracle(1, [_labelled_atoms(energy=-2.0)])

        energies = sorted(atoms.get_potential_energy() for atoms in database.labelled_atoms)
        assert energies == [-2.0, -1.0]


class TestStaging:
    def test_write_and_read_dynamic(self, database):
        """The uncertain configuration round-trips through dynamic.traj with its uncertainty."""
        database.write_dynamic(1, _uncertain_atoms())
        assert database.is_dynamic_committed(1)
        assert np.allclose(database.read_dynamic(1).info["uncertainty"], [0.7, 0.9])

    def test_write_and_read_oracle(self, database):
        """The labelled configurations round-trip through oracle.traj, energies included."""
        database.write_oracle(1, [_labelled_atoms(energy=-2.0), _labelled_atoms(energy=-3.0)])
        assert database.is_oracle_committed(1)

        read_back = database.read_oracle(1)
        assert [atoms.get_potential_energy() for atoms in read_back] == [-2.0, -3.0]

    def test_model_committed_only_when_populated(self, database):
        """An empty model directory does not count as committed; a populated one does."""
        database.model_directory(1)
        assert not database.is_model_committed(1)
        _mark_model_committed(database, 1)
        assert database.is_model_committed(1)


class TestTrainingSet:
    def test_labelled_atoms_combine_training_configurations_and_oracle(self, database):
        """The training set is training_configurations.traj plus every oracle epoch; dynamic configs excluded."""
        database.append_training_configurations([_labelled_atoms(energy=-1.0)])

        # epoch 1: fully complete round.
        database.write_dynamic(1, _uncertain_atoms())
        database.write_oracle(1, [_labelled_atoms(energy=-2.0)])
        _mark_model_committed(database, 1)
        # epoch 2: driver + oracle done, not yet trained.
        database.write_dynamic(2, _uncertain_atoms())
        database.write_oracle(2, [_labelled_atoms(energy=-3.0), _labelled_atoms(energy=-4.0)])

        # The two dynamic (uncertain, unlabelled) configs are provenance and must not appear here.
        energies = sorted(atoms.get_potential_energy() for atoms in database.labelled_atoms)
        assert energies == [-4.0, -3.0, -2.0, -1.0]

    def test_check_passes_for_labelled_data(self, database):
        """A fully labelled database passes the energy/forces check."""
        database.append_training_configurations([_labelled_atoms()])
        database.write_oracle(1, [_labelled_atoms()])
        database.check_labelled_atoms_have_energy_and_forces()

    def test_check_raises_on_missing_forces(self, database):
        """A frame without forces in its calculator results fails the check."""
        database.append_training_configurations([_labelled_atoms(with_forces=False)])
        with pytest.raises(ValueError, match="missing an energy or forces"):
            database.check_labelled_atoms_have_energy_and_forces()


class TestAutoResume:
    def test_empty_database_starts_at_precomputation(self, database):
        """With nothing trained yet, auto resume points at epoch 0 (precomputation)."""
        assert database.resume_point("auto") == (0, Stage.DRIVER)

    def test_committed_precomputation_starts_first_round(self, database):
        """Once precomputation (epoch 0) is committed, auto resume starts the driver of round 1."""
        (database.precomputation_model_directory() / "model").write_text("model")
        assert database.resume_point("auto") == (1, Stage.DRIVER)

    def test_only_dynamic_resumes_at_generate(self, database):
        """A committed dynamic but no generate resumes at the generate of that epoch."""
        database.write_dynamic(1, _uncertain_atoms())
        assert database.resume_point("auto") == (1, Stage.GENERATE)

    def test_dynamic_and_generate_resumes_at_oracle(self, database):
        """Committed dynamic + generate but no oracle resumes at the oracle of that epoch."""
        database.write_dynamic(1, _uncertain_atoms())
        database.write_generate(1, [_generated_atoms()])
        assert database.resume_point("auto") == (1, Stage.ORACLE)

    def test_dynamic_and_oracle_resumes_at_train(self, database):
        """Committed dynamic + oracle but no model resumes at the train of that epoch."""
        database.write_dynamic(1, _uncertain_atoms())
        database.write_oracle(1, [_labelled_atoms()])
        assert database.resume_point("auto") == (1, Stage.TRAIN)

    def test_complete_epoch_starts_next_driver(self, database):
        """A fully committed epoch moves on to the driver of the next epoch."""
        database.write_dynamic(1, _uncertain_atoms())
        database.write_oracle(1, [_labelled_atoms()])
        _mark_model_committed(database, 1)
        assert database.resume_point("auto") == (2, Stage.DRIVER)

    def test_crashed_driver_redoes_driver(self, database):
        """An epoch folder that exists but has no dynamic artifact resumes at that epoch's driver."""
        database.dynamic_directory(1)  # folder exists, driver never committed
        assert database.resume_point("auto") == (1, Stage.DRIVER)


class TestForcedResume:
    def test_driver_on_complete_epoch_targets_next(self, database):
        """Forcing driver after a complete epoch targets a fresh epoch."""
        database.write_dynamic(1, _uncertain_atoms())
        database.write_oracle(1, [_labelled_atoms()])
        _mark_model_committed(database, 1)
        assert database.resume_point("driver") == (2, Stage.DRIVER)

    def test_driver_on_incomplete_epoch_reruns_and_clears_partial_files(self, database):
        """Forcing driver on an incomplete epoch reruns it and the reset wipes its whole working folder."""
        partial_log = database.dynamic_directory(1) / "incomplete.log"
        partial_log.write_text("half-written driver output")
        database.write_dynamic(1, _uncertain_atoms())

        epoch, stage = database.resume_point("driver")
        assert (epoch, stage) == (1, Stage.DRIVER)

        database.reset_epoch_to_stage(epoch, stage)
        assert not database.is_dynamic_committed(1)
        assert not partial_log.exists()  # the whole dynamic/ working folder is cleared, not just dynamic.traj

    def test_generate_requires_dynamic(self, database):
        """Forcing generate without a committed dynamic is an error."""
        database.epoch_directory(1)
        with pytest.raises(ValueError, match="dynamic.traj"):
            database.resume_point("generate")

    def test_oracle_requires_generate(self, database):
        """Forcing oracle without a committed generate is an error."""
        database.write_dynamic(1, _uncertain_atoms())
        with pytest.raises(ValueError, match="generate.traj"):
            database.resume_point("oracle")

    def test_train_requires_oracle(self, database):
        """Forcing train without a committed oracle is an error."""
        database.write_dynamic(1, _uncertain_atoms())
        with pytest.raises(ValueError, match="oracle.traj"):
            database.resume_point("train")

    def test_train_resumes_when_oracle_present(self, database):
        """Forcing train with a committed oracle resumes at that epoch's train."""
        database.write_oracle(1, [_labelled_atoms()])
        assert database.resume_point("train") == (1, Stage.TRAIN)


class TestResetAndRollback:
    def test_reset_to_driver_clears_all_stage_artifacts(self, database):
        """Resetting to driver clears dynamic, generate, oracle and model of the epoch."""
        database.write_dynamic(6, _uncertain_atoms())
        database.write_generate(6, [_generated_atoms()])
        database.write_oracle(6, [_labelled_atoms()])
        _mark_model_committed(database, 6)

        database.reset_epoch_to_stage(6, Stage.DRIVER)

        assert not database.is_dynamic_committed(6)
        assert not database.is_generate_committed(6)
        assert not database.is_oracle_committed(6)
        assert not database.is_model_committed(6)

    def test_reset_to_generate_keeps_dynamic(self, database):
        """Resetting to generate wipes the model, oracle and generate artifacts, keeping the uncertain config."""
        database.write_dynamic(6, _uncertain_atoms())
        database.write_generate(6, [_generated_atoms()])
        database.write_oracle(6, [_labelled_atoms()])
        _mark_model_committed(database, 6)

        database.reset_epoch_to_stage(6, Stage.GENERATE)

        assert database.is_dynamic_committed(6)
        assert not database.is_generate_committed(6)
        assert not database.is_oracle_committed(6)
        assert not (database._root / "epoch_6" / "model").exists()

    def test_reset_to_oracle_keeps_dynamic_and_generate(self, database):
        """Resetting to oracle wipes the model and oracle folders while keeping the generated samples."""
        database.write_dynamic(6, _uncertain_atoms())
        database.write_generate(6, [_generated_atoms()])
        database.write_oracle(6, [_labelled_atoms()])
        _mark_model_committed(database, 6)

        database.reset_epoch_to_stage(6, Stage.ORACLE)

        assert database.is_dynamic_committed(6)
        assert database.is_generate_committed(6)
        assert not database.is_oracle_committed(6)
        assert not (database._root / "epoch_6" / "oracle").exists()
        assert not (database._root / "epoch_6" / "model").exists()

    def test_reset_to_train_keeps_dynamic_and_oracle(self, database):
        """Resetting to train wipes the model directory entirely, keeping the labelled data and config."""
        database.write_dynamic(6, _uncertain_atoms())
        database.write_generate(6, [_generated_atoms()])
        database.write_oracle(6, [_labelled_atoms()])
        _mark_model_committed(database, 6)

        database.reset_epoch_to_stage(6, Stage.TRAIN)

        assert database.is_dynamic_committed(6)
        assert database.is_generate_committed(6)
        assert database.is_oracle_committed(6)
        # the model directory is removed wholesale so no stale checkpoint can corrupt the retrain.
        assert not (database._root / "epoch_6" / "model").exists()

    def test_rollback_removes_latest_epoch(self, database):
        """Rolling back deletes the highest epoch folder and returns its number."""
        database.epoch_directory(1)
        database.epoch_directory(2)

        removed = database.rollback_last_epoch()

        assert removed == 2
        assert database.epoch == 1

    def test_rollback_on_empty_returns_zero(self, database):
        """Rolling back an empty database is a no-op returning 0."""
        assert database.rollback_last_epoch() == 0
