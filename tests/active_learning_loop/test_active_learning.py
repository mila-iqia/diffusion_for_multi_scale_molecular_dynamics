"""Tests for the ActiveLearning loop orchestration and crash-resume, with stubbed stages (no LAMMPS/MLIP)."""

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import ase.io
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.active_learning import \
    ActiveLearning
from diffusion_for_multi_scale_molecular_dynamics.io.dynamic_driver.calculation_state import \
    CalculationState
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.stillinger_weber import \
    StillingerWeberPotential
from diffusion_for_multi_scale_molecular_dynamics.io.training_database import \
    TrainingDatabase
from diffusion_for_multi_scale_molecular_dynamics.mlip.base_mlip import \
    BaseMLIP
from diffusion_for_multi_scale_molecular_dynamics.oracle.base_single_point_calculator import (
    SinglePointCalculation, get_active_environment_indices)
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_runner import \
    SubprocessLammpsRunner
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_single_point_calculator import \
    LammpsSinglePointCalculator
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.atom_selector.top_k_atom_selector import (
    TopKAtomSelector, TopKAtomSelectorParameters)
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.no_op_sample_maker import (
    NoOpSampleMaker, NoOpSampleMakerArguments)


def _labelled_atoms(energy: float) -> Atoms:
    atoms = Atoms("Si2", positions=[[0.0, 0.0, 0.0], [1.1, 1.1, 1.1]], cell=[5.0, 5.0, 5.0], pbc=True)
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=np.zeros((2, 3)))
    return atoms


def _uncertain_atoms() -> Atoms:
    atoms = Atoms("Si2", positions=[[0.0, 0.0, 0.0], [1.2, 1.2, 1.2]], cell=[5.0, 5.0, 5.0], pbc=True)
    atoms.info["uncertainty"] = np.array([0.7, 0.9])
    return atoms


def _stub_mlip():
    mlip = MagicMock()
    mlip.training_metrics.return_value = dict(n_training_conf=0, rmse_energy=None, rmse_forces=None)
    mlip.minimum_number_of_training_environments.return_value = {}  # no precomputation for the stubbed loop
    return mlip


def _stub_sample_maker():
    sample_maker = MagicMock()
    sample_maker.arguments.element_list = ["Si"]
    return sample_maker


class RecordingActiveLearning(ActiveLearning):
    """An ActiveLearning whose three stages are stubbed to record calls and write the database artifacts.

    This exercises the real run_campaign/_run_round/resume logic without any LAMMPS or MLIP dependency.
    """

    def __init__(self, success_at_epoch: int, mlip=None):
        super().__init__(
            oracle_single_point_calculator=MagicMock(),
            sample_maker=_stub_sample_maker(),
            dynamic_driver=MagicMock(),
            mlip=mlip if mlip is not None else _stub_mlip(),
        )
        self.calls = []
        self._success_at_epoch = success_at_epoch

    def run_dynamic_driver(self, epoch):
        self.calls.append(("driver", epoch))
        if epoch >= self._success_at_epoch:
            return None  # SUCCESS: no uncertain structure found.
        return _uncertain_atoms()

    def generate_samples(self, uncertain_configuration, epoch):
        self.calls.append(("generate", epoch))
        return [_uncertain_atoms()]

    def oracle_evaluation(self, generated_samples, epoch):
        self.calls.append(("oracle", epoch))
        return [_labelled_atoms(energy=float(epoch))]

    def _retrain(self, epoch, training_configurations):
        self.calls.append(("train", epoch))
        (self._training_database.model_directory(epoch) / "model").write_text("model")


def test_fresh_run_completes_and_commits_each_epoch(tmp_path):
    """A fresh run drives each epoch through all three stages until the driver reports SUCCESS."""
    active_learning = RecordingActiveLearning(success_at_epoch=3)
    active_learning.run_campaign(uncertainty_threshold=0.1, working_directory=tmp_path, provided_configurations=[])

    assert active_learning.calls == [("driver", 1), ("generate", 1), ("oracle", 1), ("train", 1),
                                     ("driver", 2), ("generate", 2), ("oracle", 2), ("train", 2),
                                     ("driver", 3)]
    database = active_learning._training_database
    for epoch in (1, 2):
        assert database.is_dynamic_committed(epoch)
        assert database.is_generate_committed(epoch)
        assert database.is_oracle_committed(epoch)
        assert database.is_model_committed(epoch)


def test_training_set_accumulates_across_epochs(tmp_path):
    """The database training set grows with every completed epoch (the reload-forgetting bug is gone)."""
    active_learning = RecordingActiveLearning(success_at_epoch=3)
    active_learning.run_campaign(uncertainty_threshold=0.1, working_directory=tmp_path, provided_configurations=[])

    energies = sorted(atoms.get_potential_energy() for atoms in active_learning._training_database.labelled_atoms)
    assert energies == [1.0, 2.0]


def test_resume_at_generate_skips_the_driver(tmp_path):
    """A crash after the driver resumes at generate: the driver is not re-run for that epoch."""
    database = TrainingDatabase(tmp_path)
    database.write_dynamic(1, _uncertain_atoms())  # epoch 1 crashed after the driver committed.

    active_learning = RecordingActiveLearning(success_at_epoch=2)
    active_learning.run_campaign(uncertainty_threshold=0.1, working_directory=tmp_path, provided_configurations=[])

    assert active_learning.calls == [("generate", 1), ("oracle", 1), ("train", 1), ("driver", 2)]


def test_resume_at_oracle_skips_driver_and_generate(tmp_path):
    """A crash after generate resumes at the oracle: neither the driver nor generate re-run for that epoch."""
    database = TrainingDatabase(tmp_path)
    database.write_dynamic(1, _uncertain_atoms())
    database.write_generate(1, [_uncertain_atoms()])  # epoch 1 crashed after generate committed.

    active_learning = RecordingActiveLearning(success_at_epoch=2)
    active_learning.run_campaign(uncertainty_threshold=0.1, working_directory=tmp_path, provided_configurations=[])

    assert active_learning.calls == [("oracle", 1), ("train", 1), ("driver", 2)]


def test_resume_at_train_skips_driver_and_oracle(tmp_path):
    """A crash after labelling resumes at training: neither the driver nor the oracle re-run for that epoch."""
    database = TrainingDatabase(tmp_path)
    database.write_dynamic(1, _uncertain_atoms())
    database.write_oracle(1, [_labelled_atoms(energy=1.0)])

    active_learning = RecordingActiveLearning(success_at_epoch=2)
    active_learning.run_campaign(uncertainty_threshold=0.1, working_directory=tmp_path, provided_configurations=[])

    assert active_learning.calls == [("train", 1), ("driver", 2)]


def test_forced_train_restart_retrains_a_complete_epoch(tmp_path):
    """Forcing restart_from_stage='train' on a complete epoch resets its model and retrains it."""
    database = TrainingDatabase(tmp_path)
    database.write_dynamic(1, _uncertain_atoms())
    database.write_oracle(1, [_labelled_atoms(energy=1.0)])
    (database.model_directory(1) / "model").write_text("stale")  # epoch 1 was fully complete.

    active_learning = RecordingActiveLearning(success_at_epoch=2)
    active_learning.run_campaign(uncertainty_threshold=0.1, working_directory=tmp_path, provided_configurations=[],
                                 restart_from_stage="train")

    assert active_learning.calls == [("train", 1), ("driver", 2)]
    assert active_learning._training_database.is_model_committed(1)


def test_restart_reloads_the_latest_committed_model(tmp_path):
    """Restarting a campaign with a completed epoch reloads that epoch's model from disk (no re-fit)."""
    database = TrainingDatabase(tmp_path)
    database.write_dynamic(1, _uncertain_atoms())
    database.write_oracle(1, [_labelled_atoms(energy=1.0)])
    (database.model_directory(1) / "model").write_text("model")  # epoch 1 fully committed

    mlip = _stub_mlip()
    active_learning = RecordingActiveLearning(success_at_epoch=2, mlip=mlip)
    active_learning.run_campaign(uncertainty_threshold=0.1, working_directory=tmp_path, provided_configurations=[])

    # Resumes at round 2's driver, having reloaded epoch 1's model straight from disk (no precomputation).
    assert active_learning.calls == [("driver", 2)]
    mlip.load.assert_called_once_with(database.model_directory(1))


class _PrecomputationStubMLIP(BaseMLIP):
    """A real BaseMLIP (so the greedy augmentation runs for real) with a fixed per-element environment floor
    and a stubbed train/deploy (no LAMMPS). The seed configurations are single-species (Si)."""

    def __init__(self, minimum_number_of_environments: int):
        super().__init__(trainer=MagicMock(), lammps_runner=MagicMock())
        self._minimum_number_of_environments = minimum_number_of_environments
        self.trained_model_directories = []
        self._attached_training_database = None

    def attach_training_database(self, training_database):
        self._attached_training_database = training_database  # the MagicMock trainer would not keep it

    @property
    def training_database(self):
        return self._attached_training_database

    def minimum_number_of_training_environments(self) -> dict:
        return {"Si": self._minimum_number_of_environments}

    def train(self, output_directory):
        self.trained_model_directories.append(Path(output_directory))
        (Path(output_directory) / "model").write_text("model")

    def prepare_mlip_first_round(self, output_directory):
        pass

    def training_metrics(self, reference_atoms=None):
        return dict(n_training_conf=0, rmse_energy=None, rmse_forces=None)

    def write_state_yaml(self, output_path):
        pass

    @classmethod
    def load_checkpoint(cls, checkpoint_path):
        pass


def test_run_campaign_precomputation_doubles_a_single_configuration(tmp_path):
    """run_campaign with the rounds skipped precomputes: one provided config is augmented to reach the minimum.

    The seed has 2 atoms and the minimum is 4 environments, so precomputation must produce 2 labelled
    configurations (double the single seed) before any round runs.
    """
    provided_configurations = [_labelled_atoms(energy=0.0)]  # a single seed config

    oracle = MagicMock()
    oracle.calculate_many.side_effect = lambda list_atoms: [
        SinglePointCalculation(calculation_type="stub", atoms=atoms,
                               forces=np.zeros((len(atoms), 3)), energy=-1.0)
        for atoms in list_atoms
    ]
    mlip = _PrecomputationStubMLIP(minimum_number_of_environments=4)  # 2-atom seed -> 2 structures (doubled)

    active_learning = ActiveLearning(oracle_single_point_calculator=oracle,
                                     sample_maker=_stub_sample_maker(), dynamic_driver=MagicMock(), mlip=mlip)
    active_learning.run_campaign(
        uncertainty_threshold=0.1, working_directory=tmp_path, provided_configurations=provided_configurations,
        maximum_number_of_rounds=0,  # skip the campaign; only precomputation runs
    )

    database = active_learning._training_database
    assert len(database.labelled_atoms) == 2  # one seed config doubled by augmentation
    assert database.number_of_labelled_environments() == 4
    assert mlip.trained_model_directories == [database.precomputation_model_directory()]
    assert database.is_precomputation_model_committed()


def test_run_campaign_uses_provided_database_directly_when_it_covers_the_minimum(tmp_path):
    """A provided database that already covers the minimum is used as the training set, without augmentation."""
    provided_configurations = [_labelled_atoms(energy=float(index)) for index in range(3)]  # 3 x 2 atoms = 6 envs

    oracle = MagicMock()  # must NOT be called: nothing to augment/label
    mlip = _PrecomputationStubMLIP(minimum_number_of_environments=4)  # 6 >= 4 -> use the provided configs directly

    active_learning = ActiveLearning(oracle_single_point_calculator=oracle,
                                     sample_maker=_stub_sample_maker(), dynamic_driver=MagicMock(), mlip=mlip)
    active_learning.run_campaign(
        uncertainty_threshold=0.1, working_directory=tmp_path, provided_configurations=provided_configurations,
        maximum_number_of_rounds=0,
    )

    database = active_learning._training_database
    assert len(database.labelled_atoms) == 3  # the provided configurations, adopted as-is
    oracle.calculate_many.assert_not_called()
    assert mlip.trained_model_directories == [database.precomputation_model_directory()]


def test_precomputation_raises_on_unlabelled_provided_configurations(tmp_path):
    """Precomputation refuses provided configurations that carry no energy/forces (they must be labelled)."""
    unlabelled_configuration = Atoms("Si2", positions=[[0.0, 0.0, 0.0], [1.1, 1.1, 1.1]],
                                     cell=[5.0, 5.0, 5.0], pbc=True)  # no calculator: unlabelled
    mlip = _PrecomputationStubMLIP(minimum_number_of_environments=4)

    active_learning = ActiveLearning(oracle_single_point_calculator=MagicMock(),
                                     sample_maker=_stub_sample_maker(), dynamic_driver=MagicMock(), mlip=mlip)
    with pytest.raises(ValueError, match="must be labelled"):
        active_learning.run_campaign(
            uncertainty_threshold=0.1, working_directory=tmp_path,
            provided_configurations=[unlabelled_configuration], maximum_number_of_rounds=0,
        )


@pytest.mark.requires_lammps_bin
@pytest.mark.requires_pair_style("sw")
class TestFullRound:
    """One real round with the actual stage internals: real Stillinger-Weber oracle + real sample maker.

    Only the driver, the LAMMPS-dump reader (covered by tests/io/lammps/test_outputs.py) and the MLIP
    training are stubbed; everything the refactor introduced (uncertainty commit, the SPC<->Atoms
    conversions, sample making, labelling, folding, the symlink) runs for real.
    """

    REFERENCE_FILES = Path(__file__).parent.parent / "reference_files"

    def test_one_real_round_wires_every_stage(self, tmp_path, monkeypatch):
        """A full round drives the uncertain config through the real oracle stage and into the model."""
        uncertain_structure = ase.io.read(
            str(self.REFERENCE_FILES / "structure" / "Si8.in"), format="lammps-data", atom_style="atomic"
        )
        uncertainty_per_atom = np.linspace(0.1, 1.0, len(uncertain_structure))

        # Real collaborators: a Stillinger-Weber oracle and a lightweight no-op sample maker.
        lammps_runner = SubprocessLammpsRunner(
            lammps_executable_path=Path(shutil.which("lmp") or shutil.which("lammps")), mpi_processors=1
        )
        oracle = LammpsSinglePointCalculator(
            StillingerWeberPotential(sw_coefficients_file_path=self.REFERENCE_FILES / "aSi.sw"),
            lammps_runner,
        )
        sample_maker = NoOpSampleMaker(
            sample_maker_arguments=NoOpSampleMakerArguments(element_list=["Si"]),
            atom_selector=TopKAtomSelector(TopKAtomSelectorParameters(top_k_environment=2)),
        )

        # Stubbed collaborators: the dynamic driver (INTERRUPTION then SUCCESS) and the MLIP's training.
        dynamic_driver = MagicMock()
        dynamic_driver.run.side_effect = [CalculationState.INTERRUPTION, CalculationState.SUCCESS]
        mlip = MagicMock()
        mlip.training_metrics.return_value = dict(n_training_conf=0, rmse_energy=None, rmse_forces=None)
        mlip.minimum_number_of_training_environments.return_value = {}  # no precomputation in this round test
        mlip.train.side_effect = lambda model_directory: (Path(model_directory) / "model").write_text("model")

        active_learning = ActiveLearning(oracle_single_point_calculator=oracle,
                                         sample_maker=sample_maker, dynamic_driver=dynamic_driver, mlip=mlip)
        # Bypass only the LAMMPS-dump reader; feed the uncertain configuration directly.
        monkeypatch.setattr(
            active_learning, "_get_uncertain_structure_and_uncertainties",
            lambda working_directory, uncertainty_field: (uncertain_structure, uncertainty_per_atom, 42),
        )

        active_learning.run_campaign(uncertainty_threshold=0.5, working_directory=tmp_path, provided_configurations=[])
        database = active_learning._training_database

        # Stage 1: the uncertain configuration and its per-atom uncertainty were committed.
        assert database.is_dynamic_committed(1)
        np.testing.assert_allclose(database.read_dynamic(1).info["uncertainty"], uncertainty_per_atom)

        # Stage 2: Stillinger-Weber labelled the sample with a real energy/forces and kept the active indices.
        assert database.is_oracle_committed(1)
        labelled_atoms = database.read_oracle(1)
        assert len(labelled_atoms) == 1
        oracle_energy = labelled_atoms[0].get_potential_energy()
        assert np.isfinite(oracle_energy)
        assert labelled_atoms[0].get_forces().shape == (len(uncertain_structure), 3)
        assert "active_environment_indices" in labelled_atoms[0].info
        assert (database._root / "epoch_1" / "oracle" / "oracle_single_point_calculations.pkl").is_file()

        # Stage 3: the labelled data was folded into the model via the real SinglePointCalculation<->Atoms path.
        assert database.is_model_committed(1)
        mlip.add_labelled_structure.assert_called_once()
        single_point_calculation, active_environment_indices = mlip.add_labelled_structure.call_args.args
        np.testing.assert_allclose(single_point_calculation.energy, oracle_energy)
        assert active_environment_indices == get_active_environment_indices(labelled_atoms[0])

        # The driver ran twice (INTERRUPTION then SUCCESS) and latest_mlip points at epoch 1's model.
        assert dynamic_driver.run.call_count == 2
        assert (tmp_path / "latest_mlip").resolve() == database.model_directory(1).resolve()
