from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import ase.io
import numpy as np
import pytest
import yaml
from ase import Atoms

from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.potential import \
    LammpsPotential
from diffusion_for_multi_scale_molecular_dynamics.io.training_database import \
    TrainingDatabase
from diffusion_for_multi_scale_molecular_dynamics.mlip.base_mlip import \
    BaseMLIP
from diffusion_for_multi_scale_molecular_dynamics.oracle.base_single_point_calculator import \
    SinglePointCalculation

SI8_STRUCTURE_FILE = Path(__file__).parent.parent / "reference_files" / "structure" / "Si8.in"


def make_calculation(energy, forces):
    return SinglePointCalculation(calculation_type="dummy_test",
                                  atoms=MagicMock(),
                                  forces=np.asarray(forces, dtype=float),
                                  energy=energy)


class ConcreteMLIP(BaseMLIP):
    """Minimal concrete MLIP to exercise the base class's concrete methods with a mocked trainer."""

    def train(self, output_directory):
        pass

    def write_state_yaml(self, output_path):
        pass

    @classmethod
    def load_checkpoint(cls, checkpoint_path):
        pass


class TestBaseMLIP:

    @pytest.fixture
    def trainer(self):
        return MagicMock()

    @pytest.fixture
    def mlip(self, trainer):
        return ConcreteMLIP(trainer=trainer, lammps_runner=MagicMock())

    def test_training_metrics_without_data(self, mlip, trainer):
        """With no training configuration, metrics report zero configurations and no RMSE."""
        trainer.labelled_calculations = []
        assert mlip.training_metrics() == dict(
            n_training_conf=0, n_training_atomic_environments=0, rmse_energy=None, rmse_forces=None
        )

    def test_attach_training_database_delegates_to_trainer(self, mlip, trainer):
        """Attaching a database forwards to the trainer, and the property reads it back from there."""
        database = MagicMock()
        mlip.attach_training_database(database)
        trainer.set_training_database.assert_called_once_with(database)
        assert mlip.training_database is trainer.training_database

    def test_training_metrics(self, mlip, trainer):
        """RMSE energy and forces are computed from the deployed potential's predictions against the labels."""
        trainer.labelled_calculations = [make_calculation(0.0, [[0, 0, 0]]),
                                         make_calculation(0.0, [[0, 0, 0]])]
        predictions = [make_calculation(1.0, [[1, 0, 0]]), make_calculation(3.0, [[0, 0, 0]])]

        with patch.object(mlip, "calculate", return_value=predictions) as calculate:
            metrics = mlip.training_metrics()

        calculate.assert_called_once()
        assert metrics["n_training_conf"] == 2
        assert metrics["rmse_energy"] == pytest.approx(np.sqrt((1.0 ** 2 + 3.0 ** 2) / 2))
        assert metrics["rmse_forces"] == pytest.approx(np.sqrt(1.0 ** 2 / 6))

    @pytest.mark.parametrize(
        "minimum_environments, expected_number_of_seeds",
        [(16, 1), (24, 2), (25, 3)],  # the 8-atom seed already counts; ceil((minimum - 8) / 8) copies more
    )
    def test_greedy_augmentation_single_element(self, mlip, minimum_environments, expected_number_of_seeds):
        """One 8-atom Si seed is rattled ceil((minimum - 8) / 8) times to cover a single-element floor."""
        structure = Atoms("Si8", positions=np.zeros((8, 3)), cell=5.0 * np.eye(3), pbc=True)

        seeds = mlip._greedy_augmentation_seeds([structure], {"Si": minimum_environments})

        assert len(seeds) == expected_number_of_seeds
        assert all(seed is structure for seed in seeds)

    def test_greedy_augmentation_picks_the_composition_matching_the_deficit(self, mlip):
        """The greedy rattles the configuration whose composition best matches the missing-element ratio."""
        magnesium_rich = Atoms("Mg2", positions=np.zeros((2, 3)), cell=5.0 * np.eye(3), pbc=True)
        magnesium_hydride = Atoms("MgH4", positions=np.zeros((5, 3)), cell=5.0 * np.eye(3), pbc=True)
        # have: Mg = 2 + 1 = 3, H = 0 + 4 = 4; floor {Mg: 4, H: 8} -> missing Mg 1, H 4 (ratio 1:4 == MgH4).
        seeds = mlip._greedy_augmentation_seeds([magnesium_rich, magnesium_hydride], {"Mg": 4, "H": 8})

        # One copy of the H-rich cell covers both floors at once; the pure-Mg cell is never chosen.
        assert seeds == [magnesium_hydride]

    def test_greedy_augmentation_requires_every_needed_element(self, mlip):
        """A floor for an element absent from every provided configuration is an immediate, clear error."""
        silicon = Atoms("Si2", positions=np.zeros((2, 3)), cell=5.0 * np.eye(3), pbc=True)
        with pytest.raises(ValueError, match="No provided configuration contains 'Ge'"):
            mlip._greedy_augmentation_seeds([silicon], {"Si": 2, "Ge": 4})

    def test_default_minimum_number_of_training_environments_is_empty(self, mlip):
        """With no D-optimality active set, no element requires any environments."""
        assert mlip.minimum_number_of_training_environments() == {}


def construct_derived_trainer(mlip_type, structure, list_element_symbols, training_database):
    """Build an (unfitted) trainer of the given type and the labelled structure to fold into it."""
    if mlip_type == "flare":
        from diffusion_for_multi_scale_molecular_dynamics.mlip.flare.flare_configuration import \
            FlareConfiguration
        from diffusion_for_multi_scale_molecular_dynamics.mlip.flare.flare_trainer import \
            FlareTrainer
        trainer = FlareTrainer(FlareConfiguration(cutoff=4.0,
                                                  elements=list_element_symbols,
                                                  n_radial=4,
                                                  lmax=2,
                                                  variance_type='local'),
                               training_database=training_database)
        labelled_structure = SinglePointCalculation(calculation_type="dummy_test",
                                                    atoms=structure.to_ase_atoms(),
                                                    forces=np.random.rand(len(structure), 3),
                                                    energy=-1.0)
    elif mlip_type == "mtp":
        from diffusion_for_multi_scale_molecular_dynamics.mlip.mtp.mtp_configuration import \
            MtpConfiguration
        from diffusion_for_multi_scale_molecular_dynamics.mlip.mtp.mtp_trainer import \
            MtpTrainer

        # MTP needs a fixed single-species structure (the level-6 template is single-species).
        mtp_atoms = ase.io.read(str(SI8_STRUCTURE_FILE), format="lammps-data", atom_style="atomic")
        trainer = MtpTrainer(MtpConfiguration(
            elements=["Si"],
            level=6,
            max_dist=4.0,
            training_params=dict(max_iter=100, init_params="same", scale_by_force=0.0, bfgs_conv_tol=1e-3),
        ), training_database=training_database)
        labelled_structure = SinglePointCalculation(calculation_type="dummy_test",
                                                    atoms=mtp_atoms,
                                                    forces=np.zeros((len(mtp_atoms), 3)),
                                                    energy=-26.43783)
    else:
        raise ValueError(f"Unknown MLIP type '{mlip_type}'.")
    return trainer, labelled_structure


def construct_derived_mlip(mlip_type, trainer):
    """Wrap a trainer in its MLIP (a mock LAMMPS runner: these tests never deploy to real LAMMPS)."""
    if mlip_type == "flare":
        from diffusion_for_multi_scale_molecular_dynamics.mlip.flare.flare_hyperparameter_optimizer import (
            FlareHyperparametersOptimizer, FlareOptimizerConfiguration)
        from diffusion_for_multi_scale_molecular_dynamics.mlip.flare.flare_mlip import \
            FlareMLIP
        optimizer = FlareHyperparametersOptimizer(
            FlareOptimizerConfiguration(max_optimization_iterations=3,
                                        optimize_sigma=False,
                                        optimize_sigma_e=True,
                                        optimize_sigma_f=False,
                                        optimize_sigma_s=False)
        )
        return FlareMLIP(flare_trainer=trainer, hyperparameter_optimizer=optimizer, lammps_runner=MagicMock())
    if mlip_type == "mtp":
        from diffusion_for_multi_scale_molecular_dynamics.mlip.mtp.mtp_mlip import \
            MtpMlip
        return MtpMlip(mtp_trainer=trainer, lammps_runner=MagicMock())
    raise ValueError(f"Unknown MLIP type '{mlip_type}'.")


class TestDerivedMLIP:

    @pytest.fixture(
        params=[
            pytest.param("flare", marks=pytest.mark.requires_flare),
            pytest.param("mtp", marks=pytest.mark.requires_mlp),
        ]
    )
    def mlip_type(self, request):
        return request.param

    @pytest.fixture
    def training_database(self, tmp_path_factory):
        return TrainingDatabase(tmp_path_factory.mktemp("database"))

    @pytest.fixture
    def trainer(self, mlip_type, structure, list_element_symbols, training_database):
        trainer, labelled_structure = construct_derived_trainer(
            mlip_type, structure, list_element_symbols, training_database
        )
        # Mirror the loop: persist the label to the database (stage 2) and fold it into the model (stage 3).
        active_environment_indices = list(range(len(labelled_structure.atoms)))
        training_database.write_oracle(1, [labelled_structure.to_atoms(active_environment_indices)])
        trainer.add_labelled_structure(labelled_structure, active_environment_indices)
        trainer.fit()  # a pretrained model, ready to deploy
        return trainer

    @pytest.fixture
    def mlip(self, mlip_type, trainer):
        return construct_derived_mlip(mlip_type, trainer)

    def test_prepare_mlip_first_round(self, mlip, tmp_path):
        """prepare_mlip_first_round deploys the pretrained model to a runnable potential, without training."""
        mlip.prepare_mlip_first_round(tmp_path)

        assert isinstance(mlip.lammps_potential, LammpsPotential)
        assert len(list(tmp_path.iterdir())) > 0

    def test_train(self, mlip_type, mlip, trainer, tmp_path):
        """train fits the model, deploys it and writes a checkpoint."""
        with ExitStack() as stack:
            fit = stack.enter_context(patch.object(trainer, "fit", wraps=trainer.fit))
            # The state file records training-set RMSE; stub it so writing it needs no real LAMMPS evaluation.
            stack.enter_context(patch.object(
                mlip, "training_metrics",
                return_value=dict(n_training_conf=0, n_training_atomic_environments=0,
                                  rmse_energy=None, rmse_forces=None)))
            fit_hyperparameters = None
            if mlip_type == "flare":
                fit_hyperparameters = stack.enter_context(
                    patch.object(trainer, "fit_hyperparameters", wraps=trainer.fit_hyperparameters))
            mlip.train(tmp_path)

        fit.assert_called_once()
        if fit_hyperparameters is not None:
            fit_hyperparameters.assert_called_once()
        assert isinstance(mlip.lammps_potential, LammpsPotential)
        assert mlip.model_file.is_file()

    def test_minimum_number_of_training_environments(self, mlip_type, mlip):
        """FLARE needs no active set ({}); a level-6 single-species MTP needs its descriptor-space dimension."""
        expected_minimum = {"flare": {}, "mtp": {"Si": 22}}[mlip_type]
        assert mlip.minimum_number_of_training_environments() == expected_minimum


class TestGraceMlip:
    """GraceMlip's own logic (train orchestration + state), with a mocked trainer — no gracemaker fit."""

    def test_train_deploys_and_writes_state(self, tmp_path):
        from diffusion_for_multi_scale_molecular_dynamics.mlip.grace.grace_mlip import \
            GraceMlip

        model_file_path = tmp_path / "model.yaml"
        model_file_path.write_text("model")
        active_set_file_path = tmp_path / "model.asi"
        active_set_file_path.write_text("asi")

        potential = MagicMock()
        potential.model_file_path = model_file_path
        potential.active_set_file_path = active_set_file_path

        trainer = MagicMock()
        trainer.write_checkpoint.return_value = potential
        trainer.training_database = None  # no database -> state file omits the training-set provenance
        trainer.configuration = MagicMock(elements=["Si"], cutoff=3.5, preset="FS",
                                          size="small", seed=1, target_total_updates=500)

        # Mock 'which' so __init__'s dependency check passes without the GRACE toolchain installed.
        with patch("diffusion_for_multi_scale_molecular_dynamics.mlip.grace.grace_mlip.shutil.which",
                   return_value="/usr/bin/dummy"):
            mlip = GraceMlip(grace_trainer=trainer, lammps_runner=MagicMock())
            mlip.train(tmp_path)

        trainer.fit.assert_called_once()
        assert mlip.model_file == model_file_path
        assert mlip.lammps_potential is potential

        state = yaml.safe_load((tmp_path / "state.yaml").read_text())
        assert state["model_file"] == str(model_file_path)
        assert state["lammps_potential_file"] == str(model_file_path)
        assert state["unc_file"] == str(active_set_file_path)
        assert state["hyperparameters"]["elements"] == ["Si"]


def _pretrain_and_deploy(mlip_type, structure, list_element_symbols, database, model_directory):
    """Fit a 'good' reference model once and deploy it to model_directory (the committed epoch model)."""
    trainer, labelled_structure = construct_derived_trainer(mlip_type, structure, list_element_symbols, database)
    active_environment_indices = list(range(len(labelled_structure.atoms)))
    database.write_oracle(1, [labelled_structure.to_atoms(active_environment_indices)])
    trainer.add_labelled_structure(labelled_structure, active_environment_indices)
    trainer.fit()  # the single (reference) fitting step

    mlip = construct_derived_mlip(mlip_type, trainer)
    mlip.prepare_mlip_first_round(model_directory)  # deploy the fitted model, without re-fitting
    return mlip


@pytest.mark.requires_mlp
def test_mtp_load_warm_starts_from_the_committed_potential(
    structure, list_element_symbols, tmp_path, tmp_path_factory
):
    """MtpMlip.load restores the exact committed coefficients into the trainer, so the next fit warm-starts.

    Restoring the fitted ``potential.almtp`` byte-for-byte means a restarted campaign resumes from that good
    potential (mlp ``--init-params='same'``) rather than the cold level template.
    """
    database = TrainingDatabase(tmp_path_factory.mktemp("reference_database"))
    model_directory = tmp_path / "committed_model"
    _pretrain_and_deploy("mtp", structure, list_element_symbols, database, model_directory)

    # A fresh MLIP, as a restarted process would build it: no fitted state yet.
    fresh_trainer, _ = construct_derived_trainer(
        "mtp", structure, list_element_symbols, TrainingDatabase(tmp_path_factory.mktemp("fresh_database"))
    )
    fresh_mlip = construct_derived_mlip("mtp", fresh_trainer)
    assert fresh_mlip._trainer._fitted_potential is None  # would otherwise cold-start from the template

    fresh_mlip.load(model_directory)

    assert isinstance(fresh_mlip.lammps_potential, LammpsPotential)  # the deployed potential drives the dynamics
    assert fresh_mlip._trainer._fitted_potential == (model_directory / "potential.almtp").read_bytes()


@pytest.mark.requires_flare
def test_flare_load_restores_the_sparse_gp(structure, list_element_symbols, tmp_path, tmp_path_factory):
    """FlareMLIP.load reloads the full sparse GP, so the restarted model predicts exactly as the committed one.

    FLARE folds data incrementally, so a restart that reloaded only the deployed potential would lose the
    training history; reloading the sparse GP from ``checkpoint.json`` preserves it (same prediction).
    """
    from diffusion_for_multi_scale_molecular_dynamics.oracle.flare_single_point_calculator import \
        FlareSinglePointCalculator

    database = TrainingDatabase(tmp_path_factory.mktemp("reference_database"))
    model_directory = tmp_path / "committed_model"
    reference_mlip = _pretrain_and_deploy("flare", structure, list_element_symbols, database, model_directory)
    reference_energy = FlareSinglePointCalculator(reference_mlip._trainer.sgp_model).calculate(
        structure.to_ase_atoms()
    ).energy

    # A fresh MLIP (empty sparse GP), as a restarted process would build it.
    fresh_trainer, _ = construct_derived_trainer(
        "flare", structure, list_element_symbols, TrainingDatabase(tmp_path_factory.mktemp("fresh_database"))
    )
    fresh_mlip = construct_derived_mlip("flare", fresh_trainer)

    fresh_mlip.load(model_directory)

    assert isinstance(fresh_mlip.lammps_potential, LammpsPotential)  # the deployed potential drives the dynamics
    restored_energy = FlareSinglePointCalculator(fresh_mlip._trainer.sgp_model).calculate(
        structure.to_ase_atoms()
    ).energy
    np.testing.assert_allclose(restored_energy, reference_energy, atol=1e-6)
