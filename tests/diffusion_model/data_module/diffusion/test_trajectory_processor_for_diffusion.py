from pathlib import Path

import ase
import ase.io
import numpy as np
import pytest
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.data_module.diffusion.ase_for_diffusion_data_module import (
    ASEForDiffusionDataModule, ASEForDiffusionDataModuleParameters)
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.data_module.diffusion.trajectory_processor_for_diffusion import \
    TrajectoryProcessorForDiffusion
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.loss.loss_parameters import (
    AtomTypeLossParameters, MSELossParameters)
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.models.axl_diffusion_lightning_model import (
    AXLDiffusionLightningModel, AXLDiffusionParameters)
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.models.optimizer import \
    OptimizerParameters
from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, CARTESIAN_FORCES, NOISY_AXL_COMPOSITION)
from diffusion_for_multi_scale_molecular_dynamics.score_network.mlp_score_network import \
    MLPScoreNetworkParameters


class CountingCalculator(Calculator):
    """A live calculator that counts how many times it is asked to calculate."""

    implemented_properties = ["energy", "forces"]

    def __init__(self):
        super().__init__()
        self.number_of_calculations = 0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.number_of_calculations += 1
        self.results = dict(energy=0.0, forces=np.zeros((len(self.atoms), 3)))


class TestTrajectoryProcessorForDiffusion:

    @pytest.fixture()
    def natom(self):
        return 3

    @pytest.fixture()
    def atoms(self, natom):
        rng = np.random.default_rng(42)
        positions = rng.random((natom, 3)) * 5.43
        cell = np.diag([5.43, 5.43, 5.43])
        return ase.Atoms(symbols=["Si"] * natom, positions=positions, cell=cell, pbc=True)

    @pytest.fixture()
    def processor(self, tmp_path):
        # Empty trajectory lists: the constructor does no parsing on its own, letting the tests below call
        # parse_trajectory directly on a single, purpose-built trajectory file.
        return TrajectoryProcessorForDiffusion(
            train_trajectory_list=[], validation_trajectory_list=[], processed_data_dir=str(tmp_path)
        )

    def test_parse_trajectory_with_energy_only_calculator(self, processor, atoms, tmp_path):
        """A calculator that only implements energy should yield potential_energy but no forces."""
        atoms.calc = SinglePointCalculator(atoms, energy=-1.0)

        traj_path = tmp_path / "energy_only.traj"
        ase.io.write(str(traj_path), atoms)

        df = processor.parse_trajectory(str(traj_path))

        assert "potential_energy" in df.columns
        assert df["potential_energy"].iloc[0] == pytest.approx(-1.0)
        assert CARTESIAN_FORCES not in df.columns

    def test_parse_trajectory_without_calculator(self, processor, atoms, tmp_path):
        """A trajectory with no calculator at all should parse without raising, dropping both fields."""
        assert atoms.calc is None

        traj_path = tmp_path / "no_calculator.traj"
        ase.io.write(str(traj_path), atoms)

        df = processor.parse_trajectory(str(traj_path))

        assert "potential_energy" not in df.columns
        assert CARTESIAN_FORCES not in df.columns
        # The rest of the row should still be populated normally.
        assert len(df) == 1
        assert df["natom"].iloc[0] == len(atoms)

    def test_parse_trajectory_with_energy_and_forces(self, processor, atoms, tmp_path):
        """A SinglePointCalculator with energy and forces should yield both potential_energy and forces."""
        forces = np.arange(3 * len(atoms), dtype=float).reshape(-1, 3)
        atoms.calc = SinglePointCalculator(atoms, energy=-1.0, forces=forces)

        traj_path = tmp_path / "energy_and_forces.traj"
        ase.io.write(str(traj_path), atoms)

        df = processor.parse_trajectory(str(traj_path))

        assert df["potential_energy"].iloc[0] == pytest.approx(-1.0)
        np.testing.assert_allclose(df[CARTESIAN_FORCES].iloc[0], forces.reshape(-1))

    def test_parse_trajectory_never_calls_a_live_calculator(self, processor, atoms, tmp_path, mocker):
        """Energies and forces are only read from a SinglePointCalculator: a live calculator is never called."""
        traj_path = tmp_path / "live_calculator.traj"
        ase.io.write(str(traj_path), atoms)
        calculator = CountingCalculator()
        atoms.calc = calculator
        mocker.patch.object(ase.io, "read", return_value=[atoms])

        df = processor.parse_trajectory(str(traj_path))

        assert calculator.number_of_calculations == 0
        assert "potential_energy" not in df.columns
        assert CARTESIAN_FORCES not in df.columns


class TestTrainingWithMockedFit:
    """Mocking the training procedure with different DataModules.

    Parameters currently give 3 DataModules: energy and forces, energy only, no calculator.

    train_diffusion.py::run/train breaks down as:
        1. load_data_module -- parse data, build the DataModule.
        2. load_diffusion_model -- build the AXLDiffusionLightningModel.
        3. callbacks/loggers.
        4. build the Trainer.
        5. trainer.fit(model, datamodule):
            5.1 datamodule.setup("fit") -- parse trajectory, build train/valid datasets.
            5.2 build train_dataloader()/val_dataloader().
            5.3 sanity-check validation.
            5.4 main loop, per epoch:
                5.4.1 per training batch: (a) forward through axl_network (b) loss (c) backward (d) optimizer.step
                5.4.2 LR scheduler step
                5.4.3 validation loop
                5.4.4 callbacks (checkpointing, early stopping)
            5.5 trainer.fit returns; model holds updated weights, checkpoints on disk.

    Only 5.4.1(a)/(c) -- the network's own math -- is mocked, tied to one real nn.Parameter so
    backward/optimizer.step() still do genuine work. Everything else runs for real.
    """

    @pytest.fixture(scope="class", autouse=True)
    def set_random_seed(self):
        torch.manual_seed(123)

    @pytest.fixture()
    def natoms_per_structure(self):
        return [4, 4, 4, 4]

    @pytest.fixture()
    def elements(self):
        return ["Si"]

    @pytest.fixture(params=["no_calculator", "energy_only", "energy_and_forces"])
    def calculator_mode(self, request):
        return request.param

    @pytest.fixture()
    def trajectory_path(self, calculator_mode, tmp_path, natoms_per_structure):
        traj_path = tmp_path / "training_data.traj"
        rng = np.random.default_rng(0)
        with ase.io.Trajectory(str(traj_path), "w") as traj:
            for n in natoms_per_structure:
                positions = rng.random((n, 3)) * 5.43
                cell = np.diag([5.43, 5.43, 5.43])
                atoms = ase.Atoms(symbols=["Si"] * n, positions=positions, cell=cell, pbc=True)
                if calculator_mode == "energy_only":
                    atoms.calc = SinglePointCalculator(atoms, energy=-1.0)
                elif calculator_mode == "energy_and_forces":
                    atoms.calc = SinglePointCalculator(atoms, energy=-1.0, forces=np.zeros((n, 3)))
                traj.write(atoms)
        return str(traj_path)

    @pytest.fixture()
    def data_module(self, trajectory_path, natoms_per_structure, elements, tmp_path):
        hyper_params = ASEForDiffusionDataModuleParameters(
            data_source="test",
            elements=elements,
            batch_size=2,
            num_workers=0,
            max_atom=max(natoms_per_structure),
            noise_parameters=NoiseParameters(total_time_steps=10),
            use_fixed_lattice_parameters=True,
        )
        return ASEForDiffusionDataModule(
            processed_dataset_dir=str(tmp_path / "processed"),
            hyper_params=hyper_params,
            train_trajectory_list=[trajectory_path],
            validation_trajectory_list=[trajectory_path],
            working_cache_dir=str(tmp_path / "cache"),
        )

    @pytest.fixture()
    def lightning_model(self, natoms_per_structure, elements):
        score_network_parameters = MLPScoreNetworkParameters(
            number_of_atoms=max(natoms_per_structure),
            spatial_dimension=3,
            num_atom_types=len(elements),
            n_hidden_dimensions=1,
            hidden_dimensions_size=4,
            noise_embedding_dimensions_size=4,
            relative_coordinates_embedding_dimensions_size=4,
            time_embedding_dimensions_size=4,
            atom_type_embedding_dimensions_size=4,
            lattice_parameters_embedding_dimensions_size=4,
        )
        loss_parameters = AXL(
            A=AtomTypeLossParameters(lambda_weight=0.0),
            X=MSELossParameters(lambda_weight=1.0),
            L=MSELossParameters(lambda_weight=0.0),
        )
        hyper_params = AXLDiffusionParameters(
            score_network_parameters=score_network_parameters,
            loss_parameters=loss_parameters,
            optimizer_parameters=OptimizerParameters(name="adamw", learning_rate=1e-3, weight_decay=0.0),
        )
        return AXLDiffusionLightningModel(hyper_params)

    @pytest.fixture()
    def mock_score_network_forward(self, mocker, lightning_model, elements):
        """Replace the score network's forward with a cheap fake tied to one real parameter.

        loss.backward() and optimizer.step() still act on a genuine nn.Parameter of axl_network, so training
        and checkpointing do real work; only the network's own (expensive) computation is skipped.
        """
        num_classes = len(elements) + 1
        weight = next(lightning_model.axl_network.parameters())

        def fake_forward(batch, conditional=None):
            composition = batch[NOISY_AXL_COMPOSITION]
            batch_size, natom, spatial_dim = composition.X.shape
            lattice_dim = composition.L.shape[-1]
            fake_x = weight.mean().expand(batch_size, natom, spatial_dim).clone()
            fake_a = torch.zeros(batch_size, natom, num_classes, device=composition.X.device)
            fake_l = torch.zeros(batch_size, lattice_dim, device=composition.X.device)
            return AXL(X=fake_x, A=fake_a, L=fake_l)

        mocker.patch.object(lightning_model.axl_network, "forward", side_effect=fake_forward)
        return weight

    def test_training_runs(
        self, lightning_model, data_module, mock_score_network_forward, calculator_mode, accelerator, tmp_path
    ):
        weight_before = mock_score_network_forward.clone()
        checkpoint_callback = ModelCheckpoint(dirpath=str(tmp_path / "checkpoints"), save_last=True)
        trainer = Trainer(
            max_epochs=2, accelerator=accelerator, default_root_dir=str(tmp_path), logger=False,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(lightning_model, datamodule=data_module)

        # The mocked-but-real parameter should have actually been updated by backward + optimizer.step().
        assert not torch.equal(weight_before, mock_score_network_forward)
        # 5.4.4: the checkpoint callback should have actually saved a checkpoint.
        assert checkpoint_callback.last_model_path != ""
        assert Path(checkpoint_callback.last_model_path).exists()
