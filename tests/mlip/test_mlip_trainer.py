import tempfile
from pathlib import Path

import ase.io
import numpy as np
import pytest

from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.potential import \
    LammpsPotential
from diffusion_for_multi_scale_molecular_dynamics.io.training_database import \
    TrainingDatabase
from diffusion_for_multi_scale_molecular_dynamics.oracle.base_single_point_calculator import \
    SinglePointCalculation  # noqa
from diffusion_for_multi_scale_molecular_dynamics.utils.structure_conversion import \
    to_pymatgen_structure


def stage_labelled_structure(training_database, trainer, labelled_structure, active_environment_indices):
    """Mirror the loop: persist the label to the database (stage 2) and fold it into the model (stage 3)."""
    training_database.write_oracle(1, [labelled_structure.to_atoms(active_environment_indices)])
    trainer.add_labelled_structure(labelled_structure, active_environment_indices)


# A fictitious energy label (eV); its actual value is irrelevant, we only check the fit reproduces it.
MOCK_ENERGY = -26.43783

SI8_STRUCTURE_FILE = Path(__file__).parent.parent / "reference_files" / "structure" / "Si8.in"

# Parameters read back from a level-6 MTP fitted on the fixed Si8 structure. The radial basis size, alpha scalar
# moments and species count are fixed by the level.
MTP_LEVEL = 6
MTP_MAX_DIST = 3.5
MTP_EXPECTED_RADIAL_BASIS_SIZE = 8
MTP_EXPECTED_ALPHA_SCALAR_MOMENTS = 5
MTP_EXPECTED_SPECIES_COUNT = 1


def predict_mtp_energy(trainer, atoms):
    """Predict a configuration's energy with the fitted MTP through 'mlp calculate_efs'."""
    import glob
    import os
    import subprocess

    from maml.utils import check_structures_forces_stresses, pool_from

    from diffusion_for_multi_scale_molecular_dynamics.io.mlip import \
        write_mtp_cfg

    structure = to_pymatgen_structure(atoms)
    work_directory = Path(tempfile.mkdtemp())
    potential = trainer.write_checkpoint(work_directory)
    checked_structures, checked_forces, _ = check_structures_forces_stresses(
        [structure], [np.zeros((len(structure), 3))], None
    )
    training_pool = pool_from(checked_structures, [0.0], checked_forces)

    original_directory = Path.cwd()
    os.chdir(work_directory)
    try:
        write_mtp_cfg(training_pool, trainer.configuration.elements, Path("input.cfg"))
        subprocess.run(
            ["mlp", "calculate_efs", str(potential.mtp_file_path), "input.cfg", "--output_filename=output.cfg"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        # calculate_efs writes a rank-suffixed file (e.g. output.cfg.0); read the energy from its header.
        lines = Path(sorted(glob.glob("output.cfg*"))[0]).read_text().splitlines()
        energy_index = next(index for index, line in enumerate(lines) if line.strip() == "Energy")
        return float(lines[energy_index + 1].strip())
    finally:
        os.chdir(original_directory)


def verify_fitted_model(trainer_type, trainer, labelled_structure):
    """Assert the fitted model reproduces its energy label within tolerance (per-backend predictor)."""
    if trainer_type == "flare":
        from diffusion_for_multi_scale_molecular_dynamics.oracle.flare_single_point_calculator import \
            FlareSinglePointCalculator
        predicted_energy = FlareSinglePointCalculator(trainer.sgp_model).calculate(labelled_structure.atoms).energy
        np.testing.assert_allclose(predicted_energy, labelled_structure.energy, atol=1e-1)
    elif trainer_type == "mtp":
        configuration = trainer.configuration
        assert configuration.radial_basis_size == MTP_EXPECTED_RADIAL_BASIS_SIZE
        assert configuration.alpha_scalar_moments == MTP_EXPECTED_ALPHA_SCALAR_MOMENTS
        assert configuration.species_count == MTP_EXPECTED_SPECIES_COUNT
        assert configuration.number_of_adjustable_parameters == 14
        predicted_energy = predict_mtp_energy(trainer, labelled_structure.atoms)
        np.testing.assert_allclose(predicted_energy, labelled_structure.energy, atol=1e-1)
    else:
        raise ValueError(f"Unknown trainer type '{trainer_type}'.")


def checkpoint_file_name(trainer_type):
    """The file write_checkpoint produces for each trainer type."""
    return {"flare": "checkpoint.json", "mtp": "potential.almtp"}[trainer_type]


def reload_trainer(trainer_type, trainer, checkpoint_path, elements):
    """Reconstruct a trainer from a checkpoint (MTP needs the configuration, which the .almtp does not record)."""
    if trainer_type == "flare":
        return type(trainer).load_checkpoint(checkpoint_path)
    elif trainer_type == "mtp":
        from diffusion_for_multi_scale_molecular_dynamics.mlip.mtp.mtp_configuration import \
            MtpConfiguration
        configuration = MtpConfiguration(elements=elements, level=MTP_LEVEL, max_dist=MTP_MAX_DIST)
        return type(trainer).load_checkpoint(checkpoint_path, mtp_configuration=configuration)
    raise ValueError(f"Unknown trainer type '{trainer_type}'.")


class TestMLIPTrainer:

    @pytest.fixture(
        params=[
            pytest.param("flare", marks=pytest.mark.requires_flare),
            pytest.param("mtp", marks=pytest.mark.requires_mlp),
        ]
    )
    def trainer_type(self, request):
        return request.param

    @pytest.fixture
    def elements(self, trainer_type, list_element_symbols):
        # The level-6 MTP template is single-species; flare handles the shared multi-species set.
        return ["Si"] if trainer_type == "mtp" else list_element_symbols

    @pytest.fixture
    def training_structure(self, trainer_type, structure):
        # MTP needs a fixed single-species structure so the fitted parameters are deterministic.
        if trainer_type == "mtp":
            return ase.io.read(str(SI8_STRUCTURE_FILE), format="lammps-data", atom_style="atomic")
        return structure.to_ase_atoms()

    @pytest.fixture
    def training_database(self, tmp_path):
        return TrainingDatabase(tmp_path / "database")

    @pytest.fixture
    def trainer(self, trainer_type, elements, training_database):
        if trainer_type == "flare":
            from diffusion_for_multi_scale_molecular_dynamics.mlip.flare.flare_configuration import \
                FlareConfiguration
            from diffusion_for_multi_scale_molecular_dynamics.mlip.flare.flare_trainer import \
                FlareTrainer
            return FlareTrainer(FlareConfiguration(cutoff=4.0,
                                                   elements=elements,
                                                   n_radial=4,
                                                   lmax=2,
                                                   variance_type='local',
                                                   initial_sigma_e=1e-8),
                                training_database=training_database)
        if trainer_type == "mtp":
            from diffusion_for_multi_scale_molecular_dynamics.mlip.mtp.mtp_configuration import \
                MtpConfiguration
            from diffusion_for_multi_scale_molecular_dynamics.mlip.mtp.mtp_trainer import \
                MtpTrainer
            configuration = MtpConfiguration(
                elements=elements,
                level=MTP_LEVEL,
                max_dist=MTP_MAX_DIST,
                training_params=dict(max_iter=100, init_params="same", scale_by_force=0.0, bfgs_conv_tol=1e-3),
            )
            return MtpTrainer(configuration, training_database=training_database)
        raise ValueError(f"Unknown trainer type '{trainer_type}'.")

    @pytest.fixture
    def labelled_structure(self, training_structure):
        # We drop the forces (train on energy only) so the fitted model can reproduce its label.
        number_of_atoms = len(training_structure)
        return SinglePointCalculation(calculation_type='dummy_test',
                                      atoms=training_structure,
                                      forces=np.zeros((number_of_atoms, 3)),
                                      energy=MOCK_ENERGY)

    @pytest.fixture
    def active_environment_indices(self, training_structure):
        return list(np.arange(len(training_structure)))

    @pytest.fixture
    def trained_trainer(self, trainer, training_database, labelled_structure, active_environment_indices):
        stage_labelled_structure(training_database, trainer, labelled_structure, active_environment_indices)
        trainer.fit()
        return trainer

    def test_labelled_calculations_reflect_database(
        self, trainer, training_database, labelled_structure, active_environment_indices
    ):
        """The trainer reads its training set from the database rather than remembering what it was fed."""
        assert len(trainer.labelled_calculations) == 0
        stage_labelled_structure(training_database, trainer, labelled_structure, active_environment_indices)
        assert len(trainer.labelled_calculations) == 1
        np.testing.assert_allclose(trainer.labelled_calculations[0].energy, labelled_structure.energy)

    def test_fit(self, trainer_type, trainer, training_database, labelled_structure, active_environment_indices):
        """After fitting on a single structure, the fitted model matches its label."""
        stage_labelled_structure(training_database, trainer, labelled_structure, active_environment_indices)
        trainer.fit()
        verify_fitted_model(trainer_type, trainer, labelled_structure)

    def test_write_checkpoint(self, trainer_type, trained_trainer, elements, tmp_path):
        """write_checkpoint deploys a LAMMPS potential and round-trips to an identical checkpoint."""
        potential = trained_trainer.write_checkpoint(tmp_path / "first")
        assert isinstance(potential, LammpsPotential)

        checkpoint_path = tmp_path / "first" / checkpoint_file_name(trainer_type)
        assert checkpoint_path.is_file()

        # The checkpoint is written before any model-mutating step, so reloading and re-writing it is identical.
        reloaded_trainer = reload_trainer(trainer_type, trained_trainer, checkpoint_path, elements)
        reloaded_trainer.write_checkpoint(tmp_path / "second")
        second_checkpoint_path = tmp_path / "second" / checkpoint_file_name(trainer_type)
        assert second_checkpoint_path.read_bytes() == checkpoint_path.read_bytes()
