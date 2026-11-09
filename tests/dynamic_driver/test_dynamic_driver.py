"""Tests for the base DynamicDriver logic (no LAMMPS execution, no real MLIP)."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from diffusion_for_multi_scale_molecular_dynamics.dynamic_driver.base_dynamic_driver import \
    DynamicDriver
from diffusion_for_multi_scale_molecular_dynamics.io.dynamic_driver.calculation_state import \
    CalculationState

_DYNAMICS_BLOCK = "run 0  # stub dynamics"


class _StubDriver(DynamicDriver):
    """A minimal concrete driver that records its hooks so the base logic can be tested in isolation."""

    def __init__(self, *args, calculation_state=CalculationState.SUCCESS, **kwargs):
        super().__init__(*args, **kwargs)
        self.calculation_state = calculation_state
        self.prepared_directories = []

    def _prepare_reference_files(self, working_directory: Path) -> None:
        self.prepared_directories.append(working_directory)

    def _dynamics_block(self) -> str:
        return _DYNAMICS_BLOCK

    def _get_calculation_state(self, working_directory: Path) -> CalculationState:
        return self.calculation_state


@pytest.fixture
def mock_runner():
    """A LAMMPS runner whose run_lammps is a no-op by default."""
    runner = MagicMock()
    runner.run_lammps = MagicMock(return_value=None)
    return runner


@pytest.fixture
def driver(mock_runner, initial_configuration):
    return _StubDriver(lammps_runner=mock_runner, initial_configuration=initial_configuration)


class TestInitialization:
    def test_stores_the_initial_configuration(self, driver):
        """The starting ase.Atoms is stored on the driver."""
        assert len(driver.initial_configuration) == 16
        assert set(driver.initial_configuration.get_chemical_symbols()) == {"Si"}

    def test_non_orthogonal_initial_configuration_raises(self, mock_runner, initial_configuration):
        """A tilted (non-orthogonal) starting configuration is rejected at construction time."""
        sheared_configuration = initial_configuration.copy()
        cell = sheared_configuration.cell.array.copy()
        cell[1, 0] = 1.0  # introduce an xy tilt
        sheared_configuration.set_cell(cell, scale_atoms=False)
        with pytest.raises(ValueError, match="orthogonal"):
            _StubDriver(lammps_runner=mock_runner, initial_configuration=sheared_configuration)


class TestSetupWorkingDirectory:
    def test_setup_writes_configuration_and_prepares_reference_files(self, driver, tmp_path):
        """Setting up a fresh working directory writes the starting config and runs the reference-file hook."""
        working_directory = tmp_path / "work"
        driver._setup_working_directory(working_directory)

        assert (working_directory / "initial_configuration.dat").is_file()
        assert driver.prepared_directories == [working_directory]

    def test_existing_nonempty_working_directory_raises(self, driver, tmp_path):
        """Refuse to run into a non-empty working directory so existing data is never overwritten."""
        working_directory = tmp_path / "work"
        working_directory.mkdir()
        (working_directory / "leftover.txt").write_text("existing data")
        with pytest.raises(ValueError):
            driver._setup_working_directory(working_directory)

    def test_existing_empty_working_directory_is_allowed(self, driver, tmp_path):
        """An empty pre-existing directory (e.g. created by the training database) is accepted."""
        working_directory = tmp_path / "work"
        working_directory.mkdir()
        driver._setup_working_directory(working_directory)
        assert (working_directory / "initial_configuration.dat").is_file()


class TestBuildLammpsParameters:
    def test_parameters_are_assembled(self, driver, stub_mlip):
        """The template substitutions are pulled from the MLIP potential, the threshold and the dynamics block."""
        parameters = driver._build_lammps_parameters(stub_mlip, uncertainty_threshold=0.5)

        assert parameters["uncertainty_field"] == "c_pe_atom"
        assert parameters["uncertainty_threshold"] == "0.500000000000"
        assert parameters["dynamics_block"] == _DYNAMICS_BLOCK
        assert "pair_style lj/cut 6.0" in parameters["interaction_commands"]
        assert "compute pe_atom all pe/atom" in parameters["interaction_commands"]


class TestWriteLammpsInput:
    def test_rendered_script_has_no_leftover_placeholders(self, driver, stub_mlip, tmp_path):
        """The rendered LAMMPS script fully substitutes every placeholder, including the dynamics block."""
        working_directory = tmp_path / "work"
        working_directory.mkdir()
        parameters = driver._build_lammps_parameters(stub_mlip, uncertainty_threshold=0.5)
        driver._write_lammps_input(working_directory, parameters)

        script = (working_directory / "lammps.in").read_text()
        assert "$" not in script
        assert "pair_style lj/cut 6.0" in script
        assert _DYNAMICS_BLOCK in script


class TestRunOrchestration:
    def test_run_returns_error_when_lammps_fails(self, mock_runner, initial_configuration, tmp_path):
        """A failing LAMMPS run short-circuits to ERROR without ever interpreting a calculation state."""
        mock_runner.run_lammps.side_effect = RuntimeError("boom")
        driver = _StubDriver(lammps_runner=mock_runner, initial_configuration=initial_configuration,
                             calculation_state=CalculationState.SUCCESS)
        get_state = MagicMock(wraps=driver._get_calculation_state)
        driver._get_calculation_state = get_state

        state = driver.run(MagicMock(), tmp_path / "work", uncertainty_threshold=0.5)

        assert state == CalculationState.ERROR
        get_state.assert_not_called()

    def test_run_delegates_state_when_lammps_succeeds(self, mock_runner, stub_mlip, initial_configuration, tmp_path):
        """A successful run writes the input, invokes the runner and returns the subclass's calculation state."""
        driver = _StubDriver(lammps_runner=mock_runner, initial_configuration=initial_configuration,
                             calculation_state=CalculationState.INTERRUPTION)
        working_directory = tmp_path / "work"

        state = driver.run(stub_mlip, working_directory, uncertainty_threshold=0.5)

        assert state == CalculationState.INTERRUPTION
        assert (working_directory / "lammps.in").is_file()
        mock_runner.run_lammps.assert_called_once()
        _, kwargs = mock_runner.run_lammps.call_args
        assert kwargs["lammps_input_file_name"] == "lammps.in"
