"""Tests for the MD dynamic driver (unit-level rendering + an end-to-end LAMMPS run)."""

from unittest.mock import MagicMock

import pytest

from diffusion_for_multi_scale_molecular_dynamics.dynamic_driver.md_driver.md_driver import \
    MdDriver
from diffusion_for_multi_scale_molecular_dynamics.io.dynamic_driver.calculation_state import \
    CalculationState
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_runner import \
    SubprocessLammpsRunner

_TEMPERATURE = 300.0
_TIMESTEP = 0.001
_NUMBER_OF_STEPS = 5


@pytest.fixture
def md_driver(initial_configuration):
    """An MD driver with a mock runner (unit-level: no LAMMPS is actually launched)."""
    return MdDriver(
        lammps_runner=MagicMock(),
        initial_configuration=initial_configuration,
        temperature=_TEMPERATURE,
        timestep=_TIMESTEP,
        number_of_steps=_NUMBER_OF_STEPS,
    )


class TestUnit:
    def test_prepare_reference_files_is_noop(self, md_driver, tmp_path):
        """MD needs no extra reference file, so preparing one leaves the working directory empty."""
        working_directory = tmp_path / "work"
        working_directory.mkdir()
        md_driver._prepare_reference_files(working_directory)
        assert list(working_directory.iterdir()) == []

    def test_dynamics_block_substitutes_parameters(self, md_driver):
        """The NVT block carries the temperature, timestep, step count and a damping of 100x the timestep."""
        block = md_driver._dynamics_block()
        assert "$" not in block
        assert str(_TEMPERATURE) in block
        assert f"run {_NUMBER_OF_STEPS}" in block
        assert str(100.0 * _TIMESTEP) in block

    def test_no_dump_is_success(self, md_driver, tmp_path):
        """A run that never wrote an uncertain dump means the uncertainty stayed below threshold: SUCCESS."""
        assert md_driver._get_calculation_state(tmp_path) == CalculationState.SUCCESS

    def test_empty_dump_is_success(self, md_driver, tmp_path):
        """An uncertain dump file with no content still means nothing was flagged: SUCCESS."""
        (tmp_path / "uncertain.dump").write_text("")
        assert md_driver._get_calculation_state(tmp_path) == CalculationState.SUCCESS

    def test_nonempty_dump_is_interruption(self, md_driver, tmp_path):
        """A non-empty uncertain dump means the halt fired on an uncertain structure: INTERRUPTION."""
        (tmp_path / "uncertain.dump").write_text("ITEM: TIMESTEP\n0\n")
        assert md_driver._get_calculation_state(tmp_path) == CalculationState.INTERRUPTION


@pytest.mark.requires_lammps_bin
class TestEndToEnd:
    def _make_driver(self, lammps_executable_path, initial_configuration):
        runner = SubprocessLammpsRunner(lammps_executable_path=lammps_executable_path, mpi_processors=1)
        return MdDriver(
            lammps_runner=runner,
            initial_configuration=initial_configuration,
            temperature=_TEMPERATURE,
            timestep=_TIMESTEP,
            number_of_steps=_NUMBER_OF_STEPS,
        )

    def test_high_threshold_runs_to_completion(
        self, lammps_executable_path, initial_configuration, stub_mlip, tmp_path
    ):
        """A huge threshold is never exceeded, so a real MD run completes without a halt: SUCCESS."""
        driver = self._make_driver(lammps_executable_path, initial_configuration)
        state = driver.run(stub_mlip, tmp_path / "work", uncertainty_threshold=1.0e9)
        assert state == CalculationState.SUCCESS

    def test_low_threshold_halts_immediately(self, lammps_executable_path, initial_configuration, stub_mlip, tmp_path):
        """A threshold below any per-atom energy trips the halt at once, producing the uncertain dump: INTERRUPTION."""
        driver = self._make_driver(lammps_executable_path, initial_configuration)
        state = driver.run(stub_mlip, tmp_path / "work", uncertainty_threshold=-1.0e9)
        assert state == CalculationState.INTERRUPTION
