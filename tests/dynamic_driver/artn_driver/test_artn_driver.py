"""Tests for the ARTn dynamic driver (unit-level generation/parsing + an end-to-end LAMMPS+ARTn run)."""

import shutil
from unittest.mock import MagicMock

import pytest

from diffusion_for_multi_scale_molecular_dynamics.dynamic_driver.artn_driver.artn_driver import \
    ArtnDriver
from diffusion_for_multi_scale_molecular_dynamics.io.dynamic_driver.artn import (
    INTERRUPTION_MESSAGE, read_artn_xyz)
from diffusion_for_multi_scale_molecular_dynamics.io.dynamic_driver.artn_input_configuration import \
    ArtnInputConfiguration
from diffusion_for_multi_scale_molecular_dynamics.io.dynamic_driver.calculation_state import \
    CalculationState
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_runner import \
    SubprocessLammpsRunner
from diffusion_for_multi_scale_molecular_dynamics.utils.structure_utils import \
    configurations_are_equivalent

SPECORDER = ["Si", "Ge"]

PUSH_IDS = 1
PUSH_ADD_CONST = [1.0, -1.0, -1.0, 20]


@pytest.fixture
def fake_plugin_path(tmp_path):
    """A dummy file standing in for the compiled ARTn plugin (never loaded at unit level)."""
    plugin_path = tmp_path / "libartn-lmp.so"
    plugin_path.write_text("")
    return plugin_path


@pytest.fixture
def artn_driver(initial_configuration, fake_plugin_path):
    """An ARTn driver with a mock runner (unit-level: no LAMMPS is actually launched)."""
    return ArtnDriver(
        lammps_runner=MagicMock(),
        initial_configuration=initial_configuration,
        artn_input_configuration=ArtnInputConfiguration(push_ids=PUSH_IDS, push_add_const=PUSH_ADD_CONST),
        artn_library_plugin_path=fake_plugin_path,
    )


class TestUnit:
    def test_prepare_reference_files_writes_artn_in(self, artn_driver, tmp_path):
        """Preparing the run generates 'artn.in' (with the push variables) in the working directory."""
        working_directory = tmp_path / "work"
        working_directory.mkdir()
        artn_driver._prepare_reference_files(working_directory)

        artn_input = (working_directory / "artn.in").read_text()
        assert "&ARTN_PARAMETERS" in artn_input
        assert f"push_ids = {PUSH_IDS}" in artn_input
        assert f"push_add_const(:,{PUSH_IDS}) = 1.0, -1.0, -1.0, 20" in artn_input

    def test_dynamics_block_loads_the_plugin(self, artn_driver, fake_plugin_path):
        """The ARTn LAMMPS tail loads the plugin from the configured path."""
        assert str(fake_plugin_path) in artn_driver._dynamics_block()

    def test_missing_output_is_error(self, artn_driver, tmp_path):
        """An ARTn run that produced no 'artn.out' cannot be interpreted: ERROR."""
        assert artn_driver._get_calculation_state(tmp_path) == CalculationState.ERROR

    def test_success_message_is_success(self, artn_driver, tmp_path):
        """An 'artn.out' carrying the clean-exit message maps to SUCCESS."""
        (tmp_path / "artn.out").write_text("!> CLEANING ARTn | Fail: 0\n")
        assert artn_driver._get_calculation_state(tmp_path) == CalculationState.SUCCESS

    def test_interruption_message_is_interruption(self, artn_driver, tmp_path):
        """An 'artn.out' carrying the early-stop message maps to INTERRUPTION."""
        (tmp_path / "artn.out").write_text(f"{INTERRUPTION_MESSAGE}\n")
        assert artn_driver._get_calculation_state(tmp_path) == CalculationState.INTERRUPTION


@pytest.fixture
def hopping_driver(reference_files_directory, fake_plugin_path):
    """An ARTn driver whose current configuration is the committed Si3Ge reference (restart_from_new_min on)."""
    initial_configuration = read_artn_xyz(reference_files_directory / "artn" / "Si3Ge_conf.xyz", SPECORDER)
    return ArtnDriver(
        lammps_runner=MagicMock(),
        initial_configuration=initial_configuration,
        artn_input_configuration=ArtnInputConfiguration(push_ids=PUSH_IDS, push_add_const=PUSH_ADD_CONST),
        artn_library_plugin_path=fake_plugin_path,
        restart_from_new_min=True,
    )


class TestRestartFromNewMinimum:
    @pytest.mark.parametrize("min1_name, min2_name", [
        ("Si3Ge_smalldiff.xyz", "Si3Ge_bigdiff.xyz"),  # min1 is the basin we came from -> hop to min2
        ("Si3Ge_bigdiff.xyz", "Si3Ge_smalldiff.xyz"),  # min1 is the new minimum -> hop to min1
    ])
    def test_hops_to_the_minimum_that_differs_from_the_current(
        self, hopping_driver, reference_files_directory, tmp_path, min1_name, min2_name
    ):
        """The hop restarts from the minimum that differs from the current configuration, not the basin."""
        artn_directory = reference_files_directory / "artn"
        shutil.copy(artn_directory / min1_name, tmp_path / "min1.xyz")
        shutil.copy(artn_directory / min2_name, tmp_path / "min2.xyz")
        new_minimum = read_artn_xyz(artn_directory / "Si3Ge_bigdiff.xyz", SPECORDER)
        initial_configuration = read_artn_xyz(artn_directory / "Si3Ge_conf.xyz", SPECORDER)

        hopping_driver._hop_to_new_minimum(tmp_path)

        assert configurations_are_equivalent(hopping_driver._current_configuration, new_minimum)
        assert not configurations_are_equivalent(hopping_driver._current_configuration, initial_configuration)
        # the deployed data file the next search reads must have been rewritten with the new minimum
        assert (tmp_path / "initial_configuration.dat").stat().st_size > 0


@pytest.mark.requires_lammps_bin
@pytest.mark.requires_artn
class TestEndToEnd:
    def test_runs_and_returns_a_state(self, lammps_executable_path, initial_configuration, stub_mlip, tmp_path):
        """A real LAMMPS+ARTn run drives to completion and yields a parseable calculation state.

        Requires a LAMMPS executable built with the ARTn plugin; the plugin location is resolved by the
        driver from the ARTN_PLUGIN_PATH environment variable (ArtnDriver raises if it is not set).
        """
        runner = SubprocessLammpsRunner(lammps_executable_path=lammps_executable_path, mpi_processors=1)
        driver = ArtnDriver(
            lammps_runner=runner, initial_configuration=initial_configuration,
            artn_input_configuration=ArtnInputConfiguration(push_ids=PUSH_IDS, push_add_const=PUSH_ADD_CONST))
        state = driver.run(stub_mlip, tmp_path / "work", uncertainty_threshold=1.0e9)
        assert isinstance(state, CalculationState)
