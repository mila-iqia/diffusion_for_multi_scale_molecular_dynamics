import numpy as np
import pytest

from diffusion_for_multi_scale_molecular_dynamics.io.dynamic_driver.artn import (
    INTERRUPTION_MESSAGE, SUCCESS_MESSAGE, build_artn_lammps_tail,
    get_calculation_state_from_artn_output, get_saddle_energy,
    write_artn_input_file)
from diffusion_for_multi_scale_molecular_dynamics.io.dynamic_driver.artn_input_configuration import \
    ArtnInputConfiguration
from diffusion_for_multi_scale_molecular_dynamics.io.dynamic_driver.calculation_state import \
    CalculationState
from tests.fake_data_utils import generate_random_string


@pytest.fixture(params=['success', 'interruption'])
def job_status(request):
    return request.param


@pytest.fixture()
def saddle_energy():
    return np.random.rand()


@pytest.fixture()
def saddle_energy_line(saddle_energy):
    line = ("|> DEBRIEF(SADDLE) | dE = " + f"{saddle_energy:8.6f}"
            + " eV | F_{tot,para,perp} = 0.88852E-2  0.54230E-2  0.51877E-2  eV/Ang | Eig")
    return line


@pytest.fixture()
def artn_output(job_status, saddle_energy_line):

    lines = []
    for _ in range(10):
        lines.append(generate_random_string(36))

    if job_status == 'interruption':
        lines.append("Some random text " + INTERRUPTION_MESSAGE + " some more stuff")

    for _ in range(5):
        lines.append(generate_random_string(36))

    if job_status == 'success':
        # We have to fiddle the string because "|" looks like a pipe for regex.
        success_message = SUCCESS_MESSAGE.replace("\\", "")

        lines.append("some stuff " + success_message + " some more stuff")
        lines.append(saddle_energy_line)

    return '\n'.join(lines)


def test_get_calculation_state_from_artn_output(artn_output, job_status):
    state = get_calculation_state_from_artn_output(artn_output)

    if job_status == 'success':
        assert state == CalculationState.SUCCESS

    elif job_status == 'interruption':
        assert state == CalculationState.INTERRUPTION


@pytest.mark.parametrize("job_status", ["success"])
def test_get_saddle_energy(artn_output, saddle_energy):
    computed_saddle_energy = get_saddle_energy(artn_output)
    np.testing.assert_almost_equal(computed_saddle_energy, saddle_energy, decimal=5)


def test_write_artn_input_file(tmp_path):
    """The namelist is written with the task-specific push variables and the typed defaults."""
    configuration = ArtnInputConfiguration(push_ids=441, push_add_const=[1.0, -1.0, -1.0, 20])
    path = write_artn_input_file(tmp_path / "artn.in", configuration)
    content = path.read_text()

    assert content.startswith("&ARTN_PARAMETERS")
    assert content.strip().endswith("/")
    assert "push_ids = 441" in content
    assert "push_add_const(:,441) = 1.0, -1.0, -1.0, 20" in content
    assert "engine_units = 'lammps/metal'" in content  # string is quoted
    assert "lpush_final = .true." in content            # bool is Fortran-formatted


def test_write_artn_input_file_overrides_defaults(tmp_path):
    """A non-default field on the configuration overrides the typed default in the namelist."""
    configuration = ArtnInputConfiguration(push_ids=1, push_add_const=[1, 0, 0, 10], forc_thr=0.05)
    path = write_artn_input_file(tmp_path / "artn.in", configuration)
    assert "forc_thr = 0.05" in path.read_text()


def test_build_artn_lammps_tail():
    """The ARTn LAMMPS tail loads the plugin, adds the ARTn fix, and runs a minimization."""
    tail = build_artn_lammps_tail("/plugins/libartn-lmp.so")
    assert "plugin load /plugins/libartn-lmp.so" in tail
    assert "fix artn_fix_id all artn" in tail
    assert "minimize" in tail
