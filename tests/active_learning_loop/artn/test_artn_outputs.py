import numpy as np
import pytest

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.artn.artn_outputs import (
    INTERRUPTION_MESSAGE, SUCCESS_MESSAGE,
    get_calculation_state_from_artn_output, get_saddle_energy)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.artn.calculation_state import \
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
