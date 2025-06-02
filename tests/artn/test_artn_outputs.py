import pytest

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.artn.artn_outputs import (
    INTERRUPTION_MESSAGE, SUCCESS_MESSAGE,
    get_calculation_state_from_artn_output)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.artn.calculation_state import \
    CalculationState
from tests.fake_data_utils import generate_random_string


@pytest.fixture(params=['success', 'interruption'])
def job_status(request):
    return request.param


@pytest.fixture()
def artn_output(job_status):

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

    return '\n'.join(lines)


def test_get_calculation_state_from_artn_output(artn_output, job_status):
    state = get_calculation_state_from_artn_output(artn_output)

    if job_status == 'success':
        assert state == CalculationState.SUCCESS

    elif job_status == 'interruption':
        assert state == CalculationState.INTERRUPTION
