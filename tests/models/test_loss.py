import pytest
import torch

from crystal_diffusion.models.loss import LossCalculator, LossParameters
from crystal_diffusion.utils.tensor_utils import \
    broadcast_batch_tensor_to_all_dimensions


@pytest.fixture(scope="module", autouse=True)
def set_random_seed():
    torch.manual_seed(45233423)


@pytest.fixture()
def sigma0():
    return 0.256


@pytest.fixture()
def exponent():
    return 11.234


@pytest.fixture()
def batch_size():
    return 12


@pytest.fixture()
def spatial_dimension():
    return 3


@pytest.fixture()
def number_of_atoms():
    return 8


@pytest.fixture()
def predicted_normalized_scores(batch_size, number_of_atoms, spatial_dimension):
    return torch.rand(batch_size, number_of_atoms, spatial_dimension)


@pytest.fixture()
def target_normalized_conditional_scores(batch_size, number_of_atoms, spatial_dimension):
    return torch.rand(batch_size, number_of_atoms, spatial_dimension)


@pytest.fixture()
def sigmas(batch_size, number_of_atoms, spatial_dimension):
    batch_sigmas = torch.rand(batch_size)
    shape = (batch_size, number_of_atoms, spatial_dimension)
    sigmas = broadcast_batch_tensor_to_all_dimensions(batch_values=batch_sigmas, final_shape=shape)
    return sigmas


@pytest.fixture()
def weights(sigmas, sigma0, exponent):
    return 1.0 + torch.exp(exponent * (sigmas - sigma0))


@pytest.fixture()
def loss_parameters(algorithm, sigma0, exponent):
    return LossParameters(algorithm=algorithm, sigma0=sigma0, exponent=exponent)


@pytest.fixture()
def loss_calculator(loss_parameters):
    return LossCalculator(loss_parameters)


@pytest.fixture()
def computed_loss(loss_calculator, predicted_normalized_scores, target_normalized_conditional_scores, sigmas):
    unreduced_loss = loss_calculator.calculate_unreduced_loss(predicted_normalized_scores,
                                                              target_normalized_conditional_scores,
                                                              sigmas)
    return torch.mean(unreduced_loss)


@pytest.fixture()
def expected_loss(algorithm, weights, predicted_normalized_scores, target_normalized_conditional_scores, sigmas):
    if algorithm == 'mse':
        loss = torch.nn.functional.mse_loss(
            predicted_normalized_scores, target_normalized_conditional_scores, reduction="mean"
        )
    else:
        return torch.mean(weights * (predicted_normalized_scores - target_normalized_conditional_scores)**2)

    return loss


@pytest.mark.parametrize("algorithm", ['mse', 'weighted_mse'])
def test_mse_loss(computed_loss, expected_loss):
    torch.testing.assert_close(computed_loss, expected_loss)
