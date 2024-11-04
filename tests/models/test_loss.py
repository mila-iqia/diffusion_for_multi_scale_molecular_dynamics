from unittest.mock import patch

import einops
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.loss import \
    create_loss_calculator
from diffusion_for_multi_scale_molecular_dynamics.loss.atom_type_loss_calculator import \
    D3PMLossCalculator
from diffusion_for_multi_scale_molecular_dynamics.loss.loss_parameters import (
    LossParameters, MSELossParameters, WeightedMSELossParameters)
from src.diffusion_for_multi_scale_molecular_dynamics.utils.tensor_utils import \
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
def target_normalized_conditional_scores(
    batch_size, number_of_atoms, spatial_dimension
):
    return torch.rand(batch_size, number_of_atoms, spatial_dimension)


@pytest.fixture()
def sigmas(batch_size, number_of_atoms, spatial_dimension):
    batch_sigmas = torch.rand(batch_size)
    shape = (batch_size, number_of_atoms, spatial_dimension)
    sigmas = broadcast_batch_tensor_to_all_dimensions(
        batch_values=batch_sigmas, final_shape=shape
    )
    return sigmas


@pytest.fixture()
def weights(sigmas, sigma0, exponent):
    return 1.0 + torch.exp(exponent * (sigmas - sigma0))


@pytest.fixture(params=["mse", "weighted_mse"])
def algorithm(request):
    return request.param


@pytest.fixture()
def loss_parameters(algorithm, sigma0, exponent):
    match algorithm:
        case "mse":
            parameters = MSELossParameters()
        case "weighted_mse":
            parameters = WeightedMSELossParameters(sigma0=sigma0, exponent=exponent)
        case _:
            raise ValueError(f"Unknown loss algorithm {algorithm}")
    return parameters


@pytest.fixture()
def loss_calculator(loss_parameters):
    return create_loss_calculator(loss_parameters)


@pytest.fixture()
def computed_loss(
    loss_calculator,
    predicted_normalized_scores,
    target_normalized_conditional_scores,
    sigmas,
):
    unreduced_loss = loss_calculator.X.calculate_unreduced_loss(
        predicted_normalized_scores, target_normalized_conditional_scores, sigmas
    )
    return torch.mean(unreduced_loss)


@pytest.fixture()
def expected_loss(
    algorithm,
    weights,
    predicted_normalized_scores,
    target_normalized_conditional_scores,
    sigmas,
):
    match algorithm:
        case "mse":
            loss = torch.nn.functional.mse_loss(
                predicted_normalized_scores,
                target_normalized_conditional_scores,
                reduction="mean",
            )
        case "weighted_mse":
            loss = torch.mean(
                weights
                * (predicted_normalized_scores - target_normalized_conditional_scores)
                ** 2
            )
        case _:
            raise ValueError(f"Unknown loss algorithm {algorithm}")
    return loss


def test_mse_loss(computed_loss, expected_loss):
    torch.testing.assert_close(computed_loss, expected_loss)


class TestD3PMLossCalculator:
    @pytest.fixture
    def batch_size(self):
        return 1

    @pytest.fixture
    def number_of_atoms(self):
        return 2

    @pytest.fixture
    def num_atom_types(self):
        return 3

    @pytest.fixture
    def predicted_unnormalized_probabilities(
        self, batch_size, number_of_atoms, num_atom_types
    ):
        return torch.randn(batch_size, number_of_atoms, num_atom_types)

    @pytest.fixture
    def one_hot_real_atom_types(self, batch_size, number_of_atoms, num_atom_types):
        one_hot_real_atom_types = torch.zeros(
            batch_size, number_of_atoms, num_atom_types
        )
        for i in range(number_of_atoms):
            one_hot_real_atom_types[:, i, i] = 1
        return one_hot_real_atom_types

    @pytest.fixture
    def one_hot_different_noisy_atom_types(
        self, batch_size, number_of_atoms, num_atom_types
    ):
        one_hot_noisy_atom_types = torch.zeros(
            batch_size, number_of_atoms, num_atom_types
        )
        for i in range(number_of_atoms):
            one_hot_noisy_atom_types[:, i, i + 1] = 1
        return one_hot_noisy_atom_types

    @pytest.fixture
    def one_hot_similar_noisy_atom_types(
        self, batch_size, number_of_atoms, num_atom_types
    ):
        one_hot_noisy_atom_types = torch.zeros(
            batch_size, number_of_atoms, num_atom_types
        )
        for i in range(1, number_of_atoms):
            one_hot_noisy_atom_types[:, i, i + 1] = 1
        one_hot_noisy_atom_types[:, 0, 0] = 1
        return one_hot_noisy_atom_types

    @pytest.fixture
    def q_matrices(self, num_atom_types):
        return torch.eye(num_atom_types).view(1, 1, num_atom_types, num_atom_types)

    @pytest.fixture
    def q_bar_matrices(self, num_atom_types):
        return torch.eye(num_atom_types).view(1, 1, num_atom_types, num_atom_types)

    @pytest.fixture
    def q_bar_tm1_matrices(self, num_atom_types):
        return torch.eye(num_atom_types).view(1, 1, num_atom_types, num_atom_types)

    @pytest.fixture
    def loss_eps(self):
        return 1e-8

    @pytest.fixture
    def loss_parameters(self, loss_eps):
        return LossParameters(coordinates_algorithm=None, atom_types_eps=loss_eps)

    @pytest.fixture
    def d3pm_calculator(self, loss_parameters):
        return D3PMLossCalculator(loss_parameters)

    @pytest.fixture
    def expected_q(self, batch_size, number_of_atoms, num_atom_types):
        # with q / q_bar as identities, there is no possible transitions, so all classes are equivalent
        # q=(1/num_classes) * num_classes
        return torch.ones(batch_size, number_of_atoms, num_atom_types) / num_atom_types

    def test_kl_loss(
        self,
        predicted_unnormalized_probabilities,
        one_hot_real_atom_types,
        one_hot_different_noisy_atom_types,
        one_hot_similar_noisy_atom_types,
        q_matrices,
        q_bar_matrices,
        q_bar_tm1_matrices,
        d3pm_calculator,
        expected_q,
        loss_eps,
    ):
        computed_kl = d3pm_calculator.kl_loss_term(
            predicted_unnormalized_probabilities,
            one_hot_real_atom_types,
            one_hot_different_noisy_atom_types,
            q_matrices,
            q_bar_matrices,
            q_bar_tm1_matrices,
        )
        # with diagonal Q matrices, the expected posterior q is zero if the noisy types are different from the original
        # since 1 atom type can only stay the same (diagonal Q)

        assert torch.allclose(computed_kl, torch.zeros_like(computed_kl))

        computed_kl = d3pm_calculator.kl_loss_term(
            predicted_unnormalized_probabilities,
            one_hot_real_atom_types,
            one_hot_similar_noisy_atom_types,
            q_matrices,
            q_bar_matrices,
            q_bar_tm1_matrices,
        )
        # with 1 atom as the same type, posterior q should now be (1, 0, 0, ...)
        expected_q = torch.zeros_like(computed_kl)
        expected_q[:, 0, 0] = 1
        expected_kl = expected_q * torch.log(
            expected_q + loss_eps
        ) - expected_q * torch.nn.functional.log_softmax(
            predicted_unnormalized_probabilities, dim=-1
        )
        assert torch.allclose(computed_kl, expected_kl)

    def test_get_p_atm1_at(
        self, batch_size, number_of_atoms, num_atom_types, d3pm_calculator
    ):
        predicted_unnormalized_probabilities = torch.rand(
            batch_size, number_of_atoms, num_atom_types
        )
        q_at_bar_atm1 = torch.rand(batch_size, number_of_atoms, num_atom_types)
        q_bar_tm1_matrices = torch.rand(
            batch_size, number_of_atoms, num_atom_types, num_atom_types
        )

        computed_p_atm1_at = d3pm_calculator.get_p_atm1_at(
            predicted_unnormalized_probabilities,
            q_at_bar_atm1,
            q_bar_tm1_matrices,
        )

        expected_p_atm1_at = torch.zeros(batch_size, number_of_atoms, num_atom_types)
        normalized_predictions = torch.softmax(
            predicted_unnormalized_probabilities, dim=-1
        )

        for i in range(num_atom_types):
            tilde_a_0 = torch.nn.functional.one_hot(
                torch.LongTensor([i]), num_classes=num_atom_types
            ).float()
            tilde_a_0_qbar_tm1 = einops.einsum(
                tilde_a_0,
                torch.transpose(q_bar_tm1_matrices, -2, -1),
                "... j, ... i j -> ... i",
            )
            expected_p_atm1_at += (
                q_at_bar_atm1
                * tilde_a_0_qbar_tm1
                * normalized_predictions[..., i].unsqueeze(-1)
            )

        assert torch.allclose(computed_p_atm1_at, expected_p_atm1_at)

    @pytest.mark.parametrize("time_index_zero", [True, False])
    def test_calculate_unreduced_loss(
        self,
        time_index_zero,
        d3pm_calculator,
        batch_size,
        number_of_atoms,
        num_atom_types,
    ):
        predicted_probs = torch.randn(batch_size, number_of_atoms, num_atom_types)
        real_atom_types = (
            torch.eye(num_atom_types)
            .unsqueeze(0)
            .repeat(batch_size, number_of_atoms, 1, 1)
        )
        noisy_atom_types = (
            torch.eye(num_atom_types)
            .unsqueeze(0)
            .repeat(batch_size, number_of_atoms, 1, 1)
        )
        q_matrices = torch.randn(
            batch_size, number_of_atoms, num_atom_types, num_atom_types
        )
        q_bar_matrices = torch.randn(
            batch_size, number_of_atoms, num_atom_types, num_atom_types
        )
        q_bar_tm1_matrices = torch.randn(
            batch_size, number_of_atoms, num_atom_types, num_atom_types
        )

        # Mock the KL loss term output
        mock_kl_loss_output = torch.randn(batch_size, number_of_atoms, num_atom_types)

        # Define time_indices: 0 for NLL and 1 for KL + NLL (depending on parametrize input)
        if time_index_zero:
            time_indices = torch.zeros(
                batch_size, dtype=torch.long
            )  # t == 1 case (index 0)
        else:
            time_indices = torch.ones(batch_size, dtype=torch.long)  # t > 1 case

        # Patch the kl_loss_term method
        with patch.object(
            d3pm_calculator, "kl_loss_term", return_value=mock_kl_loss_output
        ) as mock_kl_loss:
            # Call the function under test
            computed_loss = d3pm_calculator.calculate_unreduced_loss(
                predicted_probs,
                real_atom_types,
                noisy_atom_types,
                time_indices,
                q_matrices,
                q_bar_matrices,
                q_bar_tm1_matrices,
            )

            mock_kl_loss.assert_called_once_with(
                predicted_probs,
                real_atom_types,
                noisy_atom_types,
                q_matrices,
                q_bar_matrices,
                q_bar_tm1_matrices,
            )

            # Compute expected NLL term
            nll_term = -torch.nn.functional.log_softmax(predicted_probs, dim=-1)

            if time_index_zero:
                # If time_indices == 0, loss should be equal to NLL term
                assert torch.allclose(computed_loss, nll_term)
            else:
                # If time_indices != 0, loss should be KL term + ce_weight * NLL term
                expected_loss = (
                    mock_kl_loss_output + d3pm_calculator.ce_weight * nll_term
                )
                assert torch.allclose(computed_loss, expected_loss)
