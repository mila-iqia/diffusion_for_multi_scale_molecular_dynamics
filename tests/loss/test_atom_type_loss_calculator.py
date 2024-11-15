from unittest.mock import patch

import pytest
import torch
from torch.nn import KLDivLoss

from diffusion_for_multi_scale_molecular_dynamics.loss import (
    D3PMLossCalculator, LossParameters)
from diffusion_for_multi_scale_molecular_dynamics.utils.d3pm_utils import \
    class_index_to_onehot
from diffusion_for_multi_scale_molecular_dynamics.utils.tensor_utils import \
    broadcast_batch_matrix_tensor_to_all_dimensions


class TestD3PMLossCalculator:

    @pytest.fixture(scope="class", autouse=True)
    def set_seed(self):
        """Set the random seed."""
        torch.manual_seed(3423423)

    @pytest.fixture
    def batch_size(self):
        return 4

    @pytest.fixture
    def number_of_atoms(self):
        return 8

    @pytest.fixture
    def num_atom_types(self):
        return 5

    @pytest.fixture
    def num_classes(self, num_atom_types):
        return num_atom_types + 1

    @pytest.fixture
    def predicted_logits(self, batch_size, number_of_atoms, num_classes):
        logits = 10 * (torch.randn(batch_size, number_of_atoms, num_classes) - 0.5)
        logits[:, :, -1] = -torch.inf  # force the model to never predict MASK
        return logits

    @pytest.fixture
    def predicted_p_a0_given_at(self, predicted_logits):
        return torch.nn.functional.softmax(predicted_logits, dim=-1)

    @pytest.fixture
    def one_hot_a0(self, batch_size, number_of_atoms, num_atom_types, num_classes):
        # a0 CANNOT be MASK.
        one_hot_indices = torch.randint(
            0,
            num_atom_types,
            (
                batch_size,
                number_of_atoms,
            ),
        )
        one_hots = class_index_to_onehot(one_hot_indices, num_classes=num_classes)
        return one_hots

    @pytest.fixture
    def one_hot_at(self, batch_size, number_of_atoms, num_atom_types, num_classes):
        # at CAN be MASK.
        one_hot_indices = torch.randint(
            0,
            num_classes,
            (
                batch_size,
                number_of_atoms,
            ),
        )
        one_hots = class_index_to_onehot(one_hot_indices, num_classes=num_classes)
        return one_hots

    @pytest.fixture
    def one_hot_different_noisy_atom_types(
        self, batch_size, number_of_atoms, num_classes
    ):
        one_hot_noisy_atom_types = torch.zeros(batch_size, number_of_atoms, num_classes)
        for i in range(number_of_atoms):
            one_hot_noisy_atom_types[:, i, i + 1] = 1
        return one_hot_noisy_atom_types

    @pytest.fixture
    def one_hot_similar_noisy_atom_types(
        self, batch_size, number_of_atoms, num_classes
    ):
        one_hot_noisy_atom_types = torch.zeros(batch_size, number_of_atoms, num_classes)
        for i in range(1, number_of_atoms):
            one_hot_noisy_atom_types[:, i, i + 1] = 1
        one_hot_noisy_atom_types[:, 0, 0] = 1
        return one_hot_noisy_atom_types

    @pytest.fixture
    def q_matrices(self, batch_size, number_of_atoms, num_classes):
        random_q_matrices = torch.rand(batch_size, num_classes, num_classes)
        final_shape = (batch_size, number_of_atoms)
        broadcast_q_matrices = broadcast_batch_matrix_tensor_to_all_dimensions(
            random_q_matrices, final_shape=final_shape
        )
        return broadcast_q_matrices

    @pytest.fixture
    def q_bar_matrices(self, batch_size, number_of_atoms, num_classes):
        random_q_bar_matrices = torch.rand(batch_size, num_classes, num_classes)
        final_shape = (batch_size, number_of_atoms)
        broadcast_q_bar_matrices = broadcast_batch_matrix_tensor_to_all_dimensions(
            random_q_bar_matrices, final_shape=final_shape
        )
        return broadcast_q_bar_matrices

    @pytest.fixture
    def q_bar_tm1_matrices(self, batch_size, number_of_atoms, num_classes):
        random_q_bar_tm1_matrices = torch.rand(batch_size, num_classes, num_classes)
        final_shape = (batch_size, number_of_atoms)
        broadcast_q_bar_tm1_matrices = broadcast_batch_matrix_tensor_to_all_dimensions(
            random_q_bar_tm1_matrices, final_shape=final_shape
        )
        return broadcast_q_bar_tm1_matrices

    @pytest.fixture
    def loss_eps(self):
        return 1.0e-12

    @pytest.fixture
    def loss_parameters(self, loss_eps):
        return LossParameters(coordinates_algorithm=None, atom_types_eps=loss_eps)

    @pytest.fixture
    def d3pm_calculator(self, loss_parameters):
        return D3PMLossCalculator(loss_parameters)

    @pytest.fixture
    def expected_p_atm1_given_at(
        self,
        predicted_p_a0_given_at,
        one_hot_at,
        q_matrices,
        q_bar_matrices,
        q_bar_tm1_matrices,
    ):
        batch_size, natoms, num_classes = predicted_p_a0_given_at.shape

        denominator = torch.zeros(batch_size, natoms)
        numerator1 = torch.zeros(batch_size, natoms, num_classes)
        numerator2 = torch.zeros(batch_size, natoms, num_classes)

        for i in range(num_classes):
            for j in range(num_classes):
                denominator[:, :] += (
                    predicted_p_a0_given_at[:, :, i]
                    * q_bar_matrices[:, :, i, j]
                    * one_hot_at[:, :, j]
                )
                numerator1[:, :, i] += (
                    predicted_p_a0_given_at[:, :, j] * q_bar_tm1_matrices[:, :, j, i]
                )
                numerator2[:, :, i] += q_matrices[:, :, i, j] * one_hot_at[:, :, j]

        numerator = numerator1 * numerator2

        expected_p = torch.zeros(batch_size, natoms, num_classes)
        for i in range(num_classes):
            expected_p[:, :, i] = numerator[:, :, i] / denominator[:, :]

        # Note that the expected_p_atm1_given_at is not really a probability (and thus does not sum to 1) because
        # the Q matrices are random.
        return expected_p

    @pytest.fixture
    def expected_q_atm1_given_at_and_a0(
        self, one_hot_a0, one_hot_at, q_matrices, q_bar_matrices, q_bar_tm1_matrices
    ):
        batch_size, natoms, num_classes = one_hot_a0.shape

        denominator = torch.zeros(batch_size, natoms)
        numerator1 = torch.zeros(batch_size, natoms, num_classes)
        numerator2 = torch.zeros(batch_size, natoms, num_classes)

        for i in range(num_classes):
            for j in range(num_classes):
                denominator[:, :] += (
                    one_hot_a0[:, :, i]
                    * q_bar_matrices[:, :, i, j]
                    * one_hot_at[:, :, j]
                )
                numerator1[:, :, i] += (
                    one_hot_a0[:, :, j] * q_bar_tm1_matrices[:, :, j, i]
                )
                numerator2[:, :, i] += q_matrices[:, :, i, j] * one_hot_at[:, :, j]

        numerator = numerator1 * numerator2

        expected_q = torch.zeros(batch_size, natoms, num_classes)
        for i in range(num_classes):
            expected_q[:, :, i] = numerator[:, :, i] / denominator[:, :]

        return expected_q

    @pytest.fixture
    def expected_kl_loss(
        self, expected_p_atm1_given_at, expected_q_atm1_given_at_and_a0
    ):
        kl_loss = KLDivLoss(reduction="none")
        log_p = torch.log(expected_p_atm1_given_at)
        return kl_loss(input=log_p, target=expected_q_atm1_given_at_and_a0)

    def test_get_p_atm1_at(
        self,
        predicted_logits,
        one_hot_at,
        q_matrices,
        q_bar_matrices,
        q_bar_tm1_matrices,
        loss_eps,
        d3pm_calculator,
        expected_p_atm1_given_at,
    ):
        computed_p_atm1_given_at = d3pm_calculator.get_p_atm1_given_at(
            predicted_logits,
            one_hot_at,
            q_matrices,
            q_bar_matrices,
            q_bar_tm1_matrices,
            small_epsilon=loss_eps,
        )

        assert torch.allclose(computed_p_atm1_given_at, expected_p_atm1_given_at)

    def test_get_q_atm1_given_at_and_a0(
        self,
        one_hot_a0,
        one_hot_at,
        q_matrices,
        q_bar_matrices,
        q_bar_tm1_matrices,
        loss_eps,
        d3pm_calculator,
        expected_q_atm1_given_at_and_a0,
    ):
        computed_q_atm1_given_at_and_a0 = d3pm_calculator.get_q_atm1_given_at_and_a0(
            one_hot_a0,
            one_hot_at,
            q_matrices,
            q_bar_matrices,
            q_bar_tm1_matrices,
            small_epsilon=loss_eps,
        )

        assert torch.allclose(
            computed_q_atm1_given_at_and_a0, expected_q_atm1_given_at_and_a0
        )

    def test_kl_loss(
        self,
        predicted_logits,
        one_hot_a0,
        one_hot_at,
        q_matrices,
        q_bar_matrices,
        q_bar_tm1_matrices,
        d3pm_calculator,
        loss_eps,
        expected_kl_loss,
    ):
        computed_kl_loss = d3pm_calculator.kl_loss_term(
            predicted_logits,
            one_hot_a0,
            one_hot_at,
            q_matrices,
            q_bar_matrices,
            q_bar_tm1_matrices,
        )

        torch.testing.assert_close(computed_kl_loss, expected_kl_loss)

    def test_kl_loss_predicting_a0(
        self,
        one_hot_a0,
        one_hot_at,
        q_matrices,
        q_bar_matrices,
        q_bar_tm1_matrices,
        d3pm_calculator,
        loss_eps,
        expected_kl_loss,
    ):
        # The KL should vanish when p_\theta(. | a_t) predicts a0 with probability 1.

        predicted_logits = torch.log(one_hot_a0)

        computed_kl_loss = d3pm_calculator.kl_loss_term(
            predicted_logits,
            one_hot_a0,
            one_hot_at,
            q_matrices,
            q_bar_matrices,
            q_bar_tm1_matrices,
        )

        torch.testing.assert_close(computed_kl_loss, torch.zeros_like(computed_kl_loss))

    def test_kl_loss_diagonal_q_matrices(
        self,
        num_classes,
        d3pm_calculator,
    ):
        # with diagonal Q matrices, the KL is ALWAYS ZERO. This is because either:
        #    1) the posterior is all zero
        #  or
        #   2) the prediction is equal to the posterior; this follows because the prediction is normalized.
        predicted_logits = torch.rand(1, 1, num_classes)

        q_matrices = torch.eye(num_classes).view(1, 1, num_classes, num_classes)
        q_bar_matrices = torch.eye(num_classes).view(1, 1, num_classes, num_classes)
        q_bar_tm1_matrices = torch.eye(num_classes).view(1, 1, num_classes, num_classes)
        for i in range(num_classes):
            for j in range(num_classes):
                one_hot_a0 = torch.zeros(1, 1, num_classes)
                one_hot_at = torch.zeros(1, 1, num_classes)
                one_hot_a0[0, 0, i] = 1.0
                one_hot_at[0, 0, j] = 1.0

                computed_kl = d3pm_calculator.kl_loss_term(
                    predicted_logits,
                    one_hot_a0,
                    one_hot_at,
                    q_matrices,
                    q_bar_matrices,
                    q_bar_tm1_matrices,
                )
                torch.testing.assert_close(computed_kl, torch.zeros_like(computed_kl))

    @pytest.mark.parametrize("time_index_zero", [True, False])
    def test_calculate_unreduced_loss(
        self,
        time_index_zero,
        d3pm_calculator,
        batch_size,
        number_of_atoms,
        num_classes,
    ):
        predicted_logits = torch.randn(batch_size, number_of_atoms, num_classes)
        predicted_logits[..., -1] = -torch.inf

        real_atom_types = (
            torch.eye(num_classes)
            .unsqueeze(0)
            .repeat(batch_size, number_of_atoms, 1, 1)
        )
        noisy_atom_types = (
            torch.eye(num_classes)
            .unsqueeze(0)
            .repeat(batch_size, number_of_atoms, 1, 1)
        )
        q_matrices = torch.randn(batch_size, number_of_atoms, num_classes, num_classes)
        q_bar_matrices = torch.randn(
            batch_size, number_of_atoms, num_classes, num_classes
        )
        q_bar_tm1_matrices = torch.randn(
            batch_size, number_of_atoms, num_classes, num_classes
        )

        # Mock the KL loss term output
        mock_kl_loss_output = torch.randn(batch_size, number_of_atoms, num_classes)

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
                predicted_logits,
                real_atom_types,
                noisy_atom_types,
                time_indices,
                q_matrices,
                q_bar_matrices,
                q_bar_tm1_matrices,
            )

            mock_kl_loss.assert_called_once_with(
                predicted_logits,
                real_atom_types,
                noisy_atom_types,
                q_matrices,
                q_bar_matrices,
                q_bar_tm1_matrices,
            )

            # Compute expected NLL term
            nll_term = -torch.nn.functional.log_softmax(predicted_logits, dim=-1)
            nll_term[..., -1] = 0.0

            if time_index_zero:
                # If time_indices == 0, loss should be equal to NLL term
                torch.testing.assert_close(computed_loss, nll_term)
            else:
                # If time_indices != 0, loss should be KL term + ce_weight * NLL term
                expected_loss = (
                    mock_kl_loss_output + d3pm_calculator.ce_weight * nll_term
                )
                torch.testing.assert_close(computed_loss, expected_loss)
