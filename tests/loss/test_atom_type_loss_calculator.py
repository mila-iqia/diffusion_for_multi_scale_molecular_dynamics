from unittest.mock import patch

import pytest
import torch
from torch.nn import KLDivLoss

from diffusion_for_multi_scale_molecular_dynamics.loss import \
    D3PMLossCalculator
from diffusion_for_multi_scale_molecular_dynamics.loss.loss_parameters import \
    AtomTypeLossParameters
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
        return 64

    @pytest.fixture
    def number_of_atoms(self):
        return 8

    @pytest.fixture
    def num_atom_types(self):
        return 5

    @pytest.fixture
    def total_number_of_times_steps(self):
        return 8

    @pytest.fixture
    def time_indices(self, batch_size, total_number_of_times_steps):
        return torch.randint(0, total_number_of_times_steps, (batch_size,))

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
    def atom_types_ce_weight(self):
        return 0.1

    @pytest.fixture
    def loss_parameters(self, loss_eps, atom_types_ce_weight):
        return AtomTypeLossParameters(
            algorithm=None,
            eps=loss_eps,
            ce_weight=atom_types_ce_weight,
        )

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
    def expected_vb_loss(
        self,
        time_indices,
        one_hot_a0,
        expected_p_atm1_given_at,
        expected_q_atm1_given_at_and_a0,
    ):
        assert (
            0 in time_indices
        ), "For a good test, the index 0 should appear in the time indices!"

        kl_loss = KLDivLoss(reduction="none")
        log_p = torch.log(expected_p_atm1_given_at)
        vb_loss = kl_loss(input=log_p, target=expected_q_atm1_given_at_and_a0)

        for batch_idx, time_index in enumerate(time_indices):
            if time_index == 0:
                vb_loss[batch_idx] = -log_p[batch_idx] * one_hot_a0[batch_idx]

        return vb_loss

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

    def test_variational_bound_loss(
        self,
        predicted_logits,
        one_hot_a0,
        one_hot_at,
        q_matrices,
        q_bar_matrices,
        q_bar_tm1_matrices,
        time_indices,
        d3pm_calculator,
        loss_eps,
        expected_vb_loss,
    ):
        computed_vb_loss = d3pm_calculator.variational_bound_loss_term(
            predicted_logits,
            one_hot_a0,
            one_hot_at,
            q_matrices,
            q_bar_matrices,
            q_bar_tm1_matrices,
            time_indices,
        )

        torch.testing.assert_close(computed_vb_loss, expected_vb_loss)

    def test_vb_loss_predicting_a0(
        self,
        one_hot_a0,
        one_hot_at,
        q_matrices,
        q_bar_matrices,
        q_bar_tm1_matrices,
        time_indices,
        d3pm_calculator,
    ):
        # The KL should vanish when p_\theta(. | a_t) predicts a0 with probability 1.

        predicted_logits = torch.log(one_hot_a0)

        computed_vb_loss = d3pm_calculator.variational_bound_loss_term(
            predicted_logits,
            one_hot_a0,
            one_hot_at,
            q_matrices,
            q_bar_matrices,
            q_bar_tm1_matrices,
            time_indices,
        )

        non_zero_time_step_mask = time_indices != 0
        computed_kl_loss = computed_vb_loss[non_zero_time_step_mask]

        torch.testing.assert_close(computed_kl_loss, torch.zeros_like(computed_kl_loss))

    def test_cross_entropy_loss_term(
        self, predicted_logits, one_hot_a0, d3pm_calculator
    ):
        computed_ce_loss = d3pm_calculator.cross_entropy_loss_term(
            predicted_logits, one_hot_a0
        )

        p = torch.softmax(predicted_logits, dim=-1)
        log_p = torch.log(p)
        log_p[..., -1] = 0.0
        expected_ce_loss = -log_p * one_hot_a0

        torch.testing.assert_close(computed_ce_loss, expected_ce_loss)

    def test_calculate_unreduced_loss(
        self,
        predicted_logits,
        one_hot_a0,
        one_hot_at,
        q_matrices,
        q_bar_matrices,
        q_bar_tm1_matrices,
        time_indices,
        d3pm_calculator,
        atom_types_ce_weight,
    ):
        vb_loss = d3pm_calculator.variational_bound_loss_term(
            predicted_logits,
            one_hot_a0,
            one_hot_at,
            q_matrices,
            q_bar_matrices,
            q_bar_tm1_matrices,
            time_indices,
        )

        ce_loss = d3pm_calculator.cross_entropy_loss_term(predicted_logits, one_hot_a0)
        expected_losss = vb_loss + atom_types_ce_weight * ce_loss

        computed_loss = d3pm_calculator.calculate_unreduced_loss(
            predicted_logits,
            one_hot_a0,
            one_hot_at,
            time_indices,
            q_matrices,
            q_bar_matrices,
            q_bar_tm1_matrices,
        )

        torch.testing.assert_close(computed_loss, expected_losss)

    @pytest.mark.parametrize("time_index_zero", [True, False])
    def test_variational_bound_call(
        self,
        time_index_zero,
        d3pm_calculator,
        batch_size,
        number_of_atoms,
        num_classes,
    ):
        predicted_logits = torch.randn(batch_size, number_of_atoms, num_classes)
        predicted_logits[..., -1] = -torch.inf

        real_atom_types = torch.randint(0, num_classes, (batch_size, number_of_atoms))
        real_atom_types = class_index_to_onehot(
            real_atom_types, num_classes=num_classes
        )

        noisy_atom_types = torch.randint(0, num_classes, (batch_size, number_of_atoms))
        noisy_atom_types = class_index_to_onehot(
            noisy_atom_types, num_classes=num_classes
        )

        q_matrices = torch.randn(batch_size, number_of_atoms, num_classes, num_classes)
        q_bar_matrices = torch.randn(
            batch_size, number_of_atoms, num_classes, num_classes
        )
        q_bar_tm1_matrices = torch.randn(
            batch_size, number_of_atoms, num_classes, num_classes
        )

        # Mock the KL loss term output
        mock_vb_loss_output = torch.randn(batch_size, number_of_atoms, num_classes)

        # Define time_indices: 0 for NLL and 1 for KL + NLL (depending on parametrize input)
        if time_index_zero:
            time_indices = torch.zeros(
                batch_size, dtype=torch.long
            )  # t == 1 case (index 0)
        else:
            time_indices = torch.ones(batch_size, dtype=torch.long)  # t > 1 case

        # Patch the kl_loss_term method
        with patch.object(
            d3pm_calculator,
            "variational_bound_loss_term",
            return_value=mock_vb_loss_output,
        ) as mock_vb_loss:
            # Call the function under test
            _ = d3pm_calculator.calculate_unreduced_loss(
                predicted_logits,
                real_atom_types,
                noisy_atom_types,
                time_indices,
                q_matrices,
                q_bar_matrices,
                q_bar_tm1_matrices,
            )

            mock_vb_loss.assert_called_once_with(
                predicted_logits,
                real_atom_types,
                noisy_atom_types,
                q_matrices,
                q_bar_matrices,
                q_bar_tm1_matrices,
                time_indices,
            )
