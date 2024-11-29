import einops
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.langevin_generator import \
    LangevinGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from src.diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_scheduler import \
    NoiseScheduler
from tests.generators.conftest import BaseTestGenerator


class TestLangevinGenerator(BaseTestGenerator):

    @pytest.fixture()
    def num_atom_types(self):
        return 4

    @pytest.fixture()
    def num_atomic_classes(self, num_atom_types):
        return num_atom_types + 1

    @pytest.fixture(params=[0, 2])
    def number_of_corrector_steps(self, request):
        return request.param

    @pytest.fixture(params=[2, 5, 10])
    def total_time_steps(self, request):
        return request.param

    @pytest.fixture()
    def sigma_min(self):
        return 0.15

    @pytest.fixture()
    def noise_parameters(self, total_time_steps, sigma_min):
        noise_parameters = NoiseParameters(
            total_time_steps=total_time_steps,
            time_delta=0.1,
            sigma_min=sigma_min,
            corrector_step_epsilon=0.25,
        )
        return noise_parameters

    @pytest.fixture()
    def small_epsilon(self):
        return 1e-6

    @pytest.fixture(params=[True, False])
    def one_atom_type_transition_per_step(self, request):
        return request.param

    @pytest.fixture(params=[True, False])
    def atom_type_greedy_sampling(self, request):
        return request.param

    @pytest.fixture(params=[True, False])
    def atom_type_transition_in_corrector(self, request):
        return request.param

    @pytest.fixture()
    def sampling_parameters(
        self,
        number_of_atoms,
        spatial_dimension,
        cell_dimensions,
        number_of_samples,
        number_of_corrector_steps,
        unit_cell_size,
        num_atom_types,
        one_atom_type_transition_per_step,
        atom_type_greedy_sampling,
        atom_type_transition_in_corrector,
        small_epsilon,
    ):
        sampling_parameters = PredictorCorrectorSamplingParameters(
            number_of_corrector_steps=number_of_corrector_steps,
            number_of_atoms=number_of_atoms,
            number_of_samples=number_of_samples,
            cell_dimensions=cell_dimensions,
            spatial_dimension=spatial_dimension,
            num_atom_types=num_atom_types,
            one_atom_type_transition_per_step=one_atom_type_transition_per_step,
            atom_type_greedy_sampling=atom_type_greedy_sampling,
            atom_type_transition_in_corrector=atom_type_transition_in_corrector,
            small_epsilon=small_epsilon,
        )

        return sampling_parameters

    @pytest.fixture()
    def noise(self, noise_parameters, num_atomic_classes, device):
        sampler = NoiseScheduler(noise_parameters, num_classes=num_atomic_classes).to(
            device
        )
        noise, _ = sampler.get_all_sampling_parameters()
        return noise

    @pytest.fixture()
    def pc_generator(self, noise_parameters, sampling_parameters, axl_network):
        generator = LangevinGenerator(
            noise_parameters=noise_parameters,
            sampling_parameters=sampling_parameters,
            axl_network=axl_network,
        )

        return generator

    def test_smoke_sample(
        self, pc_generator, device, number_of_samples, unit_cell_sample
    ):
        # Just a smoke test that we can sample without crashing.
        pc_generator.sample(number_of_samples, device, unit_cell_sample)

    @pytest.fixture()
    def axl_i(
        self,
        number_of_samples,
        number_of_atoms,
        spatial_dimension,
        num_atomic_classes,
        device,
    ):
        return AXL(
            A=torch.randint(
                0, num_atomic_classes, (number_of_samples, number_of_atoms)
            ).to(device),
            X=map_relative_coordinates_to_unit_cell(
                torch.rand(number_of_samples, number_of_atoms, spatial_dimension)
            ).to(device),
            L=torch.zeros(
                number_of_samples, spatial_dimension * (spatial_dimension - 1)
            ).to(
                device
            ),  # TODO placeholder
        )

    def test_predictor_step_relative_coordinates(
        self,
        mocker,
        pc_generator,
        noise,
        sigma_min,
        axl_i,
        total_time_steps,
        number_of_samples,
        unit_cell_sample,
    ):
        list_sigma = noise.sigma
        list_time = noise.time
        forces = torch.zeros_like(axl_i.X)

        z = pc_generator._draw_gaussian_sample(number_of_samples).to(axl_i.X)
        mocker.patch.object(pc_generator, "_draw_gaussian_sample", return_value=z)

        for index_i in range(1, total_time_steps + 1):
            computed_sample = pc_generator.predictor_step(
                axl_i, index_i, unit_cell_sample, forces
            )

            sigma_i = list_sigma[index_i - 1]
            t_i = list_time[index_i - 1]
            if index_i == 1:
                sigma_im1 = sigma_min
            else:
                sigma_im1 = list_sigma[index_i - 2]

            g2 = sigma_i**2 - sigma_im1**2

            s_i = (
                pc_generator._get_model_predictions(
                    axl_i, t_i, sigma_i, unit_cell_sample, forces
                ).X
                / sigma_i
            )

            expected_coordinates = axl_i.X + g2 * s_i + torch.sqrt(g2) * z
            expected_coordinates = map_relative_coordinates_to_unit_cell(
                expected_coordinates
            )

            torch.testing.assert_close(computed_sample.X, expected_coordinates)

    def test_adjust_atom_types_probabilities_for_greedy_sampling(
        self, pc_generator, number_of_atoms, num_atomic_classes
    ):
        # Test that all_masked atom types are unaffected.
        fully_masked_row = pc_generator.masked_atom_type_index * torch.ones(
            number_of_atoms, dtype=torch.int64
        )

        partially_unmasked_row = fully_masked_row.clone()
        partially_unmasked_row[0] = 0

        atom_types_i = torch.stack([fully_masked_row, partially_unmasked_row])

        number_of_samples = atom_types_i.shape[0]
        u = pc_generator._draw_gumbel_sample(number_of_samples)

        one_step_transition_probs = torch.rand(
            number_of_samples, number_of_atoms, num_atomic_classes
        ).softmax(dim=-1)
        # Use cloned values because the method overrides the inputs.
        updated_one_step_transition_probs, updated_u = (
            pc_generator._adjust_atom_types_probabilities_for_greedy_sampling(
                one_step_transition_probs.clone(), atom_types_i, u.clone()
            )
        )

        # Test that the fully masked row is unaffected
        torch.testing.assert_close(
            updated_one_step_transition_probs[0], one_step_transition_probs[0]
        )
        torch.testing.assert_close(u[0], updated_u[0])

        # Test that when an atom is unmasked, the probabilities are set up for greedy sampling:
        # - the probabilities for the real atomic classes are unchanged.
        # - the probability for the MASK class (last index) is either unchanged or set to zero.
        # - the Gumbel sample is set to zero so that the unmasking is greedy.

        torch.testing.assert_close(
            updated_one_step_transition_probs[1, :, :-1],
            one_step_transition_probs[1, :, :-1],
        )

        m1 = (
            updated_one_step_transition_probs[1, :, -1]
            == one_step_transition_probs[1, :, -1]
        )
        m2 = updated_one_step_transition_probs[1, :, -1] == 0.0
        assert torch.logical_or(m1, m2).all()
        torch.testing.assert_close(updated_u[1], torch.zeros_like(updated_u[1]))

    def test_get_updated_atom_types_for_one_transition_per_step_is_idempotent(
        self,
        pc_generator,
        number_of_samples,
        number_of_atoms,
        num_atomic_classes,
        device,
    ):
        # Test that the method returns the current atom types if there is no proposed changes.
        current_atom_types = torch.randint(
            0, num_atomic_classes, (number_of_samples, number_of_atoms)
        ).to(device)
        sampled_atom_types = current_atom_types.clone()
        max_gumbel_values = torch.rand(number_of_samples, number_of_atoms).to(device)

        updated_atom_types = (
            pc_generator._get_updated_atom_types_for_one_transition_per_step(
                current_atom_types, max_gumbel_values, sampled_atom_types
            )
        )

        torch.testing.assert_close(updated_atom_types, current_atom_types)

    def test_get_updated_atom_types_for_one_transition_per_step(
        self,
        pc_generator,
        number_of_samples,
        number_of_atoms,
        num_atomic_classes,
        device,
    ):
        assert (
            num_atomic_classes > 0
        ), "Cannot run this test with a single atomic class."
        current_atom_types = torch.randint(
            0, num_atomic_classes, (number_of_samples, number_of_atoms)
        ).to(device)
        sampled_atom_types = torch.randint(
            0, num_atomic_classes, (number_of_samples, number_of_atoms)
        ).to(device)
        # Make sure at least one atom is different in every sample.
        while not (current_atom_types != sampled_atom_types).any(dim=-1).all():
            sampled_atom_types = torch.randint(
                0, num_atomic_classes, (number_of_samples, number_of_atoms)
            ).to(device)

        proposed_difference_mask = current_atom_types != sampled_atom_types

        max_gumbel_values = torch.rand(number_of_samples, number_of_atoms).to(device)

        updated_atom_types = (
            pc_generator._get_updated_atom_types_for_one_transition_per_step(
                current_atom_types, max_gumbel_values, sampled_atom_types
            )
        )

        difference_mask = updated_atom_types != current_atom_types

        # Check that there is a single difference per sample
        number_of_changes = difference_mask.sum(dim=-1)
        torch.testing.assert_close(
            number_of_changes, torch.ones(number_of_samples).to(number_of_changes)
        )

        # Check that the difference is at the location of the maximum value of the Gumbel random variable over the
        # possible changes.
        computed_changed_atom_indices = torch.where(difference_mask)[1]

        expected_changed_atom_indices = []
        for sample_idx in range(number_of_samples):
            sample_gumbel_values = max_gumbel_values[sample_idx].clone()
            sample_proposed_difference_mask = proposed_difference_mask[sample_idx]
            sample_gumbel_values[~sample_proposed_difference_mask] = -torch.inf
            max_index = torch.argmax(sample_gumbel_values)
            expected_changed_atom_indices.append(max_index)
        expected_changed_atom_indices = torch.tensor(expected_changed_atom_indices).to(
            computed_changed_atom_indices
        )

        torch.testing.assert_close(
            computed_changed_atom_indices, expected_changed_atom_indices
        )

    def test_atom_types_update(
        self,
        pc_generator,
        noise,
        total_time_steps,
        num_atomic_classes,
        number_of_samples,
        number_of_atoms,
        device,
    ):

        # Initialize to fully masked
        a_i = pc_generator.masked_atom_type_index * torch.ones(
            number_of_samples, number_of_atoms, dtype=torch.int64
        ).to(device)

        for time_index_i in range(total_time_steps, 0, -1):
            this_is_last_time_step = time_index_i == 1
            idx = time_index_i - 1
            q_matrices_i = einops.repeat(
                noise.q_matrix[idx],
                "n1 n2 -> nsamples natoms n1 n2",
                nsamples=number_of_samples,
                natoms=number_of_atoms,
            )

            q_bar_matrices_i = einops.repeat(
                noise.q_bar_matrix[idx],
                "n1 n2 -> nsamples natoms n1 n2",
                nsamples=number_of_samples,
                natoms=number_of_atoms,
            )

            q_bar_tm1_matrices_i = einops.repeat(
                noise.q_bar_tm1_matrix[idx],
                "n1 n2 -> nsamples natoms n1 n2",
                nsamples=number_of_samples,
                natoms=number_of_atoms,
            )

            random_logits = torch.rand(
                number_of_samples, number_of_atoms, num_atomic_classes
            ).to(device)
            random_logits[:, :, -1] = -torch.inf

            one_atom_type_transition_per_step = (
                pc_generator.one_atom_type_transition_per_step
                and not this_is_last_time_step
            )

            a_im1 = pc_generator._atom_types_update(
                random_logits,
                a_i,
                q_matrices_i,
                q_bar_matrices_i,
                q_bar_tm1_matrices_i,
                atom_type_greedy_sampling=pc_generator.atom_type_greedy_sampling,
                one_atom_type_transition_per_step=one_atom_type_transition_per_step,
            )

            difference_mask = a_im1 != a_i

            # Test that the changes are from MASK to not-MASK
            assert (a_i[difference_mask] == pc_generator.masked_atom_type_index).all()
            assert (a_im1[difference_mask] != pc_generator.masked_atom_type_index).all()

            if one_atom_type_transition_per_step:
                # Test that there is at most one change
                assert torch.all(difference_mask.sum(dim=-1) <= 1.0)

            if pc_generator.atom_type_greedy_sampling:
                # Test that the changes are the most probable (greedy)
                sample_indices, atom_indices = torch.where(difference_mask)
                for sample_idx, atom_idx in zip(sample_indices, atom_indices):
                    # Greedy sampling only applies if at least one atom was already unmasked.
                    if (a_i[sample_idx] == pc_generator.masked_atom_type_index).all():
                        continue
                    computed_atom_type = a_im1[sample_idx, atom_idx]
                    expected_atom_type = random_logits[sample_idx, atom_idx].argmax()
                    assert computed_atom_type == expected_atom_type

            a_i = a_im1

        # Test that no MASKED states remain
        assert not (a_i == pc_generator.masked_atom_type_index).any()

    def test_predictor_step_atom_types(
        self,
        mocker,
        pc_generator,
        total_time_steps,
        number_of_samples,
        number_of_atoms,
        num_atomic_classes,
        spatial_dimension,
        unit_cell_sample,
        device,
    ):
        zeros = torch.zeros(number_of_samples, number_of_atoms, spatial_dimension).to(
            device
        )
        forces = zeros

        random_x = map_relative_coordinates_to_unit_cell(
            torch.rand(number_of_samples, number_of_atoms, spatial_dimension)
        ).to(device)

        random_l = torch.zeros(
            number_of_samples, spatial_dimension, spatial_dimension
        ).to(device)

        # Initialize to fully masked
        a_ip1 = pc_generator.masked_atom_type_index * torch.ones(
            number_of_samples, number_of_atoms, dtype=torch.int64
        ).to(device)
        axl_ip1 = AXL(A=a_ip1, X=random_x, L=random_l)

        for idx in range(total_time_steps - 1, -1, -1):

            # Inject reasonable logits
            logits = torch.rand(
                number_of_samples, number_of_atoms, num_atomic_classes
            ).to(device)
            logits[:, :, -1] = -torch.inf
            fake_model_predictions = AXL(A=logits, X=zeros, L=zeros)
            mocker.patch.object(
                pc_generator,
                "_get_model_predictions",
                return_value=fake_model_predictions,
            )

            axl_i = pc_generator.predictor_step(
                axl_ip1, idx + 1, unit_cell_sample, forces
            )

            this_is_last_time_step = idx == 0
            a_i = axl_i.A
            a_ip1 = axl_ip1.A

            difference_mask = a_ip1 != a_i

            # Test that the changes are from MASK to not-MASK
            assert (a_ip1[difference_mask] == pc_generator.masked_atom_type_index).all()
            assert (a_i[difference_mask] != pc_generator.masked_atom_type_index).all()

            one_atom_type_transition_per_step = (
                pc_generator.one_atom_type_transition_per_step
                and not this_is_last_time_step
            )

            if one_atom_type_transition_per_step:
                # Test that there is at most one change
                assert torch.all(difference_mask.sum(dim=-1) <= 1.0)

            axl_ip1 = AXL(A=a_i, X=random_x, L=random_l)

        # Test that no MASKED states remain
        a_i = axl_i.A
        assert not (a_i == pc_generator.masked_atom_type_index).any()

    def test_corrector_step(
        self,
        mocker,
        pc_generator,
        noise_parameters,
        axl_i,
        total_time_steps,
        number_of_samples,
        unit_cell_sample,
        num_atomic_classes,
    ):

        sampler = NoiseScheduler(noise_parameters, num_classes=num_atomic_classes)
        noise, _ = sampler.get_all_sampling_parameters()
        sigma_min = noise_parameters.sigma_min
        epsilon = noise_parameters.corrector_step_epsilon
        list_sigma = noise.sigma
        list_time = noise.time
        sigma_1 = list_sigma[0]
        forces = torch.zeros_like(axl_i.X)

        z = pc_generator._draw_gaussian_sample(number_of_samples).to(axl_i.X)
        mocker.patch.object(pc_generator, "_draw_gaussian_sample", return_value=z)

        for index_i in range(0, total_time_steps):
            computed_sample = pc_generator.corrector_step(
                axl_i, index_i, unit_cell_sample, forces
            )

            if index_i == 0:
                sigma_i = sigma_min
                t_i = 0.0
            else:
                sigma_i = list_sigma[index_i - 1]
                t_i = list_time[index_i - 1]

            eps_i = 0.5 * epsilon * sigma_i**2 / sigma_1**2

            s_i = (
                pc_generator._get_model_predictions(
                    axl_i, t_i, sigma_i, unit_cell_sample, forces
                ).X
                / sigma_i
            )

            expected_coordinates = axl_i.X + eps_i * s_i + torch.sqrt(2.0 * eps_i) * z
            expected_coordinates = map_relative_coordinates_to_unit_cell(
                expected_coordinates
            )

            torch.testing.assert_close(computed_sample.X, expected_coordinates)

            if pc_generator.atom_type_transition_in_corrector:
                a_i = axl_i.A
                corrected_a_i = computed_sample.A

                difference_mask = corrected_a_i != a_i

                # Test that the changes are from MASK to not-MASK
                assert (
                    a_i[difference_mask] == pc_generator.masked_atom_type_index
                ).all()
                assert (
                    corrected_a_i[difference_mask]
                    != pc_generator.masked_atom_type_index
                ).all()

                if pc_generator.one_atom_type_transition_per_step:
                    # Test that there is at most one change
                    assert torch.all(difference_mask.sum(dim=-1) <= 1.0)

            else:
                assert torch.all(computed_sample.A == axl_i.A)
