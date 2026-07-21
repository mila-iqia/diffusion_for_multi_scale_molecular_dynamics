import einops
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.langevin_generator import \
    LangevinGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks import \
    ScoreNetworkParameters
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from src.diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_scheduler import \
    NoiseScheduler
from tests.generators.conftest import BaseTestGenerator, FakeAXLNetwork


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
    def sigma_min_cart(self):
        return 1.5

    @pytest.fixture()
    def sigma_max_cart(self):
        return 5.

    @pytest.fixture()
    def noise_parameters(self, total_time_steps, sigma_min_cart, sigma_max_cart):
        noise_parameters = NoiseParameters(
            total_time_steps=total_time_steps,
            time_delta=0.1,
            sigma_min_cart=sigma_min_cart,
            sigma_max_cart=sigma_max_cart,
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
        number_of_samples,
        number_of_corrector_steps,
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
        self,
        pc_generator,
        device,
        number_of_samples,
    ):
        # Just a smoke test that we can sample without crashing.
        pc_generator.sample(number_of_samples, device)

    @pytest.fixture()
    def cell_dimensions(self, spatial_dimension):
        # Rectangular (non-cubic) cell: each direction has a distinct length.
        generator = torch.Generator().manual_seed(23)
        cell_dimensions = torch.rand(3, generator=generator).tolist()
        return cell_dimensions[:spatial_dimension]

    @pytest.fixture()
    def axl_i(
        self,
        cell_dimensions,
        number_of_samples,
        number_of_atoms,
        spatial_dimension,
        num_atomic_classes,
        device,
    ):
        num_lattice_params = int(spatial_dimension * (spatial_dimension + 1) / 2)
        L = torch.zeros(number_of_samples, num_lattice_params).to(device)
        L[:, :spatial_dimension] = torch.tensor(cell_dimensions, dtype=torch.float32, device=device)
        return AXL(
            A=torch.randint(
                0, num_atomic_classes, (number_of_samples, number_of_atoms)
            ).to(device),
            X=map_relative_coordinates_to_unit_cell(
                torch.rand(number_of_samples, number_of_atoms, spatial_dimension)
            ).to(device),
            L=L,
        )

    def test_predictor_step_relative_coordinates_and_lattice(
        self,
        mocker,
        pc_generator,
        noise,
        axl_i,
        total_time_steps,
        number_of_samples,
        spatial_dimension,
        cell_dimensions,
    ):
        list_sigma = noise.sigma  # Cartesian sigmas
        list_time = noise.time
        forces = torch.zeros_like(axl_i.X)
        # Diagonal lattice elements per direction: shape [number_of_samples, spatial_dimension].
        lattice_diagonals = axl_i.L[:, :spatial_dimension]

        z_coordinates = pc_generator._draw_coordinates_gaussian_sample(
            number_of_samples
        ).to(axl_i.X)
        mocker.patch.object(
            pc_generator,
            "_draw_coordinates_gaussian_sample",
            return_value=z_coordinates,
        )

        z_lattice = pc_generator._draw_lattice_gaussian_sample(number_of_samples).to(
            axl_i.L
        )
        mocker.patch.object(
            pc_generator, "_draw_lattice_gaussian_sample", return_value=z_lattice
        )

        for index_i in range(1, total_time_steps + 1):
            computed_sample = pc_generator.predictor_step(axl_i, index_i, forces)

            idx = index_i - 1
            sigma_cart_i = list_sigma[idx]
            t_i = list_time[idx]
            g_cart_i = noise.g[idx]
            g2_cart_i = noise.g_squared[idx]

            model_predictions = pc_generator._get_model_predictions(
                axl_i, t_i, sigma_cart_i, forces
            )

            dx_cart = g2_cart_i * model_predictions.X / sigma_cart_i + g_cart_i * z_coordinates
            expected_coordinates = map_relative_coordinates_to_unit_cell(
                axl_i.X + dx_cart / lattice_diagonals[:, None, :]
            )

            torch.testing.assert_close(computed_sample.X, expected_coordinates)

            # TODO: Unsure if there should be a dependence on the number of atoms.
            number_of_atoms = axl_i.X.shape[1]
            sigma_n_i = sigma_cart_i / (number_of_atoms ** (1.0 / spatial_dimension))
            g2_n_i = g2_cart_i / (number_of_atoms ** (2.0 / spatial_dimension))
            g_n_i = g_cart_i / (number_of_atoms ** (1.0 / spatial_dimension))
            expected_lattice = axl_i.L + g2_n_i * model_predictions.L / sigma_n_i + g_n_i * z_lattice

            torch.testing.assert_close(computed_sample.L, expected_lattice)

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

    @pytest.fixture()
    def repaint_is_used(self):
        return False

    def test_predictor_step_atom_types(
        self,
        mocker,
        pc_generator,
        cell_dimensions,
        total_time_steps,
        number_of_samples,
        number_of_atoms,
        num_atomic_classes,
        spatial_dimension,
        repaint_is_used,
        device,
    ):
        random_x = map_relative_coordinates_to_unit_cell(
            torch.rand(number_of_samples, number_of_atoms, spatial_dimension)
        ).to(device)

        forces = torch.zeros_like(random_x)

        num_lattice_params = int(spatial_dimension * (spatial_dimension + 1) / 2)
        random_l = torch.zeros(number_of_samples, num_lattice_params).to(device)
        random_l[:, :spatial_dimension] = torch.tensor(cell_dimensions, dtype=torch.float32, device=device)

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
            fake_model_predictions = AXL(
                A=logits, X=torch.zeros_like(random_x), L=torch.zeros_like(random_l)
            )
            mocker.patch.object(
                pc_generator,
                "_get_model_predictions",
                return_value=fake_model_predictions,
            )

            axl_i = pc_generator.predictor_step(axl_ip1, idx + 1, forces)

            this_is_last_time_step = idx == 0
            a_i = axl_i.A
            a_ip1 = axl_ip1.A

            difference_mask = a_ip1 != a_i

            if not repaint_is_used:
                # Test that the changes are from MASK to not-MASK
                # This is not applicable if atom types are repainted.
                assert (a_ip1[difference_mask] == pc_generator.masked_atom_type_index).all()
                assert (a_i[difference_mask] != pc_generator.masked_atom_type_index).all()

            one_atom_type_transition_per_step = (
                pc_generator.one_atom_type_transition_per_step
                and not this_is_last_time_step
            )

            if one_atom_type_transition_per_step and not repaint_is_used:
                # Test that there is at most one change. This is not applicable if
                # atom types are repainted.
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
        num_atomic_classes,
        number_of_atoms,
        spatial_dimension,
    ):

        sampler = NoiseScheduler(noise_parameters, num_classes=num_atomic_classes)
        noise, _ = sampler.get_all_sampling_parameters()
        sigma_min_cart = noise_parameters.sigma_min_cart
        epsilon = noise_parameters.corrector_step_epsilon
        list_sigma = noise.sigma  # Cartesian sigmas
        list_time = noise.time
        sigma_1_cart = list_sigma[0]
        forces = torch.zeros_like(axl_i.X)
        lattice_diagonals = axl_i.L[:, :spatial_dimension]

        z_coordinates = pc_generator._draw_coordinates_gaussian_sample(
            number_of_samples
        ).to(axl_i.X)
        mocker.patch.object(
            pc_generator,
            "_draw_coordinates_gaussian_sample",
            return_value=z_coordinates,
        )

        z_lattice = pc_generator._draw_lattice_gaussian_sample(number_of_samples).to(
            axl_i.L
        )
        mocker.patch.object(
            pc_generator, "_draw_lattice_gaussian_sample", return_value=z_lattice
        )

        for index_i in range(0, total_time_steps):
            computed_sample = pc_generator.corrector_step(axl_i, index_i, forces)

            if index_i == 0:
                sigma_cart_i = sigma_min_cart
                t_i = 0.0
            else:
                sigma_cart_i = list_sigma[index_i - 1]
                t_i = list_time[index_i - 1]

            eps_i = 0.5 * epsilon * sigma_cart_i**2 / sigma_1_cart**2

            model_predictions = pc_generator._get_model_predictions(
                axl_i, t_i, sigma_cart_i, forces
            )

            # test coordinates
            expected_coordinates = map_relative_coordinates_to_unit_cell(
                axl_i.X
                + eps_i / lattice_diagonals[:, None, :] * model_predictions.X / sigma_cart_i
                + torch.sqrt(2.0 * eps_i) / lattice_diagonals[:, None, :] * z_coordinates
            )

            torch.testing.assert_close(computed_sample.X, expected_coordinates)

            # TODO: Unsure if there should be a dependence on the number of atoms.
            sigma_n_i_corrector = sigma_cart_i / (number_of_atoms ** (1.0 / spatial_dimension))
            expected_lattice = (
                axl_i.L + eps_i * model_predictions.L / sigma_n_i_corrector + torch.sqrt(2.0 * eps_i) * z_lattice
            )

            torch.testing.assert_close(computed_sample.L, expected_lattice)

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


class TestPredictorStepDenoisingDirection:
    """Tests that the deterministic predictor step correctly inverts the training score normalization."""

    @pytest.fixture(scope="class", autouse=True)
    def set_random_seed(self):
        torch.manual_seed(12345)

    @pytest.fixture()
    def batch_size(self):
        return 4

    @pytest.fixture()
    def number_of_atoms(self):
        return 3

    @pytest.fixture(params=[2, 3])
    def spatial_dimension(self, request):
        return request.param

    @pytest.fixture()
    def num_atom_types(self):
        return 2

    @pytest.fixture()
    def generator(self, spatial_dimension, num_atom_types):
        noise_parameters = NoiseParameters(total_time_steps=5, sigma_min_cart=0.01, sigma_max_cart=1.0)
        sampling_parameters = PredictorCorrectorSamplingParameters(
            number_of_atoms=3,
            number_of_samples=4,
            spatial_dimension=spatial_dimension,
            num_atom_types=num_atom_types,
            number_of_corrector_steps=1,
        )
        axl_network = FakeAXLNetwork(
            ScoreNetworkParameters(
                architecture="dummy",
                spatial_dimension=spatial_dimension,
                num_atom_types=num_atom_types,
            )
        )
        return LangevinGenerator(
            noise_parameters=noise_parameters,
            sampling_parameters=sampling_parameters,
            axl_network=axl_network,
        )

    @pytest.mark.parametrize("sigma_rel", [0.05, 0.15, 0.3])
    def test_predictor_step_denoises_gaussian_samples(
        self,
        generator,
        sigma_rel,
        batch_size,
        number_of_atoms,
        spatial_dimension,
    ):
        """Deterministic predictor step with perfect Gaussian score moves x_t strictly closer to x_0.

        The training convention is sigma_normalized_scores = sigma_rel * score_rel = -z_noise for Gaussian noising
        x_t = x_0 + sigma_rel * z_noise. With z=0 (deterministic step) and this perfect score as model output,
        dx_rel = -g2_rel/sigma_rel^2 * (x_t - x_0), which contracts the displacement toward x_0 for any sigma_rel
        satisfying g2_rel < sigma_rel^2.
        """
        # g_rel small enough that g2_rel < sigma_rel^2 for all tested sigma_rel, so no overshoot.
        g_rel = 0.02
        g2_rel = g_rel ** 2
        L_diag = 10.0

        # Scale z_noise so x_t stays well inside (0, 1) for all tested sigma_rel values (max displacement = 0.03).
        z_noise = torch.randn(batch_size, number_of_atoms, spatial_dimension)
        z_noise = z_noise / (z_noise.abs().max() + 1e-8) * 0.1

        x_0 = 0.3 + 0.4 * torch.rand(batch_size, number_of_atoms, spatial_dimension)
        x_t = x_0 + sigma_rel * z_noise

        sigma_normalized_scores = -z_noise  # perfect model: sigma_rel * score_rel = -z_noise

        z_zero = torch.zeros(batch_size, number_of_atoms, spatial_dimension)
        lattice_diagonals = L_diag * torch.ones(batch_size, spatial_dimension)
        sigma_cart = torch.tensor(sigma_rel * L_diag)
        g2_cart = torch.tensor(g2_rel * L_diag ** 2)
        g_cart = torch.tensor(g_rel * L_diag)

        x_updated = generator._relative_coordinates_update_predictor_step(
            x_t, sigma_normalized_scores, sigma_cart, g2_cart, g_cart, lattice_diagonals, z_zero
        )

        distance_before = (x_t - x_0).norm(dim=-1)
        distance_after = (x_updated - x_0).norm(dim=-1)
        assert (distance_after <= distance_before).all()


class TestCellSizeIndependence:
    """Tests that coordinate updates produce cell-size-independent Cartesian displacements."""

    @pytest.fixture(scope="class", autouse=True)
    def set_random_seed(self):
        torch.manual_seed(54321)

    @pytest.fixture()
    def batch_size(self):
        return 4

    @pytest.fixture()
    def number_of_atoms(self):
        return 3

    @pytest.fixture(params=[2, 3])
    def spatial_dimension(self, request):
        return request.param

    @pytest.fixture()
    def num_atom_types(self):
        return 2

    @pytest.fixture()
    def generator(self, spatial_dimension, num_atom_types):
        noise_parameters = NoiseParameters(total_time_steps=5, sigma_min_cart=0.01, sigma_max_cart=1.0)
        sampling_parameters = PredictorCorrectorSamplingParameters(
            number_of_atoms=3,
            number_of_samples=4,
            spatial_dimension=spatial_dimension,
            num_atom_types=num_atom_types,
            number_of_corrector_steps=1,
        )
        axl_network = FakeAXLNetwork(
            ScoreNetworkParameters(
                architecture="dummy",
                spatial_dimension=spatial_dimension,
                num_atom_types=num_atom_types,
            )
        )
        return LangevinGenerator(
            noise_parameters=noise_parameters,
            sampling_parameters=sampling_parameters,
            axl_network=axl_network,
        )

    @pytest.fixture()
    def sigma_rel(self):
        return 0.1

    @pytest.fixture()
    def relative_coordinates(self, batch_size, number_of_atoms, spatial_dimension):
        return 0.25 + 0.5 * torch.rand(batch_size, number_of_atoms, spatial_dimension)

    @pytest.fixture()
    def sigma_normalized_scores(self, batch_size, number_of_atoms, spatial_dimension):
        return torch.randn(batch_size, number_of_atoms, spatial_dimension)

    @pytest.fixture()
    def gaussian_noise_z(self, batch_size, number_of_atoms, spatial_dimension):
        return torch.randn(batch_size, number_of_atoms, spatial_dimension)

    def test_predictor_step_is_cell_independent(
        self,
        generator,
        relative_coordinates,
        sigma_normalized_scores,
        gaussian_noise_z,
        sigma_rel,
        batch_size,
        spatial_dimension,
    ):
        """Predictor dx_rel is identical for L=5 and L=20 when sigma_rel and g_rel are fixed.

        With g2_cart = g2_rel * L^2, g_cart = g_rel * L, sigma_cart = sigma_rel * L, the pre-scaling
        by L^{-1} cancels all L-dependence: dx_rel = g2_rel/sigma_rel * sigma_normalized_scores + g_rel * z.
        """
        g_rel = 0.05
        list_updated_coordinates = []

        for L_diag in [5.0, 20.0]:
            lattice_diagonals = L_diag * torch.ones(batch_size, spatial_dimension)
            sigma_cart = torch.tensor(sigma_rel * L_diag)
            g2_cart = torch.tensor(g_rel ** 2 * L_diag ** 2)
            g_cart = torch.tensor(g_rel * L_diag)

            x_updated = generator._relative_coordinates_update_predictor_step(
                relative_coordinates, sigma_normalized_scores,
                sigma_cart, g2_cart, g_cart, lattice_diagonals, gaussian_noise_z,
            )
            list_updated_coordinates.append(x_updated)

        torch.testing.assert_close(list_updated_coordinates[0], list_updated_coordinates[1])

    def test_corrector_step_is_cell_independent(
        self,
        generator,
        relative_coordinates,
        sigma_normalized_scores,
        gaussian_noise_z,
        batch_size,
        spatial_dimension,
    ):
        """Corrector dx_cart = L * dx_rel is identical for L=5 and L=20 when sigma_cart and eps are fixed.

        sigma_cart is L-independent (fixed in Angstroms). Both score_weight = eps/L and
        noise_weight = sqrt_2eps/L scale as 1/L, so dx_rel scales as 1/L and
        dx_cart = L * dx_rel = eps * score_cart + sqrt_2eps * z is L-independent.
        """
        sigma_cart = torch.tensor(0.5)  # fixed in Angstroms, L-independent
        eps = 0.002
        sqrt_2eps = torch.sqrt(torch.tensor(2.0 * eps))
        list_dx_cart = []

        for L_diag in [5.0, 20.0]:
            lattice_diagonals = L_diag * torch.ones(batch_size, spatial_dimension)

            x_updated = generator._relative_coordinates_update_corrector_step(
                relative_coordinates, sigma_normalized_scores,
                sigma_cart, torch.tensor(eps), sqrt_2eps, lattice_diagonals, gaussian_noise_z,
            )
            dx_cart = L_diag * (x_updated - relative_coordinates)
            list_dx_cart.append(dx_cart)

        torch.testing.assert_close(list_dx_cart[0], list_dx_cart[1])
