import einops
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.noisers.atom_types_noiser import (
    AtomTypesNoiser,
)


@pytest.mark.parametrize("shape", [(10, 1), (4, 5, 3), (2, 2, 2, 2)])
class TestNoisyAtomTypesSampler:

    @pytest.fixture(scope="class", autouse=True)
    def set_random_seed(self):
        torch.manual_seed(23423)

    @pytest.fixture()
    def num_atom_types(self):
        return 4

    @pytest.fixture()
    def real_atom_types(self, shape, num_atom_types):
        return torch.randint(0, num_atom_types, shape).long()

    @pytest.fixture()
    def real_atom_types_one_hot(self, real_atom_types, num_atom_types):
        return torch.nn.functional.one_hot(real_atom_types, num_classes=num_atom_types)

    @pytest.fixture()
    def q_bar_matrices(self, shape, num_atom_types):
        return torch.rand(shape + (num_atom_types, num_atom_types))

    @pytest.fixture()
    def computed_noisy_atom_types(self, real_atom_types_one_hot, q_bar_matrices):
        return AtomTypesNoiser.get_noisy_atom_types_sample(
            real_atom_types_one_hot, q_bar_matrices
        )

    @pytest.fixture()
    def fake_uniform_noise(self, shape, num_atom_types):
        return torch.rand(shape + (num_atom_types,))

    def test_shape(self, computed_noisy_atom_types, shape):
        assert computed_noisy_atom_types.shape == shape

    def test_range(self, computed_noisy_atom_types, num_atom_types):
        assert torch.all(computed_noisy_atom_types >= 0)
        assert torch.all(computed_noisy_atom_types < num_atom_types)

    def test_get_noisy_relative_coordinates_sample(
        self, mocker, real_atom_types_one_hot, q_bar_matrices, fake_uniform_noise
    ):
        mocker.patch.object(
            AtomTypesNoiser,
            "_get_uniform_noise",
            return_value=fake_uniform_noise,
        )
        computed_samples = AtomTypesNoiser.get_noisy_atom_types_sample(
            real_atom_types_one_hot, q_bar_matrices
        )

        flat_q_matrices = q_bar_matrices.flatten(end_dim=-3)
        flat_atom_types = real_atom_types_one_hot.flatten(end_dim=-2).float()
        flat_computed_samples = computed_samples.flatten()
        flat_fake_noise = fake_uniform_noise.flatten(end_dim=-2)

        for qmat, x0, computed_sample, epsilon in zip(
            flat_q_matrices,
            flat_atom_types,
            flat_computed_samples,
            flat_fake_noise,
        ):
            post_q = einops.einsum(x0, qmat, "... j, ... j i -> ... i")
            expected_sample = torch.log(post_q) - torch.log(-torch.log(epsilon))
            expected_sample = torch.argmax(expected_sample, dim=-1)

            assert torch.all(computed_sample == expected_sample)
