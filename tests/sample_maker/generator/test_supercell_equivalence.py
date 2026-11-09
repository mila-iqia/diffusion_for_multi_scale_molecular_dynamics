"""The same local environment must give the same atomic displacement, whatever the size of the cell.

A perturbed Si8 diamond cell and its periodic supercell describe exactly the same infinite crystal, so every
supercell atom has the same local environment as its counterpart in the Si8 cell. The tests check, in two steps,
that the counterparts are displaced identically:
    1. the score network gives the same outputs for the counterparts;
    2. the same network outputs give the same cartesian displacements in the predictor and corrector steps.
Only the cartesian EGNN is tested: the torus EGNN embeds the relative coordinates, so the same cartesian environment
is a different environment on the torus of a larger cell.
"""

import itertools

import einops
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, NOISE, NOISY_AXL_COMPOSITION, NUMBER_OF_ATOMS, TIME)
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.generator.adaptive_corrector import \
    AdaptiveCorrectorGenerator
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.generator.langevin_generator import \
    LangevinGenerator
from diffusion_for_multi_scale_molecular_dynamics.sample_maker.generator.predictor_corrector_axl_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.score_network import \
    ScoreNetworkParameters
from diffusion_for_multi_scale_molecular_dynamics.score_network.cartesian_egnn_score_network import (
    CartesianEGNNScoreNetwork, CartesianEGNNScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from diffusion_for_multi_scale_molecular_dynamics.utils.d3pm_utils import \
    class_index_to_onehot
from tests.sample_maker.generator.conftest import FakeAXLNetwork

SI_LATTICE_CONSTANT = 5.43
DIAMOND_RELATIVE_COORDINATES = [
    [0.00, 0.00, 0.00],
    [0.00, 0.50, 0.50],
    [0.50, 0.00, 0.50],
    [0.50, 0.50, 0.00],
    [0.25, 0.25, 0.25],
    [0.25, 0.75, 0.75],
    [0.75, 0.25, 0.75],
    [0.75, 0.75, 0.25],
]
SPATIAL_DIMENSION = 3
NUM_ATOM_TYPES = 1


def get_lattice_parameters(cell_lengths: torch.Tensor) -> torch.Tensor:
    """Orthogonal-cell lattice parameters [L_x, L_y, L_z, 0, 0, 0]."""
    return torch.cat([cell_lengths, torch.zeros_like(cell_lengths)], dim=1)


def tile(per_atom_values: torch.Tensor, number_of_images: int) -> torch.Tensor:
    """Repeat per-atom values for every image of the primitive cell, image-major like get_supercell_coordinates."""
    return einops.repeat(
        per_atom_values, "batch natoms ... -> batch (images natoms) ...", images=number_of_images
    )


def get_supercell_coordinates(relative_coordinates: torch.Tensor, repetitions: torch.Tensor) -> torch.Tensor:
    """Relative coordinates of the supercell, image-major: supercell atom (image * natoms + i) is a copy of atom i."""
    images = torch.tensor(
        list(itertools.product(*[range(int(n)) for n in repetitions])), dtype=relative_coordinates.dtype
    )
    supercell_coordinates = (relative_coordinates[:, None, :, :] + images[None, :, None, :]) / repetitions
    return einops.rearrange(supercell_coordinates, "batch images natoms d -> batch (images natoms) d")


def get_cartesian_displacements(
    relative_coordinates_before: torch.Tensor, relative_coordinates_after: torch.Tensor, cell_lengths: torch.Tensor
) -> torch.Tensor:
    """Minimum-image cartesian displacement of every atom."""
    relative_displacements = relative_coordinates_after - relative_coordinates_before
    relative_displacements = relative_displacements - torch.round(relative_displacements)
    return relative_displacements * cell_lengths[:, None, :]


@pytest.fixture(autouse=True)
def set_random_seed():
    torch.manual_seed(2345)


@pytest.fixture()
def batch_size():
    return 2


@pytest.fixture()
def number_of_primitive_atoms():
    return len(DIAMOND_RELATIVE_COORDINATES)


@pytest.fixture(params=[(2, 2, 2), (3, 3, 3), (2, 1, 1)], ids=["2x2x2", "3x3x3", "2x1x1"])
def repetitions(request):
    return torch.tensor(request.param, dtype=torch.get_default_dtype())


@pytest.fixture()
def number_of_images(repetitions):
    return int(repetitions.prod())


@pytest.fixture()
def primitive_cell_lengths(batch_size):
    return torch.full((batch_size, SPATIAL_DIMENSION), SI_LATTICE_CONSTANT)


@pytest.fixture()
def supercell_lengths(primitive_cell_lengths, repetitions):
    return primitive_cell_lengths * repetitions


@pytest.fixture()
def primitive_relative_coordinates(batch_size):
    diamond = torch.tensor(DIAMOND_RELATIVE_COORDINATES).repeat(batch_size, 1, 1)
    return map_relative_coordinates_to_unit_cell(diamond + 0.03 * torch.randn_like(diamond))


@pytest.fixture()
def supercell_relative_coordinates(primitive_relative_coordinates, repetitions):
    return get_supercell_coordinates(primitive_relative_coordinates, repetitions)


class TestScoreNetworkOutputs:
    """Step 1: the score network gives the same outputs for atoms with the same local environment.

    Double precision, so that the different summation orders over the edges of the two cells do not matter.
    """

    @pytest.fixture(scope="class", autouse=True)
    def set_default_dtype(self):
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        yield
        torch.set_default_dtype(default_dtype)

    @pytest.fixture()
    def sigma(self):
        return 0.2

    @pytest.fixture()
    def time(self):
        return 0.3

    @pytest.fixture(params=[True, False], ids=["smooth_cutoff", "hard_cutoff"])
    def smooth_cutoff(self, request):
        return request.param

    @pytest.fixture(params=[True, False], ids=["rebuild_edges", "edges_once"])
    def rebuild_edges_every_layer(self, request):
        return request.param

    @pytest.fixture()
    def score_network(self, smooth_cutoff, rebuild_edges_every_layer):
        score_network_parameters = CartesianEGNNScoreNetworkParameters(
            radial_cutoff=5.0,
            num_atom_types=NUM_ATOM_TYPES,
            smooth_cutoff=smooth_cutoff,
            rebuild_edges_every_layer=rebuild_edges_every_layer,
        )
        score_network = CartesianEGNNScoreNetwork(score_network_parameters).double()
        score_network.eval()
        return score_network

    def compute_outputs(self, score_network, relative_coordinates, cell_lengths, sigma, time):
        """Return the score network outputs and the EGNN cartesian displacements, per atom."""
        batch_size, number_of_atoms, _ = relative_coordinates.shape
        batch = {
            NOISY_AXL_COMPOSITION: AXL(
                A=torch.zeros(batch_size, number_of_atoms, dtype=torch.long),
                X=relative_coordinates,
                L=get_lattice_parameters(cell_lengths),
            ),
            TIME: torch.full((batch_size, 1), time),
            NOISE: torch.full((batch_size, 1), sigma),
            NUMBER_OF_ATOMS: torch.full((batch_size,), number_of_atoms, dtype=torch.long),
        }

        egnn_input_positions = []
        egnn_output_positions = []
        pre_hook = score_network.egnn.register_forward_pre_hook(
            lambda module, args, kwargs: egnn_input_positions.append(kwargs["x"].clone()), with_kwargs=True
        )
        hook = score_network.egnn.register_forward_hook(
            lambda module, args, output: egnn_output_positions.append(output.X.clone())
        )
        with torch.no_grad():
            model_outputs = score_network(batch, conditional=False)
        pre_hook.remove()
        hook.remove()

        egnn_displacements = einops.rearrange(
            egnn_output_positions[0] - egnn_input_positions[0],
            "(batch natoms) d -> batch natoms d",
            batch=batch_size,
        )
        return model_outputs, egnn_displacements

    @pytest.fixture()
    def primitive_outputs(self, score_network, primitive_relative_coordinates, primitive_cell_lengths, sigma, time):
        return self.compute_outputs(score_network, primitive_relative_coordinates, primitive_cell_lengths, sigma, time)

    @pytest.fixture()
    def supercell_outputs(self, score_network, supercell_relative_coordinates, supercell_lengths, sigma, time):
        return self.compute_outputs(score_network, supercell_relative_coordinates, supercell_lengths, sigma, time)

    def test_egnn_cartesian_displacements_match(self, primitive_outputs, supercell_outputs, number_of_images):
        _, primitive_displacements = primitive_outputs
        _, supercell_displacements = supercell_outputs
        torch.testing.assert_close(supercell_displacements, tile(primitive_displacements, number_of_images))

    def test_atom_type_logits_match(self, primitive_outputs, supercell_outputs, number_of_images):
        primitive_model_outputs, _ = primitive_outputs
        supercell_model_outputs, _ = supercell_outputs
        torch.testing.assert_close(supercell_model_outputs.A, tile(primitive_model_outputs.A, number_of_images))

    def test_sigma_normalized_scores_match(self, primitive_outputs, supercell_outputs, number_of_images):
        primitive_model_outputs, _ = primitive_outputs
        supercell_model_outputs, _ = supercell_outputs
        torch.testing.assert_close(supercell_model_outputs.X, tile(primitive_model_outputs.X, number_of_images))


class GeneratorDisplacementCalculator:
    """Fixtures and helpers to compute the cartesian displacements of a generator step in the primitive cell and in
    the supercell, with the network outputs and the gaussian noise replaced by the given tensors."""

    @pytest.fixture()
    def total_time_steps(self):
        return 10

    @pytest.fixture()
    def noise_parameters(self, total_time_steps):
        return NoiseParameters(total_time_steps=total_time_steps, sigma_min_cart=0.01, sigma_max_cart=0.5)

    @pytest.fixture()
    def primitive_scores(self, batch_size, number_of_primitive_atoms):
        return 0.5 * torch.randn(batch_size, number_of_primitive_atoms, SPATIAL_DIMENSION)

    @pytest.fixture()
    def primitive_gaussian_noise(self, batch_size, number_of_primitive_atoms):
        return 0.5 * torch.randn(batch_size, number_of_primitive_atoms, SPATIAL_DIMENSION)

    def make_generator(
        self, mocker, generator_class, noise_parameters, batch_size, relative_coordinates, scores, gaussian_noise
    ):
        """Generator whose network outputs and gaussian noise are replaced by the given tensors."""
        number_of_atoms = relative_coordinates.shape[1]
        sampling_parameters = PredictorCorrectorSamplingParameters(
            number_of_atoms=number_of_atoms,
            number_of_samples=batch_size,
            spatial_dimension=SPATIAL_DIMENSION,
            num_atom_types=NUM_ATOM_TYPES,
            number_of_corrector_steps=1,
        )
        axl_network = FakeAXLNetwork(
            ScoreNetworkParameters(
                architecture="dummy", spatial_dimension=SPATIAL_DIMENSION, num_atom_types=NUM_ATOM_TYPES
            )
        )
        generator = generator_class(
            noise_parameters=noise_parameters, sampling_parameters=sampling_parameters, axl_network=axl_network
        )

        def get_fixed_model_predictions(composition, time, sigma_noise, cartesian_forces):
            return AXL(
                A=class_index_to_onehot(composition.A, num_classes=NUM_ATOM_TYPES + 1).to(scores),
                X=scores,
                L=torch.zeros_like(composition.L),
            )

        mocker.patch.object(generator, "_get_model_predictions", side_effect=get_fixed_model_predictions)
        mocker.patch.object(generator, "_draw_coordinates_gaussian_sample", return_value=gaussian_noise)
        mocker.patch.object(
            generator, "_draw_lattice_gaussian_sample", return_value=torch.zeros(batch_size, 2 * SPATIAL_DIMENSION)
        )
        return generator

    def compute_displacements(
        self,
        mocker,
        step_name,
        index_i,
        generator_class,
        noise_parameters,
        relative_coordinates,
        cell_lengths,
        scores,
        gaussian_noise,
    ):
        batch_size, number_of_atoms, _ = relative_coordinates.shape
        generator = self.make_generator(
            mocker, generator_class, noise_parameters, batch_size, relative_coordinates, scores, gaussian_noise
        )
        composition = AXL(
            A=torch.zeros(batch_size, number_of_atoms, dtype=torch.long),
            X=relative_coordinates,
            L=get_lattice_parameters(cell_lengths),
        )
        forces = torch.zeros_like(relative_coordinates)
        updated_composition = getattr(generator, step_name)(composition, index_i, forces)
        return get_cartesian_displacements(relative_coordinates, updated_composition.X, cell_lengths)

    def compute_primitive_and_supercell_displacements(
        self,
        mocker,
        step_name,
        index_i,
        generator_class,
        noise_parameters,
        primitive_relative_coordinates,
        supercell_relative_coordinates,
        primitive_cell_lengths,
        supercell_lengths,
        primitive_scores,
        primitive_gaussian_noise,
        number_of_images,
    ):
        """Return the primitive cell displacements tiled to the supercell, and the supercell displacements."""
        primitive_displacements = self.compute_displacements(
            mocker,
            step_name,
            index_i,
            generator_class,
            noise_parameters,
            primitive_relative_coordinates,
            primitive_cell_lengths,
            primitive_scores,
            primitive_gaussian_noise,
        )
        supercell_displacements = self.compute_displacements(
            mocker,
            step_name,
            index_i,
            generator_class,
            noise_parameters,
            supercell_relative_coordinates,
            supercell_lengths,
            tile(primitive_scores, number_of_images),
            tile(primitive_gaussian_noise, number_of_images),
        )
        assert primitive_displacements.abs().max() < 0.25 * SI_LATTICE_CONSTANT
        return tile(primitive_displacements, number_of_images), supercell_displacements


class TestGeneratorDisplacements(GeneratorDisplacementCalculator):
    """Step 2: the same network outputs give the same cartesian displacements in both cells."""

    @pytest.fixture(params=[LangevinGenerator, AdaptiveCorrectorGenerator], ids=["langevin", "adaptive_corrector"])
    def generator_class(self, request):
        return request.param

    @pytest.mark.parametrize(
        "step_name, index_i",
        [
            ("predictor_step", 1),
            ("predictor_step", 5),
            ("predictor_step", 10),
            ("corrector_step", 0),
            ("corrector_step", 5),
            ("corrector_step", 9),
        ],
    )
    def test_cartesian_displacements_match(
        self,
        mocker,
        step_name,
        index_i,
        generator_class,
        noise_parameters,
        primitive_relative_coordinates,
        supercell_relative_coordinates,
        primitive_cell_lengths,
        supercell_lengths,
        primitive_scores,
        primitive_gaussian_noise,
        number_of_images,
    ):
        tiled_primitive_displacements, supercell_displacements = self.compute_primitive_and_supercell_displacements(
            mocker,
            step_name,
            index_i,
            generator_class,
            noise_parameters,
            primitive_relative_coordinates,
            supercell_relative_coordinates,
            primitive_cell_lengths,
            supercell_lengths,
            primitive_scores,
            primitive_gaussian_noise,
            number_of_images,
        )
        torch.testing.assert_close(supercell_displacements, tiled_primitive_displacements)


class TestDriftAndNoiseScaling(GeneratorDisplacementCalculator):
    """The drift and the noise of the Langevin predictor and corrector steps, taken separately, are the same in every
    cell size.

    A step size defined in reduced units would instead make the corrector drift grow as L^2 and its noise as L.
    """

    @pytest.mark.parametrize("step_name, index_i", [("predictor_step", 5), ("corrector_step", 5)])
    @pytest.mark.parametrize("component", ["drift", "noise"])
    def test_component_is_cell_size_independent(
        self,
        mocker,
        step_name,
        index_i,
        component,
        noise_parameters,
        primitive_relative_coordinates,
        supercell_relative_coordinates,
        primitive_cell_lengths,
        supercell_lengths,
        primitive_scores,
        primitive_gaussian_noise,
        number_of_images,
    ):
        zeros = torch.zeros_like(primitive_scores)
        scores = primitive_scores if component == "drift" else zeros
        gaussian_noise = primitive_gaussian_noise if component == "noise" else zeros
        tiled_primitive_displacements, supercell_displacements = self.compute_primitive_and_supercell_displacements(
            mocker,
            step_name,
            index_i,
            LangevinGenerator,
            noise_parameters,
            primitive_relative_coordinates,
            supercell_relative_coordinates,
            primitive_cell_lengths,
            supercell_lengths,
            scores,
            gaussian_noise,
            number_of_images,
        )
        assert tiled_primitive_displacements.abs().max() > 1.0e-4
        torch.testing.assert_close(supercell_displacements, tiled_primitive_displacements)
