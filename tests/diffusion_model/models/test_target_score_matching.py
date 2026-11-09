import ase
import ase.io
import numpy as np
import pytest
import torch
from ase.calculators.singlepoint import SinglePointCalculator
from ase.units import kB

from diffusion_for_multi_scale_molecular_dynamics.diffusion_model.models.target_score_matching import (
    TargetScoreMatchingParameters, clip_forces, get_one_sided_squared_error,
    get_target_score_matching_target, get_thermal_widths_by_atom_type,
    read_bad_frames)

CELL_LENGTH = 5.43


@pytest.fixture()
def valid_arguments():
    return dict(temperature=300.0, thermal_width={"Si": 0.044}, minimum_sigma=0.01, maximum_sigma=0.044)


class TestTargetScoreMatchingParameters:

    def test_beta(self, valid_arguments):
        parameters = TargetScoreMatchingParameters(**valid_arguments)
        assert parameters.beta == pytest.approx(1.0 / (kB * 300.0))
        assert parameters.beta == pytest.approx(38.68, abs=0.01)

    @pytest.mark.parametrize(
        "invalid_arguments",
        [
            dict(temperature=0.0),
            dict(thermal_width={"Si": 0.0}),
            dict(minimum_sigma=0.0),
            dict(minimum_sigma=0.1, maximum_sigma=0.05),
            dict(bad_frames_path="bad_frames.traj"),
        ],
    )
    def test_invalid_parameters_are_rejected(self, valid_arguments, invalid_arguments):
        with pytest.raises(AssertionError):
            TargetScoreMatchingParameters(**{**valid_arguments, **invalid_arguments})


class TestThermalWidthsByAtomType:

    def test_thermal_widths_follow_sorted_element_ids(self):
        thermal_widths = get_thermal_widths_by_atom_type({"Si": 0.04, "Ge": 0.06}, ["Si", "Ge"])
        torch.testing.assert_close(thermal_widths, torch.tensor([0.06, 0.04]))

    def test_missing_thermal_width_raises(self):
        with pytest.raises(AssertionError):
            get_thermal_widths_by_atom_type({"Si": 0.04}, ["Si", "Ge"])


class TestTargetScoreMatchingTarget:

    @pytest.fixture(autouse=True)
    def set_random_seed(self):
        torch.manual_seed(1234)

    @pytest.fixture()
    def beta(self):
        return 38.68

    def test_low_noise_limit_is_the_force_target(self, beta):
        forces = torch.randn(2, 3, 3, dtype=torch.float64)
        denoising_target = torch.randn(2, 3, 3, dtype=torch.float64)
        sigma = torch.full((2,), 1.0e-5, dtype=torch.float64)
        target = get_target_score_matching_target(denoising_target, forces, torch.zeros(2, 3, dtype=torch.long),
                                                  sigma, torch.tensor([0.05], dtype=torch.float64), beta)
        torch.testing.assert_close(target, sigma[:, None, None] * beta * forces, rtol=1e-6, atol=1e-6)

    def test_high_noise_limit_is_the_denoising_target(self, beta):
        forces = torch.randn(2, 3, 3, dtype=torch.float64)
        denoising_target = torch.randn(2, 3, 3, dtype=torch.float64)
        sigma = torch.full((2,), 1.0e3, dtype=torch.float64)
        target = get_target_score_matching_target(denoising_target, forces, torch.zeros(2, 3, dtype=torch.long),
                                                  sigma, torch.tensor([0.05], dtype=torch.float64), beta)
        torch.testing.assert_close(target, denoising_target, rtol=0.0, atol=1e-3)

    def test_kappa_is_per_atom(self, beta):
        thermal_widths = torch.tensor([0.02, 0.08], dtype=torch.float64)
        atom_types = torch.tensor([[0, 1]])
        sigma = torch.tensor([0.04], dtype=torch.float64)
        forces = torch.randn(1, 2, 3, dtype=torch.float64)
        denoising_target = torch.randn(1, 2, 3, dtype=torch.float64)

        target = get_target_score_matching_target(denoising_target, forces, atom_types, sigma, thermal_widths, beta)

        for atom_index, thermal_width in enumerate([0.02, 0.08]):
            kappa = thermal_width**2 / (thermal_width**2 + 0.04**2)
            expected = (kappa * 0.04 * beta * forces[0, atom_index]
                        + (1.0 - kappa) * denoising_target[0, atom_index])
            torch.testing.assert_close(target[0, atom_index], expected)

    def test_harmonic_thermal_vibrations_give_the_exact_score(self, beta):
        """For x0 = mu + s u (thermal), F = -(x0 - mu) / (beta s^2), and xt = x0 + sigma z, the mixed target is
        exactly the score of the noised harmonic distribution, -sigma (xt - mu) / (s^2 + sigma^2), sample by sample.
        """
        thermal_width = 0.05
        sigma = torch.tensor([0.005, 0.02, 0.05, 0.2, 1.0], dtype=torch.float64)
        mu = torch.rand(5, 4, 3, dtype=torch.float64)
        x0 = mu + thermal_width * torch.randn(5, 4, 3, dtype=torch.float64)
        z = torch.randn(5, 4, 3, dtype=torch.float64)
        xt = x0 + sigma[:, None, None] * z
        forces = -(x0 - mu) / (beta * thermal_width**2)

        target = get_target_score_matching_target(-z, forces, torch.zeros(5, 4, dtype=torch.long), sigma,
                                                  torch.tensor([thermal_width], dtype=torch.float64), beta)

        expected = -sigma[:, None, None] * (xt - mu) / (thermal_width**2 + sigma[:, None, None] ** 2)
        torch.testing.assert_close(target, expected)


class TestClipForces:

    def test_clip_forces(self):
        forces = torch.tensor([[[1.0, 0.0, 0.0], [30.0, 40.0, 0.0], [0.0, 0.0, 0.0]]])
        clipped_forces = clip_forces(forces, maximum_force=5.0)
        expected = torch.tensor([[[1.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]])
        torch.testing.assert_close(clipped_forces, expected)
        assert not torch.isnan(clipped_forces).any()


class TestOneSidedSquaredError:

    @pytest.fixture()
    def target(self):
        return torch.tensor([[[-8.5, 0.0, 0.0]]])

    @pytest.mark.parametrize(
        "prediction, expected_error, expected_gradient",
        [
            ([-10.0, 0.0, 0.0], 0.0, [0.0, 0.0, 0.0]),
            ([-148.0, 0.0, 0.0], 0.0, [0.0, 0.0, 0.0]),
            ([-4.0, 0.0, 0.0], 4.5**2, [9.0, 0.0, 0.0]),
            ([148.0, 0.0, 0.0], 156.5**2, [313.0, 0.0, 0.0]),
        ],
    )
    def test_saturated_atom_along_the_target(self, target, prediction, expected_error, expected_gradient):
        predictions = torch.tensor([[prediction]], requires_grad=True)
        error = get_one_sided_squared_error(predictions, target, torch.tensor([[True]]))
        error.sum().backward()
        torch.testing.assert_close(error, torch.tensor([[expected_error]]))
        torch.testing.assert_close(predictions.grad, torch.tensor([[expected_gradient]]))

    @pytest.mark.parametrize("angle", [0.1, 1.0, 5.0, 10.0, 15.0, 20.0, 30.0])
    @pytest.mark.parametrize("prediction_norm", [6.0, 1.0e6])
    def test_saturated_atom_direction_is_penalized_at_the_target_norm(self, angle, prediction_norm):
        """A prediction stronger than the target costs |target|^2 sin^2(angle), whatever its norm."""
        target = torch.tensor([[[5.0, 0.0, 0.0]]], dtype=torch.float64)
        radians = np.radians(angle)
        predictions = prediction_norm * torch.tensor([[[np.cos(radians), np.sin(radians), 0.0]]], dtype=torch.float64)
        error = get_one_sided_squared_error(predictions, target, torch.tensor([[True]]))
        torch.testing.assert_close(error, torch.tensor([[25.0 * np.sin(radians) ** 2]], dtype=torch.float64))

    def test_saturated_atom_stronger_predictions_in_the_same_direction_have_the_same_error(self):
        """Predictions of norm 10 and 148, both stronger than a target of norm 5, 8 degrees off the target."""
        target = torch.tensor([[[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]], dtype=torch.float64)
        radians = np.radians(8.0)
        direction = torch.tensor([np.cos(radians), np.sin(radians), 0.0], dtype=torch.float64)
        predictions = torch.stack([10.0 * direction, 148.0 * direction]).unsqueeze(0)
        error = get_one_sided_squared_error(predictions, target, torch.tensor([[True, True]]))
        torch.testing.assert_close(error[0, 0], error[0, 1])
        torch.testing.assert_close(error[0, 0], torch.tensor(25.0 * np.sin(radians) ** 2, dtype=torch.float64))

    def test_saturated_atom_weaker_prediction_has_the_plain_squared_error(self):
        torch.manual_seed(0)
        target = torch.randn(2, 4, 3)
        perpendicular_directions = torch.nn.functional.normalize(
            torch.linalg.cross(target, torch.randn(2, 4, 3)), dim=-1
        )
        target_norms = torch.linalg.norm(target, dim=-1, keepdim=True)
        predictions = 0.5 * target + 0.3 * target_norms * perpendicular_directions
        error = get_one_sided_squared_error(predictions, target, torch.ones(2, 4, dtype=torch.bool))
        torch.testing.assert_close(error, ((predictions - target) ** 2).sum(dim=-1))

    @pytest.mark.parametrize(
        "prediction", [[-10.0, 0.0, 0.0], [-148.0, 0.0, 0.0], [-4.0, 0.0, 0.0], [148.0, 0.0, 0.0], [-148.0, 30.0, 0.0]]
    )
    def test_unsaturated_atom_has_the_plain_squared_error(self, target, prediction):
        """The target of an unsaturated atom is exact: a stronger prediction is penalized too."""
        predictions = torch.tensor([[prediction]])
        error = get_one_sided_squared_error(predictions, target, torch.tensor([[False]]))
        torch.testing.assert_close(error, ((predictions - target) ** 2).sum(dim=-1))

    def test_relaxed_atom_must_stay_put(self):
        """An almost relaxed atom has a tiny exact target: a prediction of 0.5 is penalized, not free."""
        target = torch.tensor([[[4.0e-5, 0.0, 0.0]]])
        predictions = torch.tensor([[[0.5, 0.0, 0.0]]])
        error = get_one_sided_squared_error(predictions, target, torch.tensor([[False]]))
        torch.testing.assert_close(error, torch.tensor([[(0.5 - 4.0e-5) ** 2]]))

    def test_saturated_atoms_are_selected_per_atom(self, target):
        predictions = torch.tensor([[[-148.0, 0.0, 0.0], [-148.0, 0.0, 0.0]]])
        targets = target.repeat(1, 2, 1)
        error = get_one_sided_squared_error(predictions, targets, torch.tensor([[True, False]]))
        torch.testing.assert_close(error, torch.tensor([[0.0, 139.5**2]]))


def make_frame(number_of_atoms=4, forces=None, cell=None, with_calculator=True, seed=0):
    rng = np.random.default_rng(seed)
    cell = np.diag([CELL_LENGTH] * 3) if cell is None else cell
    atoms = ase.Atoms(symbols=["Si"] * number_of_atoms, positions=rng.random((number_of_atoms, 3)) * CELL_LENGTH,
                      cell=cell, pbc=True)
    if with_calculator:
        forces = rng.normal(size=(number_of_atoms, 3)) if forces is None else forces
        atoms.calc = SinglePointCalculator(atoms, energy=0.0, forces=forces)
    return atoms


def write_frames(path, frames):
    with ase.io.Trajectory(str(path), "w") as trajectory:
        for atoms in frames:
            trajectory.write(atoms)


class TestReadBadFrames:

    def test_missing_file_gives_none(self, tmp_path):
        assert read_bad_frames(tmp_path / "bad_frames.traj", ["Si"], maximum_force=5.0) is None

    def test_empty_file_gives_none(self, tmp_path):
        bad_frames_path = tmp_path / "bad_frames.traj"
        bad_frames_path.touch()
        assert read_bad_frames(bad_frames_path, ["Si"], maximum_force=5.0) is None

    def test_read_bad_frames(self, tmp_path):
        forces = np.array([[100.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
        frames = [make_frame(forces=forces, seed=seed) for seed in range(3)]
        bad_frames_path = tmp_path / "bad_frames.traj"
        write_frames(bad_frames_path, frames)

        bad_frames = read_bad_frames(bad_frames_path, ["Si"], maximum_force=5.0)

        composition = bad_frames["composition"]
        assert composition.X.shape == (3, 4, 3)
        assert composition.X.is_contiguous()
        assert torch.all((composition.X >= 0.0) & (composition.X < 1.0))
        torch.testing.assert_close(composition.A, torch.zeros(3, 4, dtype=torch.long))
        torch.testing.assert_close(composition.L, torch.tensor([[CELL_LENGTH] * 3 + [0.0] * 3] * 3))
        expected_forces = torch.tensor([[5.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
        torch.testing.assert_close(bad_frames["forces"], expected_forces.repeat(3, 1, 1))
        torch.testing.assert_close(bad_frames["saturated_atoms"], torch.tensor([[True, False, False, False]] * 3))

    @pytest.mark.parametrize(
        "frames",
        [
            [make_frame(with_calculator=False)],
            [make_frame(number_of_atoms=4), make_frame(number_of_atoms=5)],
            [make_frame(cell=np.array([[CELL_LENGTH, 0.0, 0.0], [1.0, CELL_LENGTH, 0.0], [0.0, 0.0, CELL_LENGTH]]))],
        ],
        ids=["no_single_point_calculator", "different_number_of_atoms", "non_orthogonal_cell"],
    )
    def test_invalid_bad_frames_raise(self, tmp_path, frames):
        bad_frames_path = tmp_path / "bad_frames.traj"
        write_frames(bad_frames_path, frames)
        with pytest.raises(AssertionError):
            read_bad_frames(bad_frames_path, ["Si"], maximum_force=5.0)
