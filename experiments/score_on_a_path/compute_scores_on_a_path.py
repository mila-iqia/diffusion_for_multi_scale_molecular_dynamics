"""Compute the score along a trajectory where all atoms are fixed except one, going from it's equilibrium position
to its nearest neighbor."""

import glob
from pathlib import Path
from typing import Dict, List

import torch
from torch.func import jacrev
from tqdm import tqdm

from diffusion_for_multi_scale_molecular_dynamics.models.axl_diffusion_lightning_model import \
    AXLDiffusionLightningModel
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, CARTESIAN_FORCES, NOISE, NOISY_AXL_COMPOSITION, TIME, UNIT_CELL)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_scheduler import (
    Noise, NoiseScheduler)
from diffusion_for_multi_scale_molecular_dynamics.sample_diffusion import \
    get_axl_network

# directory where to save the results
RESULTS_DIR = Path(
    "/home/mila/b/blackbus/diffusion_for_multi_scale_molecular_dynamics/experiments/score_on_a_path/results/"
)

# model checkpoint to analyse
MODEL_CKPT_PATH = Path(
    "/home/mila/b/blackbus/diffusion_for_multi_scale_molecular_dynamics/experiments/doublelinear_sigma/si222/baseline2/"
)

# number of steps in the atom movement
N_SPATIAL_STEPS = 100

# number of time steps and related parameters
N_TIME_STEPS = 200
SIGMA_MIN = 0.0001
SIGMA_MAX = 0.5

# 10.86 for Si222
UNIT_CELL_SIZE = 10.86

# path to a reference sample - ideally, this should contain a sample with very low energy i.e. equilibrium positions
REFERENCE_SAMPLES_DIR = Path(
    "/Users/simonblackburn/projects/courtois2024/experiments/double_linear_debug/samples/baseline_ac/epoch54/"
)
TRAJ_PATH = REFERENCE_SAMPLES_DIR / "trajectories.pt"
ENERGY_PATH = REFERENCE_SAMPLES_DIR / "energies.pt"

# if True, the script will add the jacobians as a [num_atoms * spatial_dimension, num_atoms * spatial] for each frame
# this operation is slower than just getting the sigma-normalized score
GET_JACOBIAN = False

# select an atom that will move towards its closest neighbor
# it will start at its location in the reference sample and end up on the same location as its nearest neighbor.
# for visualisation purpose, we recommend selecting an atom away from the unit cell boundaries
MOVING_ATOM_INDEX = 9

SPATIAL_DIMENSION = 3


def main():
    """Make multiple frames and compute the score for each frame."""
    noise_parameters = NoiseParameters(
        total_time_steps=N_TIME_STEPS,
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
    )
    noise, _ = NoiseScheduler(
        noise_parameters, num_classes=2
    ).get_all_sampling_parameters()

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    # create the AXLs for each spatial steps on the path of "atomic collapse"
    interpolated_axls = make_interpolated_frames()
    get_score_on_path(interpolated_axls, noise, device)


def make_interpolated_frames() -> List[AXL]:
    """Create the different AXLs for each movement step.

    The atom selected as moving in the hyperparameters will start at its initial point and end up on the same position
    as its nearest neighbor. The path is linear.

    Returns:
        interpolated_positions: list of N_SPATIAL_STEPS AXLs, each with the moving atom at a different position.
    """
    with open(TRAJ_PATH, "rb") as f:
        trajectories_data = torch.load(f, map_location="cpu")

    with open(ENERGY_PATH, "rb") as f:
        energy_data = torch.load(f, map_location="cpu")

    min_energy_idx = energy_data.argmin()
    best_configuration_axl = AXL(
        A=trajectories_data["predictor_step"][-1]["composition_i"].A[min_energy_idx],
        X=trajectories_data["predictor_step"][-1]["composition_i"].X[min_energy_idx],
        L=trajectories_data["predictor_step"][-1]["composition_i"].L[min_energy_idx],
    )

    # get the position of the fixed atom
    start_position_moving_atom = best_configuration_axl.X[MOVING_ATOM_INDEX, :]
    # find its nearest neighbor index
    target_atom_index = find_closest_atom(MOVING_ATOM_INDEX, best_configuration_axl.X)
    # get the final position of the moving atom
    final_position_moving_atom = best_configuration_axl.X[target_atom_index, :]

    interpolated_positions = []
    for i in range(N_SPATIAL_STEPS):
        # we need to interpolate on N_SPATIAL_STEPS - 1 to get the starting and ending points with N_SPATIAL_STEPS
        moved_atom_position = get_interpolated_position(
            start_position_moving_atom,
            final_position_moving_atom,
            i,
            N_SPATIAL_STEPS - 1,
        )
        new_x = best_configuration_axl.X.clone()
        new_x[MOVING_ATOM_INDEX] = moved_atom_position
        new_axl = AXL(A=best_configuration_axl.A, X=new_x, L=best_configuration_axl.L)
        interpolated_positions.append(new_axl)

    with open(RESULTS_DIR / "interpolated_positions.pt", "wb") as f:
        torch.save(interpolated_positions, f)

    return interpolated_positions


def find_closest_atom(ref_idx: int, positions_tensor: torch.Tensor) -> int:
    """Find the atom closest to a reference atom.

    We assume the positions_tensor represent reduced coordinates with a cubic unit cell.

    Args:
        ref_idx: index of the reference atom.
        positions_tensor: coordinates of all atoms as a [num_atoms, spatial_dimension] tensor

    Returns:
        index of the atom closest to the reference atom
    """
    ref_positions = positions_tensor[ref_idx]  # should be a [spatial_dimension] tensor
    distance_zero = (positions_tensor - ref_positions) ** 2
    distance_plus = (positions_tensor - ref_positions + 1) ** 2
    distance_minus = (positions_tensor - ref_positions - 1) ** 2
    distances = torch.minimum(distance_zero, distance_plus)
    distances = torch.minimum(distances, distance_minus)
    distances = distances.sum(dim=-1)
    distances[ref_idx] = torch.inf  # do not return ref_idx
    return distances.argmin()


def get_interpolated_position(
    start_position: torch.Tensor,
    end_position: torch.Tensor,
    step_idx: int,
    num_step: int,
) -> torch.Tensor:
    """Interpolate linearly the position of an atom at an intermediate step.

    Args:
        start_position: starting position as a [spatial_dimension] tensor
        end_position: ending position as a [spatial_dimension] tensor
        step_idx: current step index. 0 returns the start_position and num_step returns the end_position
        num_step: number of interpolation steps. Should be greater or equal to 1.

    Returns:
        interpolated position as a [spatial_dimension] tensor.
    """
    assert (
        num_step > 0
    ), "There should be at least 1 spatial step to interpolate a trajectory."
    step_size = step_idx / num_step
    return (1 - step_size) * start_position + step_size * end_position


def get_score_on_path(axls_on_path: List[AXL], noise: Noise, device: torch.device):
    """Compute the sigma-normalized score for each frame in a path.

    The scores are saved in a .pt file in the RESULT_DIR.
    Optionally, compute the jacobians as well.

    Args:
        axls_on_path: list of AXLs along the desired path. AXL are assumed to not have a batch dimension
        noise: noise parameters to get sigma and time for the model
        device: device to use for the model
    """
    checkpoint_path = glob.glob(
        str(MODEL_CKPT_PATH / "output/best_model/best_model*.ckpt"), recursive=True
    )[0]

    # load the model
    axl_network = get_axl_network(checkpoint_path)

    time_idx = noise.time
    sigma = noise.sigma

    all_model_predictions = []

    if GET_JACOBIAN:
        all_jacobians = []

    for frame_idx, composition_i in enumerate(axls_on_path):
        # we will repeat the AXL N_TIME_STEPS times on the batch dimension
        # so we can call the model once for each spatial step, getting the score for all time / sigma steps.
        batch_size = time_idx.shape[0]
        # AXLs have no batch size, we need to add one
        composition_i = AXL(
            A=composition_i.A.unsqueeze(0).repeat(batch_size, 1),
            X=composition_i.X.unsqueeze(0).repeat(batch_size, 1, 1),
            L=composition_i.L.unsqueeze(0).repeat(batch_size, 1, 1),
        )
        with torch.no_grad():
            model_predictions = get_model_predictions(
                axl_network, composition_i, time_idx, sigma, device
            )

        all_model_predictions.append(
            model_predictions.cpu()
        )  # move to cpu to avoid overloading the GPU memory

        if GET_JACOBIAN:
            jacobian_i = get_model_coordinates_jacobian(
                axl_network, composition_i, time_idx, sigma, device
            )
            all_jacobians.append(
                jacobian_i.detach().cpu()
            )  # move to cpu to avoid gpu memory issues

    all_data_dict = dict(
        model_predictions=all_model_predictions,
        trajectories=axls_on_path,
        time=noise.time,
        sigma=noise.sigma,
    )
    if GET_JACOBIAN:
        all_jacobians = torch.stack(all_jacobians, dim=0)
        all_data_dict["jacobians"] = all_jacobians

    with open(RESULTS_DIR / "model_predictions.pt", "wb") as f:
        torch.save(all_data_dict, f)


def make_batch(
    axl_composition: AXL,
    times: torch.Tensor,
    sigmas: torch.Tensor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Create a batch for an AXLScoreNetwork from a given AXL composition, sigma and time.

    Args:
        axl_composition: AXL where to evaluate the model
        times: times where to evaluate the model
        sigmas: sigmas, like time
        device: device where to move the batch

    Returns:
        batch as a dict usable by an AXLScoreNetwork
    """
    batch_size = axl_composition.X.shape[0]
    axl_composition = AXL(
        A=axl_composition.A.to(device),
        X=axl_composition.X.to(device),
        L=axl_composition.L.to(device),
    )
    batch = {
        NOISY_AXL_COMPOSITION: axl_composition,
        NOISE: sigmas.unsqueeze(-1).to(device),  # batchsize, 1
        TIME: times.unsqueeze(-1).to(device),
        UNIT_CELL: torch.eye(1).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        * UNIT_CELL_SIZE,
        CARTESIAN_FORCES: torch.zeros_like(axl_composition.X),
    }
    return batch


def get_model_predictions(
    axl_network: AXLDiffusionLightningModel,
    composition: AXL,
    time: torch.Tensor,
    sigma_noise: torch.Tensor,
    device: torch.Tensor,
) -> AXL:
    """Get the AXL score for a given set of atom types, coordinates, lattice parameters, times and sigmas.

    Args:
        axl_network: network to evaluate
        composition: AXL
        time: time
        sigma_noise: sigma
        device: device to use (gpu, cpu, mps)

    Returns:
        AXL output of the model
    """
    batch = make_batch(composition, time, sigma_noise, device)
    return axl_network(batch, conditional=False)


def get_model_coordinates_jacobian(
    axl_network,
    composition: AXL,
    time: torch.Tensor,
    sigma_noise: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Get the jacobian of an axl-network for the coordinates input

     Args:
        composition : AXL composition with:
            atom types, of shape [number of samples, number_of_atoms]
            relative coordinates, of shape [number_of_samples, number_of_atoms, spatial_dimension]
            lattice vectors, of shape [number_of_samples, spatial_dimension * (spatial_dimension - 1)]  # TODO check
        time : time at which to evaluate the score
        sigma_noise: the diffusion sigma parameter corresponding to the time at which to evaluate the score
        device: cpu, cuda or mps

    Returns:
        coordinates jacobian of shape
        [number_of_samples, number_of_atoms * spatial_dimension, number_of_atoms * spatial_dimension]
    """

    # to get the jacobian wrt to coordinates, we need a wrapper around the _get_model_output so that the input and
    # the output are tensors, and not an AXL
    def _get_model_output_wrapper(
        coordinates, atom_types, lattice_parameters, time, sigma_noise, device
    ):
        composition = AXL(
            X=coordinates.view(1, -1, SPATIAL_DIMENSION),
            A=atom_types.unsqueeze(0),
            L=lattice_parameters.unsqueeze(0),
        )
        model_output = get_model_predictions(
            axl_network, composition, time, sigma_noise, device
        )
        # get only the coordinates output and reshape as a flattened tensor of shape
        # [number_of_samples, number_of_atoms * spatial_dimension]
        return model_output.X.view(-1)

    # flatten the last dimensions of the input coordinates tensor
    coordinates_input = composition.X.view(composition.X.shape[0], -1)
    # result is shape [number_of_samples, number_of_atoms * spatial_dimension]
    jacobian_function = jacrev(_get_model_output_wrapper, argnums=0)

    # we call this function using the relevant inputs
    coordinates_jacobians = []
    for i in tqdm(range(coordinates_input.shape[0]), "computing jacobians"):
        sample_jacobian = jacobian_function(
            coordinates_input[i],
            composition.A[i],
            composition.L[i],
            time[i].view(1),
            sigma_noise.view(1),
            device,
        )
        coordinates_jacobians.append(sample_jacobian.detach())

    return torch.stack(coordinates_jacobians, dim=0)


if __name__ == "__main__":
    main()
