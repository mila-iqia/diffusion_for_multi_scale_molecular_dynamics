"""Get sampling constraints.

This script selects a semi-random configuration from the validation dataset and noises it to various time steps.
"""
from pathlib import Path

import einops
import torch
from tqdm import tqdm

from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.lammps_for_diffusion_data_module import (
    LammpsDataModuleParameters, LammpsForDiffusionDataModule)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    ATOM_TYPES, AXL, NOISY_AXL_COMPOSITION, RELATIVE_COORDINATES)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_scheduler import \
    NoiseScheduler
from diffusion_for_multi_scale_molecular_dynamics.noisers.atom_types_noiser import \
    AtomTypesNoiser
from diffusion_for_multi_scale_molecular_dynamics.noisers.relative_coordinates_noiser import \
    RelativeCoordinatesNoiser
from diffusion_for_multi_scale_molecular_dynamics.utils.d3pm_utils import \
    class_index_to_onehot
from diffusion_for_multi_scale_molecular_dynamics.utils.tensor_utils import (
    broadcast_batch_matrix_tensor_to_all_dimensions,
    broadcast_batch_tensor_to_all_dimensions)
from experiments.start_time_constrained_trajectories import EXPERIMENT_DIR


def get_noisy_composition(reference_composition: AXL,
                          num_atom_types: int,
                          sigma: float,
                          q_bar_matrix: torch.Tensor):
    """Generate noisy composition."""
    atom_type_noiser = AtomTypesNoiser()
    relative_coordinates_noiser = RelativeCoordinatesNoiser()

    x0 = reference_composition.X
    a0 = reference_composition.A
    lt = reference_composition.L  # we will not noise the lattice parameters

    batch_size = x0.shape[0]

    # The input sigma should have dimension [batch_size]. Broadcast these values to be of shape
    # [batch_size, number_of_atoms, spatial_dimension] , which can be interpreted as
    # [batch_size, (configuration)]. All the sigma values must be the same for a given configuration.
    sigmas = broadcast_batch_tensor_to_all_dimensions(
        batch_values=sigma * torch.ones(batch_size), final_shape=x0.shape
    )
    # we can now get noisy coordinates
    xt = relative_coordinates_noiser.get_noisy_relative_coordinates_sample(x0, sigmas)

    # to get noisy atom types, we need to broadcast the transition matrices q_bar from size
    # [batch_size, num_atom_types, num_atom_types] to [batch_size, number_of_atoms, num_atom_types, num_atom_types].
    # All the matrices must be the same for all atoms in a given configuration.
    batch_q_bar_matrix = einops.repeat(q_bar_matrix,
                                       "... -> batch_size ...",
                                       batch_size=batch_size)

    q_bar_matrices = broadcast_batch_matrix_tensor_to_all_dimensions(
        batch_values=batch_q_bar_matrix, final_shape=a0.shape
    )

    # we also need the atom types to be one-hot vector and not a class index
    a0_onehot = class_index_to_onehot(a0, num_atom_types + 1)

    at = atom_type_noiser.get_noisy_atom_types_sample(a0_onehot, q_bar_matrices)

    noisy_composition = AXL(A=at, X=xt, L=lt)

    return noisy_composition


lammps_run_dir = Path("/Users/brunorousseau/courtois/data/SiGe_diffusion_2x2x2")
processed_dataset_dir = lammps_run_dir / "processed"
cache_dir = lammps_run_dir / "cache"

elements = ['Si', 'Ge']
num_atom_types = len(elements)

loader_parameters = LammpsDataModuleParameters(batch_size=1024,
                                               num_workers=8,
                                               max_atom=64,
                                               elements=elements)

output_dir = EXPERIMENT_DIR / "sampling_constraints"
output_dir.mkdir(parents=True, exist_ok=True)

schedule_type = "linear"

noise_parameters = NoiseParameters(total_time_steps=1000,
                                   sigma_min=0.0001,
                                   sigma_max=0.2,
                                   schedule_type=schedule_type)

batch_size = 128
list_start_time_index = torch.arange(100, 1050, 50)

noise_scheduler = NoiseScheduler(
    noise_parameters,
    num_classes=num_atom_types + 1,  # add 1 for the MASK class
)

if __name__ == "__main__":

    data_module = LammpsForDiffusionDataModule(lammps_run_dir=str(lammps_run_dir),
                                               processed_dataset_dir=str(processed_dataset_dir),
                                               hyper_params=loader_parameters,
                                               working_cache_dir=str(cache_dir))
    data_module.setup()
    valid_example = data_module.valid_dataset[0]

    x0 = einops.repeat(valid_example[RELATIVE_COORDINATES],
                       "... -> batch_size ...",
                       batch_size=batch_size)

    a0 = einops.repeat(valid_example[ATOM_TYPES],
                       "... -> batch_size ...",
                       batch_size=batch_size)

    l0 = einops.repeat(torch.diag(valid_example['box']),
                       "... -> batch_size ...",
                       batch_size=batch_size)

    noise, _ = noise_scheduler.get_all_sampling_parameters()

    reference_composition = AXL(A=a0, X=x0, L=l0)

    for start_time_index in tqdm(list_start_time_index, "T"):
        idx = start_time_index - 1  # python starts indices at zero
        sigma = noise.sigma[idx]
        q_matrix = noise.sigma[idx]

        noisy_composition = get_noisy_composition(reference_composition,
                                                  num_atom_types=num_atom_types,
                                                  sigma=noise.sigma[idx],
                                                  q_bar_matrix=noise.q_bar_matrix[idx])

        output_path = output_dir / f"noise_composition_{schedule_type}_schedule_start_T_{start_time_index}.pickle"

        data = {NOISY_AXL_COMPOSITION: noisy_composition,
                'start_time_step_index': start_time_index,
                'noise_parameters': noise_parameters
                }

        torch.save(data, output_path)
