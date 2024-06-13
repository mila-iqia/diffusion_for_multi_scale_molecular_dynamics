import logging
from pathlib import Path

import torch
from tqdm import tqdm

from crystal_diffusion.models.position_diffusion_lightning_model import \
    PositionDiffusionLightningModel
from crystal_diffusion.namespace import (CARTESIAN_FORCES, NOISE,
                                         NOISY_RELATIVE_COORDINATES, TIME,
                                         UNIT_CELL)
from crystal_diffusion.samplers.noisy_relative_coordinates_sampler import \
    NoisyRelativeCoordinatesSampler
from crystal_diffusion.samplers.variance_sampler import (
    ExplodingVarianceSampler, NoiseParameters)
from crystal_diffusion.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell

logger = logging.getLogger(__name__)


# Some hardcoded paths and parameters. Change as needed!
base_data_dir = Path("/Users/bruno/courtois/difface_ode/run1")
position_samples_dir = base_data_dir / "diffusion_position_samples"
energy_data_directory = base_data_dir / "energy_samples"
model_path = base_data_dir / "best_model" / "best_model-epoch=016-step=001666.ckpt"
model_name = "difface_ode_run1"

output_dir = base_data_dir / "scores_along_a_path"
output_dir.mkdir(exist_ok=True)
output_path = output_dir / f"{model_name}_path_scores.pkl"


canonical_relative_coordinates = torch.tensor([[0.00, 0.00, 0.00],
                                               [0.00, 0.50, 0.50],
                                               [0.50, 0.00, 0.50],
                                               [0.00, 0.50, 0.50],
                                               [0.25, 0.25, 0.25],
                                               [0.25, 0.75, 0.75],
                                               [0.75, 0.25, 0.75],
                                               [0.75, 0.75, 0.25]])

sigma_min = 0.001
sigma_max = 0.5
total_time_steps = 10

noise_parameters = NoiseParameters(total_time_steps=total_time_steps, sigma_min=sigma_min, sigma_max=sigma_max)
noise_sampler = ExplodingVarianceSampler(noise_parameters)

cell_dimensions = torch.tensor([5.43, 5.43, 5.43])

number_of_atoms = 8
spatial_dimension = 3

number_of_steps = 101

if __name__ == '__main__':

    torch.manual_seed(42)
    epsilon = torch.randn((number_of_atoms, spatial_dimension))
    record = dict(epsilon=epsilon)

    trajectory_sigmas = torch.linspace(sigma_min, sigma_max, number_of_steps)
    record['trajectory_sigmas'] = trajectory_sigmas

    noisy_relative_coordinates_sampler = NoisyRelativeCoordinatesSampler()
    unit_cell = torch.diag(torch.Tensor(cell_dimensions)).unsqueeze(0).repeat(number_of_steps, 1, 1)

    x = torch.stack([canonical_relative_coordinates + sigma * epsilon for sigma in trajectory_sigmas])
    x = map_relative_coordinates_to_unit_cell(x)

    times = noise_sampler._time_array
    sigmas = noise_sampler._sigma_array
    record['times'] = times
    record['sigmas'] = sigmas

    model = PositionDiffusionLightningModel.load_from_checkpoint(model_path)
    model.eval()

    normalized_scores = []
    for time, sigma in tqdm(zip(times, sigmas), 'Times'):
        batch = {NOISY_RELATIVE_COORDINATES: x,
                 NOISE: sigma * torch.ones(number_of_steps, 1),
                 TIME: time * torch.ones(number_of_steps, 1),
                 UNIT_CELL: unit_cell,
                 CARTESIAN_FORCES: torch.zeros_like(x)
                 }
        with torch.no_grad():
            normalized_scores.append(model.sigma_normalized_score_network(batch))

    record['normalized_scores'] = torch.stack(normalized_scores)

    with open(str(output_path), 'wb') as fd:
        torch.save(record, fd)
