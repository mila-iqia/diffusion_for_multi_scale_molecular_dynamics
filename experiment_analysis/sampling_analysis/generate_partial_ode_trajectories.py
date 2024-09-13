import logging
import tempfile
from pathlib import Path

import einops
import numpy as np
import torch
from pymatgen.core import Lattice, Structure
from tqdm import tqdm

from crystal_diffusion.analysis.generator_sample_analysis_utils import \
    PartialODEPositionGenerator
from crystal_diffusion.data.diffusion.data_loader import LammpsLoaderParameters, LammpsForDiffusionDataModule
from crystal_diffusion.generators.ode_position_generator import ODESamplingParameters
from crystal_diffusion.models.position_diffusion_lightning_model import \
    PositionDiffusionLightningModel
from crystal_diffusion.oracle.lammps import get_energy_and_forces_from_lammps
from crystal_diffusion.samplers.noisy_relative_coordinates_sampler import \
    NoisyRelativeCoordinatesSampler
from crystal_diffusion.samplers.variance_sampler import NoiseParameters
from crystal_diffusion.utils.logging_utils import setup_analysis_logger
from crystal_diffusion.utils.tensor_utils import \
    broadcast_batch_tensor_to_all_dimensions

logger = logging.getLogger(__name__)

setup_analysis_logger()
# Some hardcoded paths and parameters. Change as needed!

data_directory = Path("/home/mila/r/rousseab/scratch/data/")
dataset_name = 'si_diffusion_2x2x2'
lammps_run_dir = data_directory / dataset_name
processed_dataset_dir = lammps_run_dir / "processed"
cache_dir = lammps_run_dir / "cache"

data_params = LammpsLoaderParameters(batch_size=1024, max_atom=64)

checkpoint_path = "/network/scratch/r/rousseab/checkpoints/EGNN_Sept_10/last_model-epoch=045-step=035972.ckpt"

partial_samples_dir = Path("/network/scratch/r/rousseab/partial_samples_EGNN_Sept_10/")
partial_samples_dir.mkdir(exist_ok=True)

sigma_min = 0.001
sigma_max = 0.5
total_time_steps = 100

noise_parameters = NoiseParameters(total_time_steps=total_time_steps,
                                   sigma_min=sigma_min,
                                   sigma_max=sigma_max)

absolute_solver_tolerance = 1.0e-3
relative_solver_tolerance = 1.0e-2

spatial_dimension = 3
batch_size = 4
device = torch.device('cuda')

if __name__ == '__main__':
    logger.info("Extracting a validation configuration")
    # Extract a configuration from the validation set
    datamodule = LammpsForDiffusionDataModule(
        lammps_run_dir=lammps_run_dir,
        processed_dataset_dir=processed_dataset_dir,
        hyper_params=data_params,
        working_cache_dir=cache_dir)
    datamodule.setup()

    validation_example = datamodule.valid_dataset[0]
    reference_relative_coordinates = validation_example['relative_coordinates']
    number_of_atoms = int(validation_example['natom'])
    cell_dimensions = validation_example['box']

    logger.info("Writing validation configuration to cif file")
    a, b, c = cell_dimensions.numpy()
    lattice = Lattice.from_parameters(a=a, b=b, c=c, alpha=90, beta=90, gamma=90)

    reference_structure =  Structure(lattice=lattice,
                                     species=number_of_atoms*['Si'],
                                     coords=reference_relative_coordinates.numpy())

    reference_structure.to(str(partial_samples_dir  / "reference_validation_structure.cif"))

    logger.info("Extracting checkpoint")
    noisy_relative_coordinates_sampler = NoisyRelativeCoordinatesSampler()

    unit_cell = torch.diag(torch.Tensor(cell_dimensions)).unsqueeze(0).repeat(batch_size, 1, 1)
    box = unit_cell[0].numpy()

    x0 = einops.repeat(reference_relative_coordinates, "n d -> b n d", b=batch_size)

    model = PositionDiffusionLightningModel.load_from_checkpoint(checkpoint_path)
    model.eval()

    list_tf = np.linspace(0.1, 1, 20)
    atom_types = np.ones(number_of_atoms, dtype=int)

    logger.info("Draw samples")
    with torch.no_grad():
        for tf in tqdm(list_tf, 'times'):
            times = torch.ones(batch_size) * tf
            sigmas = sigma_min ** (1.0 - times) * sigma_max ** times

            broadcast_sigmas = broadcast_batch_tensor_to_all_dimensions(batch_values=sigmas,
                                                                        final_shape=x0.shape)
            xt = noisy_relative_coordinates_sampler.get_noisy_relative_coordinates_sample(x0, broadcast_sigmas)

            noise_parameters.total_time_steps = int(1000 * tf) + 1
            sampling_parameters = ODESamplingParameters(
                number_of_atoms=number_of_atoms,
                number_of_samples=batch_size,
                record_samples=True,
                cell_dimensions=list(cell_dimensions.cpu().numpy()),
                absolute_solver_tolerance=absolute_solver_tolerance,
                relative_solver_tolerance=relative_solver_tolerance)

            generator = PartialODEPositionGenerator(noise_parameters=noise_parameters,
                                                    sampling_parameters=sampling_parameters,
                                                    sigma_normalized_score_network=model.sigma_normalized_score_network,
                                                    initial_relative_coordinates=xt,
                                                    tf=tf)

            logger.info("Generating Samples")
            batch_relative_coordinates = generator.sample(number_of_samples=batch_size,
                                                          device=device,
                                                          unit_cell=unit_cell).cpu()
            sample_output_path = str(partial_samples_dir / f"diffusion_position_sample_time={tf:3.2f}.pt")
            generator.sample_trajectory_recorder.write_to_pickle(sample_output_path)
            logger.info("Done Generating Samples")

            batch_cartesian_positions = torch.bmm(batch_relative_coordinates, unit_cell)

            list_energy = []
            logger.info("Compute energy from Oracle")
            with tempfile.TemporaryDirectory() as tmp_work_dir:
                for positions in batch_cartesian_positions.numpy():
                    energy, forces = get_energy_and_forces_from_lammps(positions,
                                                                       box,
                                                                       atom_types,
                                                                       tmp_work_dir=tmp_work_dir)
                    list_energy.append(energy)

            energies = torch.tensor(list_energy)
            logger.info("Done Computing energy from Oracle")

            energy_output_path = str(partial_samples_dir / f"diffusion_energies_sample_time={tf:2.1f}.pt")
            with open(energy_output_path, 'wb') as fd:
                torch.save(energies, fd)
