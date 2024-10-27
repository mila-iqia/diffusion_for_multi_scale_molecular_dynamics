import itertools
import pickle
import tempfile

import einops
import numpy as np
import torch
from tqdm import tqdm

from diffusion_for_multi_scale_molecular_dynamics.callbacks.sampling_visualization_callback import \
    logger
from diffusion_for_multi_scale_molecular_dynamics.generators.langevin_generator import \
    LangevinGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_position_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.analytical_score_network import (
    AnalyticalScoreNetworkParameters, TargetScoreBasedAnalyticalScoreNetwork)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps import \
    get_energy_and_forces_from_lammps
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import (
    get_positions_from_coordinates, map_relative_coordinates_to_unit_cell)
from experiments.analysis.analytic_score.exploring_langevin_generator import \
    LANGEVIN_EXPLORATION_DIRECTORY
from experiments.analysis.analytic_score.utils import (get_exact_samples,
                                                       get_silicon_supercell)


class EnergyCalculator:
    """Calculate Energies with LAMMPS."""

    def __init__(self, unit_cell: torch.Tensor, number_of_atoms: int):
        """Init method."""
        self.cell_dimensions = cell_dimensions
        self.atom_types = np.ones(number_of_atoms, dtype=int)
        self.unit_cell = unit_cell

    def _get_basis_vectors(self, batch_size: int) -> torch.Tensor:
        """Get basis vectors."""
        basis_vectors = self.unit_cell.unsqueeze(0).repeat(batch_size, 1, 1)
        return basis_vectors

    def compute_oracle_energies(
        self, batch_relative_coordinates: torch.Tensor
    ) -> np.ndarray:
        """Compute oracle energies."""
        batch_size = batch_relative_coordinates.shape[0]
        batched_unit_cells = einops.repeat(
            self.unit_cell, "d1 d2 -> b d1 d2", b=batch_size
        )

        batch_cartesian_positions = get_positions_from_coordinates(
            batch_relative_coordinates, batched_unit_cells
        )

        list_energy = []

        logger.info("Compute energy from Oracle")
        with tempfile.TemporaryDirectory() as tmp_work_dir:
            for positions, box in zip(
                batch_cartesian_positions.cpu().numpy(),
                batched_unit_cells.cpu().numpy(),
            ):
                energy, forces = get_energy_and_forces_from_lammps(
                    positions, box, self.atom_types, tmp_work_dir=tmp_work_dir
                )
                list_energy.append(energy)

        return np.array(list_energy)


def generate_exact_samples(
    equilibrium_relative_coordinates: torch.Tensor,
    sigma_d: float,
    number_of_samples: int,
):
    """Generate Gaussian samples about the equilibrium relative coordinates."""
    variance_parameter = sigma_d**2

    number_of_atoms, spatial_dimension = equilibrium_relative_coordinates.shape
    nd = number_of_atoms * spatial_dimension

    inverse_covariance = torch.diag(torch.ones(nd)) / variance_parameter
    inverse_covariance = inverse_covariance.reshape(
        number_of_atoms, spatial_dimension, number_of_atoms, spatial_dimension
    )

    exact_samples = get_exact_samples(
        equilibrium_relative_coordinates, inverse_covariance, number_of_samples
    )
    exact_samples = map_relative_coordinates_to_unit_cell(exact_samples)
    return exact_samples


device = torch.device("cpu")

number_of_atoms = 64
spatial_dimension = 3
kmax = 8
acell = 5.43
supercell_factor = 2
cell_dimensions = 3 * [acell * supercell_factor]

pickle_directory = (
    LANGEVIN_EXPLORATION_DIRECTORY
    / f"Si_{supercell_factor}x{supercell_factor}x{supercell_factor}"
)
pickle_directory.mkdir(exist_ok=True, parents=True)

number_of_samples = 1000

list_sigma_d = [0.1, 0.01, 0.001]
list_sigma_min = [0.00001, 0.0001, 0.001]
list_number_of_corrector_steps = [1, 10, 100]
list_total_time_steps = [10, 100, 1000]


if __name__ == "__main__":

    unit_cell = torch.diag(torch.tensor(cell_dimensions))
    batched_unit_cells = einops.repeat(
        unit_cell, "d1 d2 -> b d1 d2", b=number_of_samples
    )

    energy_calculator = EnergyCalculator(
        unit_cell=unit_cell, number_of_atoms=number_of_atoms
    )

    equilibrium_relative_coordinates = torch.from_numpy(
        get_silicon_supercell(supercell_factor=supercell_factor)
    ).to(torch.float32)

    for sigma_d in tqdm(list_sigma_d, "exact energies"):
        name = f"Sd={sigma_d}"
        output_path = pickle_directory / f"exact_energies_{name}.pkl"
        if output_path.is_file():
            logger.info(f"file {name} already exists. Moving on.")
            continue

        exact_samples = generate_exact_samples(
            equilibrium_relative_coordinates,
            sigma_d=sigma_d,
            number_of_samples=number_of_samples,
        )

        exact_samples_energies = energy_calculator.compute_oracle_energies(
            exact_samples
        )
        result = dict(sigma_d=sigma_d, energies=exact_samples_energies)

        with open(output_path, "wb") as fd:
            pickle.dump(result, fd)

    prod = itertools.product(
        list_number_of_corrector_steps,
        list_sigma_d,
        list_sigma_min,
        list_total_time_steps,
    )
    for number_of_corrector_steps, sigma_d, sigma_min, total_time_steps in tqdm(
        prod, "sample energies"
    ):
        name = f"Sd={sigma_d}_Sm={sigma_min}_S={total_time_steps}_C={number_of_corrector_steps}"
        output_path = pickle_directory / f"sampled_energies_{name}.pkl"
        if output_path.is_file():
            logger.info(f"file {name} already exists. Moving on.")
            continue

        score_network_parameters = AnalyticalScoreNetworkParameters(
            number_of_atoms=number_of_atoms,
            spatial_dimension=spatial_dimension,
            use_permutation_invariance=False,
            kmax=kmax,
            equilibrium_relative_coordinates=equilibrium_relative_coordinates,
            variance_parameter=sigma_d**2,
        )

        sigma_normalized_score_network = TargetScoreBasedAnalyticalScoreNetwork(
            score_network_parameters
        )

        sampling_parameters = PredictorCorrectorSamplingParameters(
            number_of_atoms=number_of_atoms,
            number_of_corrector_steps=number_of_corrector_steps,
            number_of_samples=number_of_samples,
            sample_batchsize=number_of_samples,
            sample_every_n_epochs=1,
            first_sampling_epoch=0,
            cell_dimensions=cell_dimensions,
            record_samples=False,
        )

        noise_parameters = NoiseParameters(
            total_time_steps=total_time_steps, sigma_min=sigma_min, sigma_max=0.5
        )

        generator = LangevinGenerator(
            noise_parameters=noise_parameters,
            sampling_parameters=sampling_parameters,
            sigma_normalized_score_network=sigma_normalized_score_network,
        )

        generated_samples = generator.sample(
            number_of_samples, device=device, unit_cell=batched_unit_cells
        )

        generated_samples_energies = energy_calculator.compute_oracle_energies(
            generated_samples
        )

        result = dict(
            sigma_d=sigma_d,
            sigma_min=sigma_min,
            total_time_steps=total_time_steps,
            number_of_corrector_steps=number_of_corrector_steps,
            energies=generated_samples_energies,
        )
        name = f"Sd={sigma_d}_Sm={sigma_min}_S={total_time_steps}_C={number_of_corrector_steps}"
        # Pickle a lot of intermediate things to avoid losing everything in a crash.
        with open(output_path, "wb") as fd:
            pickle.dump(result, fd)
