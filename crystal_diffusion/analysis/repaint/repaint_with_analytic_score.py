import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

from crystal_diffusion import ANALYSIS_RESULTS_DIR
from crystal_diffusion.analysis import PLOT_STYLE_PATH
from crystal_diffusion.analysis.analytic_score.utils import (
    get_silicon_supercell, get_unit_cells)
from crystal_diffusion.generators.constrained_langevin_dynamics_generator import (
    ConstrainedAnnealedLangevinDyamicsGeneratorParameters,
    ConstrainedAnnealedLangevinDynamicsGenerator)
from crystal_diffusion.models.score_networks.analytical_score_network import (
    AnalyticalScoreNetwork, AnalyticalScoreNetworkParameters)
from crystal_diffusion.samplers.variance_sampler import NoiseParameters
from crystal_diffusion.utils.logging_utils import setup_analysis_logger
from crystal_diffusion.utils.structure_utils import create_structure

logger = logging.getLogger(__name__)
setup_analysis_logger()

repaint_dir = ANALYSIS_RESULTS_DIR / "repaint"
repaint_dir.mkdir(exist_ok=True)

plt.style.use(PLOT_STYLE_PATH)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

kmax = 1
supercell_factor = 1
variance_parameter = 0.001 / supercell_factor
number_of_samples = 100
total_time_steps = 101
number_of_corrector_steps = 1

acell = 5.43  # Angstroms.

constrained_relative_coordinates = np.array([[0.5, 0.5, 0.25],
                                             [0.5, 0.5, 0.5],
                                             [0.5, 0.5, 0.75]], dtype=np.float32)

if __name__ == '__main__':
    logger.info("Setting up parameters")

    equilibrium_relative_coordinates = get_silicon_supercell(supercell_factor=supercell_factor).astype(np.float32)
    number_of_atoms, spatial_dimension = equilibrium_relative_coordinates.shape

    unit_cells = get_unit_cells(acell=acell,
                                spatial_dimension=spatial_dimension,
                                number_of_samples=number_of_samples)

    noise_parameters = NoiseParameters(total_time_steps=total_time_steps,
                                       sigma_min=0.001,
                                       sigma_max=0.5)

    score_network_parameters = AnalyticalScoreNetworkParameters(
        number_of_atoms=number_of_atoms,
        spatial_dimension=spatial_dimension,
        kmax=kmax,
        equilibrium_relative_coordinates=torch.from_numpy(equilibrium_relative_coordinates).to(device),
        variance_parameter=variance_parameter)

    sigma_normalized_score_network = AnalyticalScoreNetwork(score_network_parameters)

    sampling_parameters = ConstrainedAnnealedLangevinDyamicsGeneratorParameters(
        number_of_corrector_steps=number_of_corrector_steps,
        spatial_dimension=spatial_dimension,
        number_of_atoms=number_of_atoms,
        number_of_samples=number_of_samples,
        cell_dimensions=3 * [acell],
        constrained_relative_coordinates=constrained_relative_coordinates,
        record_samples=False)

    position_generator = ConstrainedAnnealedLangevinDynamicsGenerator(
        noise_parameters=noise_parameters,
        sampling_parameters=sampling_parameters,
        sigma_normalized_score_network=sigma_normalized_score_network)

    logger.info("Drawing constrained samples")
    samples = position_generator.sample(number_of_samples=number_of_samples,
                                        device=device,
                                        unit_cell=unit_cells).detach()

    logger.info("Creating CIF files")

    lattice_basis_vectors = np.diag([acell, acell, acell])
    species = number_of_atoms * ['Si']

    relative_coordinates = equilibrium_relative_coordinates
    equilibrium_structure = create_structure(lattice_basis_vectors, relative_coordinates, species)
    equilibrium_structure.to_file(str(repaint_dir / "equilibrium_positions.cif"))

    relative_coordinates[:len(constrained_relative_coordinates)] = constrained_relative_coordinates
    forced_structure = create_structure(lattice_basis_vectors, relative_coordinates, species)
    forced_structure.to_file(str(repaint_dir / "forced_constraint_positions.cif"))

    samples_dir = repaint_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    for idx, sample in enumerate(samples.cpu().numpy()):
        structure = create_structure(lattice_basis_vectors, sample, species)
        structure.to_file(str(samples_dir / f"sample_{idx}.cif"))
