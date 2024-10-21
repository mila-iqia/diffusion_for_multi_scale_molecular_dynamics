import logging

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from crystal_diffusion.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from crystal_diffusion.models.position_diffusion_lightning_model import \
    PositionDiffusionLightningModel
from crystal_diffusion.samplers.exploding_variance import ExplodingVariance
from crystal_diffusion.samplers.variance_sampler import NoiseParameters
from crystal_diffusion.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell
from crystal_diffusion.utils.logging_utils import setup_analysis_logger
from experiments import create_fixed_time_normalized_score_function
from experiments.analysis.analytic_score.utils import get_silicon_supercell

plt.style.use(PLOT_STYLE_PATH)

logger = logging.getLogger(__name__)
setup_analysis_logger()


checkpoint_path = ("/home/mila/r/rousseab/scratch/experiments/oct2_egnn_1x1x1/run1/"
                   "output/last_model/last_model-epoch=049-step=039100.ckpt")

spatial_dimension = 3
number_of_atoms = 8
atom_types = np.ones(number_of_atoms, dtype=int)

acell = 5.43
basis_vectors = torch.diag(torch.tensor([acell, acell, acell]))

total_time_steps = 1000
noise_parameters = NoiseParameters(
    total_time_steps=total_time_steps,
    sigma_min=0.0001,
    sigma_max=0.2,
)

device = torch.device("cuda")
if __name__ == "__main__":
    variance_calculator = ExplodingVariance(noise_parameters)

    logger.info("Loading checkpoint...")
    pl_model = PositionDiffusionLightningModel.load_from_checkpoint(checkpoint_path)
    pl_model.eval()

    sigma_normalized_score_network = pl_model.sigma_normalized_score_network

    for parameter in sigma_normalized_score_network.parameters():
        parameter.requires_grad_(False)

    equilibrium_relative_coordinates = torch.from_numpy(
        get_silicon_supercell(supercell_factor=1)
    ).to(torch.float32)

    direction = torch.zeros_like(equilibrium_relative_coordinates)

    # Move a single atom
    # direction[0, 0] = 1.0
    # list_delta = torch.linspace(-0.5, 0.5, 101)

    # Put two particles on top of each other
    dv = equilibrium_relative_coordinates[0] - equilibrium_relative_coordinates[1]
    direction[0] = -0.5 * dv
    direction[1] = 0.5 * dv
    list_delta = torch.linspace(0., 2.0, 201)

    relative_coordinates = []
    for delta in list_delta:
        relative_coordinates.append(
            equilibrium_relative_coordinates + delta * direction
        )
    relative_coordinates = map_relative_coordinates_to_unit_cell(
        torch.stack(relative_coordinates)
    ).to(device)

    list_t = torch.tensor([0.8, 0.7, 0.5, 0.3, 0.1, 0.01])
    list_sigmas = variance_calculator.get_sigma(list_t)
    list_norms = []
    for t in tqdm(list_t, "norms"):
        vector_field_fn = create_fixed_time_normalized_score_function(
            sigma_normalized_score_network,
            noise_parameters,
            time=t,
            basis_vectors=basis_vectors,
        )

        normalized_scores = vector_field_fn(relative_coordinates)
        flat_normalized_scores = einops.rearrange(
            normalized_scores, " b n s -> b (n s)"
        )
        list_norms.append(flat_normalized_scores.norm(dim=-1).cpu())

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle("Normalized Score Norm Along Specific Direction")
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(r"$\delta$")
    ax1.set_ylabel(r"$|{\bf n}({\bf x}, t)|$")

    for t, sigma, norms in zip(list_t, list_sigmas, list_norms):
        ax1.plot(
            list_delta, norms, "-", label=f"t = {t: 3.2f}, $\\sigma$ = {sigma: 5.2e}"
        )

    ax1.legend(loc=0)

    fig.tight_layout()

    plt.show()
