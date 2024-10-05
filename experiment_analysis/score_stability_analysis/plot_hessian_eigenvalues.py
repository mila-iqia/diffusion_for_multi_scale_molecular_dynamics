import logging
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.func import jacrev

from crystal_diffusion.analysis import PLOT_STYLE_PATH, PLEASANT_FIG_SIZE
from crystal_diffusion.analysis.analytic_score.utils import get_silicon_supercell
from crystal_diffusion.models.position_diffusion_lightning_model import PositionDiffusionLightningModel
from crystal_diffusion.samplers.exploding_variance import ExplodingVariance
from crystal_diffusion.samplers.variance_sampler import NoiseParameters
from crystal_diffusion.utils.logging_utils import setup_analysis_logger
from experiment_analysis.score_stability_analysis.util import get_normalized_score_function

plt.style.use(PLOT_STYLE_PATH)





logger = logging.getLogger(__name__)
setup_analysis_logger()


checkpoint_path = Path("/home/mila/r/rousseab/scratch/experiments/oct2_egnn_1x1x1/run1/output/last_model/last_model-epoch=049-step=039100.ckpt")

spatial_dimension = 3
number_of_atoms = 8
atom_types = np.ones(number_of_atoms, dtype=int)

acell = 5.43

total_time_steps = 1000
sigma_min = 0.0001
sigma_max = 0.2
noise_parameters = NoiseParameters(
    total_time_steps=total_time_steps,
    sigma_min=sigma_min,
    sigma_max=sigma_max,
)


nsteps = 501

hessian_batch_size = 10

device = torch.device('cuda')
if __name__ == "__main__":
    variance_calculator = ExplodingVariance(noise_parameters)

    basis_vectors = torch.diag(torch.tensor([acell, acell, acell])).to(device)
    equilibrium_relative_coordinates = torch.from_numpy(
        get_silicon_supercell(supercell_factor=1)).to(torch.float32).to(device)

    logger.info("Loading checkpoint...")
    pl_model = PositionDiffusionLightningModel.load_from_checkpoint(checkpoint_path)
    pl_model.eval()

    sigma_normalized_score_network = pl_model.sigma_normalized_score_network

    for parameter in sigma_normalized_score_network.parameters():
        parameter.requires_grad_(False)

    normalized_score_function = get_normalized_score_function(
        noise_parameters=noise_parameters,
        sigma_normalized_score_network=sigma_normalized_score_network,
        basis_vectors=basis_vectors)


    times = torch.linspace(1, 0, nsteps).unsqueeze(-1)
    sigmas = variance_calculator.get_sigma(times)
    g2 = variance_calculator.get_g_squared(times)

    prefactor = -g2 / sigmas

    relative_coordinates = einops.repeat(equilibrium_relative_coordinates, "n s -> b n s", b=nsteps)


    batch_hessian_function = jacrev(normalized_score_function, argnums=0)

    list_flat_hessians = []
    for x, t in zip(torch.split(relative_coordinates, hessian_batch_size), torch.split(times, hessian_batch_size)):
        batch_hessian = batch_hessian_function(x, t)
        flat_hessian = einops.rearrange(torch.diagonal(batch_hessian, dim1=0, dim2=3),
                                        "n1 s1 n2 s2 b -> b (n1 s1) (n2 s2)")
        list_flat_hessians.append(flat_hessian)

    flat_hessian = torch.concat(list_flat_hessians)

    p = einops.repeat(prefactor ,"b 1 -> b d1 d2",
                      d1=number_of_atoms * spatial_dimension,
                      d2=number_of_atoms * spatial_dimension).to(flat_hessian)

    normalized_hessian = p * flat_hessian

    eigenvalues, eigenvectors = torch.linalg.eigh(normalized_hessian)
    eigenvalues = eigenvalues.cpu().transpose(1, 0)

    small_count = (eigenvalues < 5e-4).sum(dim=0)
    list_times = times.flatten().cpu()
    list_sigmas = sigmas.flatten().cpu()


    fig = plt.figure(figsize=(PLEASANT_FIG_SIZE[0], PLEASANT_FIG_SIZE[0]))
    fig.suptitle("Hessian Eigenvalues")
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    ax3.set_xlabel(r'$\sigma(t)$')
    ax3.set_ylabel('Small Count')
    ax3.set_ylim(0, number_of_atoms * spatial_dimension)

    ax3.semilogx(list_sigmas, small_count, '-', color='black')

    for ax in [ax1, ax2]:
        ax.set_xlabel(r'$\sigma(t)$')
        ax.set_ylabel('Eigenvalue')

        for list_e in eigenvalues:
            ax.semilogx(list_sigmas, list_e, '.', color='grey')

    ax2.set_ylim([-2.5e-4, 2.5e-4])
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(sigma_min, sigma_max)
    fig.tight_layout()

    plt.show()
