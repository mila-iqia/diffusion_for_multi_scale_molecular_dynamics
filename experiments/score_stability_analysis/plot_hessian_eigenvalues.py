import logging
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from crystal_diffusion.models.position_diffusion_lightning_model import \
    PositionDiffusionLightningModel
from crystal_diffusion.samplers.exploding_variance import ExplodingVariance
from crystal_diffusion.utils.logging_utils import setup_analysis_logger
from src.crystal_diffusion.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from src.crystal_diffusion.samplers.variance_sampler import NoiseParameters
from torch.func import jacrev
from tqdm import tqdm

from experiments import get_normalized_score_function
from experiments.analysis.analytic_score.utils import get_silicon_supercell

plt.style.use(PLOT_STYLE_PATH)

logger = logging.getLogger(__name__)
setup_analysis_logger()

system = "Si_1x1x1"

if system == "Si_1x1x1":
    checkpoint_path = Path(
        "/home/mila/r/rousseab/scratch/experiments/oct2_egnn_1x1x1/run1/output/"
        "last_model/last_model-epoch=049-step=039100.ckpt"
    )
    number_of_atoms = 8
    acell = 5.43
    supercell_factor = 1

    hessian_batch_size = 10


elif system == "Si_2x2x2":
    pickle_path = Path("/home/mila/r/rousseab/scratch/checkpoints/sota_egnn_2x2x2.pkl")
    number_of_atoms = 64
    acell = 10.86
    supercell_factor = 2
    hessian_batch_size = 1

spatial_dimension = 3
atom_types = np.ones(number_of_atoms, dtype=int)

total_time_steps = 1000
sigma_min = 0.0001
sigma_max = 0.2
noise_parameters = NoiseParameters(
    total_time_steps=total_time_steps,
    sigma_min=sigma_min,
    sigma_max=sigma_max,
)


nsteps = 501


device = torch.device("cuda")
if __name__ == "__main__":
    variance_calculator = ExplodingVariance(noise_parameters)

    basis_vectors = torch.diag(torch.tensor([acell, acell, acell])).to(device)
    equilibrium_relative_coordinates = (
        torch.from_numpy(get_silicon_supercell(supercell_factor=supercell_factor))
        .to(torch.float32)
        .to(device)
    )

    logger.info("Loading checkpoint...")

    if system == "Si_1x1x1":
        pl_model = PositionDiffusionLightningModel.load_from_checkpoint(checkpoint_path)
        pl_model.eval()
        sigma_normalized_score_network = pl_model.sigma_normalized_score_network

    elif system == "Si_2x2x2":
        sigma_normalized_score_network = torch.load(pickle_path)

    for parameter in sigma_normalized_score_network.parameters():
        parameter.requires_grad_(False)

    normalized_score_function = get_normalized_score_function(
        noise_parameters=noise_parameters,
        sigma_normalized_score_network=sigma_normalized_score_network,
        basis_vectors=basis_vectors,
    )

    times = torch.linspace(1, 0, nsteps).unsqueeze(-1)
    sigmas = variance_calculator.get_sigma(times)
    g2 = variance_calculator.get_g_squared(times)

    prefactor = -g2 / sigmas

    relative_coordinates = einops.repeat(
        equilibrium_relative_coordinates, "n s -> b n s", b=nsteps
    )

    batch_hessian_function = jacrev(normalized_score_function, argnums=0)

    list_flat_hessians = []
    for x, t in tqdm(
        zip(
            torch.split(relative_coordinates, hessian_batch_size),
            torch.split(times, hessian_batch_size),
        ),
        "Hessian",
    ):
        batch_hessian = batch_hessian_function(x, t)
        flat_hessian = einops.rearrange(
            torch.diagonal(batch_hessian, dim1=0, dim2=3),
            "n1 s1 n2 s2 b -> b (n1 s1) (n2 s2)",
        )
        list_flat_hessians.append(flat_hessian)

    flat_hessian = torch.concat(list_flat_hessians)

    p = einops.repeat(
        prefactor,
        "b 1 -> b d1 d2",
        d1=number_of_atoms * spatial_dimension,
        d2=number_of_atoms * spatial_dimension,
    ).to(flat_hessian)

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

    ax3.set_xlabel(r"$\sigma(t)$")
    ax3.set_ylabel("Small Count")
    ax3.set_ylim(0, number_of_atoms * spatial_dimension)

    ax3.semilogx(list_sigmas, small_count, "-", color="black")

    for ax in [ax1, ax2]:
        ax.set_xlabel(r"$\sigma(t)$")
        ax.set_ylabel("Eigenvalue")

        for list_e in eigenvalues:
            ax.semilogx(list_sigmas, list_e, "-", color="grey")

    ax2.set_ylim([-2.5e-4, 2.5e-4])
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(sigma_min, sigma_max)
    fig.tight_layout()

    plt.show()

    fig2 = plt.figure(figsize=(PLEASANT_FIG_SIZE[0], PLEASANT_FIG_SIZE[0]))
    fig2.suptitle("Hessian Eigenvalues At Small Time")
    ax1 = fig2.add_subplot(111)

    ax1.set_xlabel(r"$\sigma(t)$")
    ax1.set_ylabel("Eigenvalues")

    for list_e in eigenvalues:
        ax1.loglog(list_sigmas, list_e, "-", color="grey")

    ax1.set_xlim(sigma_min, 1e-2)

    fig2.tight_layout()

    plt.show()

    fig3 = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig3.suptitle("Hessian Eigenvalues of Normalized Score")
    ax1 = fig3.add_subplot(121)
    ax2 = fig3.add_subplot(122)

    ax1.set_xlabel(r"$\sigma(t)$")
    ax1.set_ylabel("Eigenvalues")

    ax2.set_ylabel(r"$g^2(t) / \sigma(t)$")
    ax2.set_xlabel(r"$\sigma(t)$")

    label1 = r"$\sigma(t)/g^2 \times \bf H$"
    label2 = r"$\bf H$"
    for list_e in eigenvalues:
        ax1.semilogx(
            list_sigmas,
            list_e / (-prefactor.flatten()),
            "-",
            lw=1,
            color="red",
            label=label1,
        )
        ax1.semilogx(list_sigmas, list_e, "-", color="grey", lw=1, label=label2)
        label1 = "__nolabel__"
        label2 = "__nolabel__"

    ax1.legend(loc=0)
    ax2.semilogx(list_sigmas, (-prefactor.flatten()), "-", color="blue")

    for ax in [ax1, ax2]:
        ax.set_xlim(sigma_min, sigma_max)

    fig3.tight_layout()

    plt.show()

    jacobian_eig_at_t0 = eigenvalues[:, -1] / (-prefactor[-1])
