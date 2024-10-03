import logging
import matplotlib.pyplot as plt

import numpy as np
import torch

from crystal_diffusion.analysis import PLOT_STYLE_PATH, PLEASANT_FIG_SIZE
from crystal_diffusion.analysis.analytic_score.utils import get_silicon_supercell
from crystal_diffusion.models.position_diffusion_lightning_model import PositionDiffusionLightningModel
from crystal_diffusion.samplers.exploding_variance import ExplodingVariance
from crystal_diffusion.samplers.variance_sampler import NoiseParameters
from crystal_diffusion.utils.logging_utils import setup_analysis_logger
from experiment_analysis.score_stability_analysis.util import (
    create_fixed_time_normalized_score_function, get_hessian_function, get_square_norm_and_grad_functions)


plt.style.use(PLOT_STYLE_PATH)

logger = logging.getLogger(__name__)
setup_analysis_logger()


checkpoint_path = "/home/mila/r/rousseab/scratch/experiments/oct2_egnn_1x1x1/run1/output/last_model/last_model-epoch=049-step=039100.ckpt"

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

device = torch.device('cuda')
if __name__ == "__main__":
    variance_calculator = ExplodingVariance(noise_parameters)


    logger.info("Loading checkpoint...")
    pl_model = PositionDiffusionLightningModel.load_from_checkpoint(checkpoint_path)
    pl_model.eval()

    sigma_normalized_score_network = pl_model.sigma_normalized_score_network

    for parameter in sigma_normalized_score_network.parameters():
        parameter.requires_grad_(False)

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle("Probing model along specific dimension")
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.set_title("Score Norm")
    ax2.set_title("Gradient of Score Norm")

    for ax in [ax1, ax2]:
        ax.set_xlabel('x[2][0]')

    eq_rel_coords = get_silicon_supercell(supercell_factor=1)
    for t in [1.0, 0.8, 0.5, 0.1, 0.01]:
        print(f"Doing t = {t}")

        sigma = float(variance_calculator.get_sigma(t))

        vector_field_fn = create_fixed_time_normalized_score_function(sigma_normalized_score_network,
                                                                      noise_parameters,
                                                                      time=t,
                                                                      basis_vectors=basis_vectors)

        func, grad_func = get_square_norm_and_grad_functions(vector_field_fn,
                                                             number_of_atoms,
                                                             spatial_dimension,
                                                             device)


        x0 = eq_rel_coords.flatten()

        list_dx = np.linspace(-0.5, 0.5, 101)

        list_f = []
        list_g = []
        for dx in list_dx:
            r = eq_rel_coords.copy()
            r[2][0] += dx
            x = r.flatten()
            list_f.append(func(x))
            list_g.append(grad_func(x)[6])

        list_f = np.array(list_f)
        list_g = np.array(list_g)

        ax1.plot(list_dx, list_f/np.sqrt(sigma), '-', label=f't = {t}, $\sigma$ = {sigma: 5.2e}')
        ax2.plot(list_dx, list_g, '-')

    ax1.legend(loc=0)

    fig.tight_layout()

    plt.show()

    #out = so.minimize(func, x0, jac=grad_func)