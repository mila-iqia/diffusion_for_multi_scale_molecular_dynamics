import logging

import numpy as np
import scipy.optimize as so
import torch

from crystal_diffusion.models.score_networks.mlp_score_network import \
    MLPScoreNetworkParameters
from crystal_diffusion.models.score_networks.score_network_factory import \
    create_score_network
from crystal_diffusion.samplers.variance_sampler import NoiseParameters
from crystal_diffusion.utils.logging_utils import setup_analysis_logger
from experiment_analysis.score_stability_analysis.util import (
    create_fixed_time_vector_field_function, get_flat_vector_field_function,
    get_hessian_function)

logger = logging.getLogger(__name__)
setup_analysis_logger()

checkpoint_path = ("/network/scratch/r/rousseab/experiments/sept21_egnn_2x2x2/run4/"
                   "output/best_model/best_model-epoch=024-step=019550.ckpt")


spatial_dimension = 3
number_of_atoms = 64
atom_types = np.ones(number_of_atoms, dtype=int)

acell = 10.86
basis_vectors = torch.diag(torch.tensor([acell, acell, acell]))

total_time_steps = 1000
noise_parameters = NoiseParameters(
    total_time_steps=total_time_steps,
    sigma_min=0.0001,
    sigma_max=0.2,
)


if __name__ == "__main__":
    """
    logger.info("Loading checkpoint...")
    pl_model = PositionDiffusionLightningModel.load_from_checkpoint(checkpoint_path)
    pl_model.eval()

    sigma_normalized_score_network = pl_model.sigma_normalized_score_network
    """

    score_network_parameters = MLPScoreNetworkParameters(
        number_of_atoms=number_of_atoms,
        n_hidden_dimensions=3,
        embedding_dimensions_size=8,
        hidden_dimensions_size=8,
        spatial_dimension=spatial_dimension,
    )
    sigma_normalized_score_network = create_score_network(score_network_parameters)
    for parameter in sigma_normalized_score_network.parameters():
        parameter.requires_grad_(False)

    vector_field_fn = create_fixed_time_vector_field_function(sigma_normalized_score_network,
                                                              noise_parameters,
                                                              time=0.1,
                                                              basis_vectors=basis_vectors)

    hessian_fn = get_hessian_function(vector_field_fn)

    batch_size = 12

    relative_coordinates = torch.rand(batch_size, number_of_atoms, spatial_dimension)

    vector_field = vector_field_fn(relative_coordinates)

    hessian = hessian_fn(relative_coordinates)

    eigenvalues, _ = torch.linalg.eig(hessian)

    func = get_flat_vector_field_function(vector_field_fn, number_of_atoms, spatial_dimension)
    x0 = torch.rand(192).numpy()
    out = so.root(func, x0)
