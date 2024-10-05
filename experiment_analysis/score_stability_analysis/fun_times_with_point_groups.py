import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from einops import einops

from torch.func import jacrev

from crystal_diffusion.analysis import PLOT_STYLE_PATH
from crystal_diffusion.analysis.analytic_score.utils import get_silicon_supercell
from crystal_diffusion.models.position_diffusion_lightning_model import PositionDiffusionLightningModel
from crystal_diffusion.samplers.variance_sampler import NoiseParameters
from crystal_diffusion.utils.basis_transformations import map_relative_coordinates_to_unit_cell
from crystal_diffusion.utils.logging_utils import setup_analysis_logger
from experiment_analysis.score_stability_analysis.util import get_cubic_point_group_symmetries, \
    get_normalized_score_function
from tests.fake_data_utils import find_aligning_permutation

plt.style.use(PLOT_STYLE_PATH)

logger = logging.getLogger(__name__)
setup_analysis_logger()

checkpoint_path = Path("/home/mila/r/rousseab/scratch/experiments/oct2_egnn_1x1x1/run1/output/last_model/last_model-epoch=049-step=039100.ckpt")

spatial_dimension = 3
number_of_atoms = 8

acell = 5.43

total_time_steps = 1000
sigma_min = 0.0001
sigma_max = 0.2
noise_parameters = NoiseParameters(
    total_time_steps=total_time_steps,
    sigma_min=sigma_min,
    sigma_max=sigma_max,
)

device = torch.device('cuda')
if __name__ == "__main__":
    equilibrium_relative_coordinates = torch.from_numpy(get_silicon_supercell(supercell_factor=1)).to(torch.float32)

    list_g = get_cubic_point_group_symmetries()
    nsym = len(list_g)

    # Find the group operations that leave the set of equilibrium coordinates unchanged
    list_stabilizing_g = []
    list_permutations = []

    for g in list_g:
        x = map_relative_coordinates_to_unit_cell(equilibrium_relative_coordinates @ g.transpose(1, 0))
        try:
            permutation = find_aligning_permutation(equilibrium_relative_coordinates, x, tol=1e-8)
            list_stabilizing_g.append(g)
            permutation_op = torch.diag(torch.ones(number_of_atoms))[permutation, :]
            list_permutations.append(permutation_op)
            print("Found stabilizing operation.")
        except:
            continue

    # Confirm that the equilibrium coordinates are indeed left unchanged.
    for g, p in zip(list_stabilizing_g, list_permutations):
        x = map_relative_coordinates_to_unit_cell(equilibrium_relative_coordinates @ g.transpose(1, 0))
        error = torch.linalg.norm(equilibrium_relative_coordinates - p @ x)
        torch.testing.assert_close(error, torch.tensor(0.0))


    # build operators in the flat number_of_atoms x spatial_dimension space
    list_op = []
    for g, p in zip(list_stabilizing_g, list_permutations):
        flat_g = torch.block_diag(*(number_of_atoms * [g]))
        flat_p = torch.zeros(number_of_atoms * spatial_dimension, number_of_atoms * spatial_dimension)
        for i in range(number_of_atoms):
            for j in range(number_of_atoms):
                if p[i, j] == 1:
                    for k in range(spatial_dimension):
                        flat_p[spatial_dimension * i + k, spatial_dimension * j + k] = 1.

        op = flat_p @ flat_g
        list_op.append(op)

    # Double check that positions are left invariant
    x0 = einops.rearrange(equilibrium_relative_coordinates, "n d -> (n d)")
    for op in list_op:
        x = map_relative_coordinates_to_unit_cell(op @ x0)
        torch.testing.assert_close(torch.norm(x - x0), torch.tensor(0.))

    # Double check that the operators form a group
    # Inverse is present
    for op in list_op:
        inv_op = op.transpose(1, 0)

        found = False
        for check_op in list_op:
            if torch.linalg.norm(inv_op - check_op) < 1e-8:
                found = True
        assert found

    # closed to product
    for op1 in list_op:
        for op2 in list_op:
            new_op = op1 @ op2

            found = False
            for op3 in list_op:
                if torch.linalg.norm(new_op - op3) < 1e-8:
                    found = True
            assert found


    times = torch.ones(nsym).unsqueeze(-1)
    logger.info("Loading checkpoint...")
    pl_model = PositionDiffusionLightningModel.load_from_checkpoint(checkpoint_path)
    pl_model.eval()

    sigma_normalized_score_network = pl_model.sigma_normalized_score_network

    for parameter in sigma_normalized_score_network.parameters():
        parameter.requires_grad_(False)

    basis_vectors = torch.diag(torch.tensor([acell, acell, acell])).to(device)

    normalized_score_function = get_normalized_score_function(
        noise_parameters=noise_parameters,
        sigma_normalized_score_network=sigma_normalized_score_network,
        basis_vectors=basis_vectors)


    normalized_scores = normalized_score_function(relative_coordinates, times)

    batch_hessian_function = jacrev(normalized_score_function, argnums=0)

    hessian_batch_size = 10
    list_flat_hessians = []
    for x, t in zip(torch.split(relative_coordinates, hessian_batch_size), torch.split(times, hessian_batch_size)):
        batch_hessian = batch_hessian_function(x, t)
        flat_hessian = einops.rearrange(torch.diagonal(batch_hessian, dim1=0, dim2=3),
                                        "n1 s1 n2 s2 b -> b (n1 s1) (n2 s2)")
        list_flat_hessians.append(flat_hessian)

    flat_hessian = torch.concat(list_flat_hessians)


    list_flat_g = []
    for g in list_g:
        flat_g = torch.block_diag(*(number_of_atoms * [g]))
        list_flat_g.append(flat_g)

    list_flat_g = torch.stack(list_flat_g).to(device)

    identity_id = 7 # by inspection

    x0 = relative_coordinates[identity_id]
    h0 = flat_hessian[identity_id]

    list_errors = []
    for h, g in zip(flat_hessian, list_flat_g):
        new_h = (g.transpose(1, 0) @ h) @ g
        error = (new_h - h0).abs().max()

        list_errors.append(error)

    print(torch.tensor(list_errors).max())

    n = number_of_atoms * spatial_dimension
    random_h = torch.zeros(n, n).to(device)

    for g in list_flat_g:
        r = torch.rand(n, n).to(device)
        r = 0.5 * (r + r.transpose(1, 0))

        random_h += (g.transpose(1, 0) @ r) @ g

    random_h  = random_h / len(list_g)
