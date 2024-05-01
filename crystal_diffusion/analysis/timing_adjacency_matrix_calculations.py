import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from crystal_diffusion import ANALYSIS_RESULTS_DIR
from crystal_diffusion.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH
from crystal_diffusion.utils.logging_utils import setup_analysis_logger
from crystal_diffusion.utils.neighbors import \
    get_periodic_neighbor_indices_and_displacements

plt.style.use(PLOT_STYLE_PATH)

logger = logging.getLogger(__name__)


def create_basis_vectors(batch_size, min_dimension, max_dimension):
    """Create random basis vectors."""
    delta = max_dimension - min_dimension
    # orthogonal boxes with dimensions between min_dimension and max_dimension.
    orthogonal_boxes = torch.stack([torch.diag(min_dimension + delta * torch.rand(3)) for _ in range(batch_size)])
    # add a bit of noise to make the vectors not quite orthogonal
    basis_vectors = orthogonal_boxes + 0.1 * torch.randn(batch_size, 3, 3)
    return basis_vectors


batch_size = 64
powers = np.arange(3, 13)

min_dimension = 5
max_dimension = 8

dimension_increasing_factor = 2**(1. / 3.)  # this keeps the mean volume per atom roughly constant.

cutoff = 4.
if __name__ == '__main__':
    setup_analysis_logger()

    if torch.cuda.is_available():
        device = 'GPU'
    else:
        device = 'CPU'

    for _ in range(2):
        # A first pass for KeOps compilation
        torch.manual_seed(1234)
        list_natoms = []
        list_timing = []
        for it, power in enumerate(powers, 1):
            natom = 2 ** power
            list_natoms.append(natom)
            logging.info(f"Doing {natom} atoms ...")
            basis_vectors = create_basis_vectors(batch_size, min_dimension, max_dimension)
            min_dimension = dimension_increasing_factor * min_dimension
            max_dimension = dimension_increasing_factor * max_dimension

            relative_coordinates = torch.rand(batch_size, natom, 3)

            t1 = time.time()
            _ = get_periodic_neighbor_indices_and_displacements(relative_coordinates, basis_vectors, cutoff)
            t2 = time.time()
            dt = t2 - t1
            list_timing.append(t2 - t1)
            logging.info(f"  - It took {dt: 8.4e} seconds to generate A ({dt/batch_size: 8.4e} sec/structure)")

        list_natoms = np.array(list_natoms)
        list_relative_timing = np.array(list_timing) / batch_size

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)
    fig.suptitle(f"Adjacency Computation Time on {device} per Structure\n"
                 f"Random Coordinates, Batch Size of {batch_size}")
    ax = fig.add_subplot(111)
    ax.set_xlabel("Number of Atoms")
    ax.set_ylabel("Time")
    ax.loglog(list_natoms, list_relative_timing, 'y-o', alpha=0.5, label='A Calculation Times')
    ax.legend(loc=0)
    fig.tight_layout()
    fig.savefig(ANALYSIS_RESULTS_DIR.joinpath(f"timing_tests_{device}_adjacency_matrix.png"))
