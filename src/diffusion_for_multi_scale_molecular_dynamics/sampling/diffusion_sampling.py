import logging

import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.axl_generator import (
    AXLGenerator, SamplingParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    CARTESIAN_POSITIONS, RELATIVE_COORDINATES, UNIT_CELL)
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    get_positions_from_coordinates
from diffusion_for_multi_scale_molecular_dynamics.utils.structure_utils import \
    get_orthogonal_basis_vectors

logger = logging.getLogger(__name__)


def create_batch_of_samples(
    generator: AXLGenerator,
    sampling_parameters: SamplingParameters,
    device: torch.device,
):
    """Create batch of samples.

    Utility function to drive the generation of samples.

    Args:
        generator : AXL generator.
        sampling_parameters : parameters defining how to sample.
        device: device where the generator is located.

    Returns:
        sample_batch: drawn samples in the same dictionary format as the training data.
    """
    logger.info("Creating a batch of samples")
    number_of_samples = sampling_parameters.number_of_samples
    cell_dimensions = sampling_parameters.cell_dimensions
    basis_vectors = get_orthogonal_basis_vectors(number_of_samples, cell_dimensions).to(
        device
    )

    if sampling_parameters.sample_batchsize is None:
        sample_batch_size = number_of_samples
    else:
        sample_batch_size = sampling_parameters.sample_batchsize

    list_sampled_relative_coordinates = []
    for sampling_batch_indices in torch.split(
        torch.arange(number_of_samples), sample_batch_size
    ):
        basis_vectors_ = basis_vectors[sampling_batch_indices]
        sampled_relative_coordinates = generator.sample(
            len(sampling_batch_indices), unit_cell=basis_vectors_, device=device
        )
        list_sampled_relative_coordinates.append(sampled_relative_coordinates)

    relative_coordinates = torch.concat(list_sampled_relative_coordinates)
    cartesian_positions = get_positions_from_coordinates(
        relative_coordinates, basis_vectors
    )

    batch = {
        CARTESIAN_POSITIONS: cartesian_positions,
        RELATIVE_COORDINATES: relative_coordinates,
        UNIT_CELL: basis_vectors,
    }

    return batch
