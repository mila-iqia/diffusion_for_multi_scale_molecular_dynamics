import logging

import torch

from diffusion_for_multi_scale_molecular_dynamics.generators.axl_generator import (
    AXLGenerator, SamplingParameters)
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, AXL_COMPOSITION, CARTESIAN_POSITIONS)
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import (
    get_positions_from_coordinates,
    map_lattice_parameters_to_unit_cell_vectors)

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

    if sampling_parameters.sample_batchsize is None:
        sample_batch_size = number_of_samples
    else:
        sample_batch_size = sampling_parameters.sample_batchsize

    list_sampled_relative_coordinates = []
    list_sampled_atom_types = []
    list_sampled_lattice_vectors = []
    for sampling_batch_indices in torch.split(
        torch.arange(number_of_samples), sample_batch_size
    ):
        sampled_axl = generator.sample(len(sampling_batch_indices), device=device)
        list_sampled_atom_types.append(sampled_axl.A)
        list_sampled_relative_coordinates.append(sampled_axl.X)
        list_sampled_lattice_vectors.append(sampled_axl.L)

    atom_types = torch.concat(list_sampled_atom_types)
    relative_coordinates = torch.concat(list_sampled_relative_coordinates)
    lattice_vectors = torch.concat(list_sampled_lattice_vectors)
    axl_composition = AXL(
        A=atom_types,
        X=relative_coordinates,
        L=lattice_vectors,
    )

    basis_vectors = map_lattice_parameters_to_unit_cell_vectors(lattice_vectors)
    cartesian_positions = get_positions_from_coordinates(
        relative_coordinates, basis_vectors
    )

    batch = {
        CARTESIAN_POSITIONS: cartesian_positions,
        AXL_COMPOSITION: axl_composition,
    }

    return batch
