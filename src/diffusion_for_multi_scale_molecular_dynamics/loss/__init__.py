from diffusion_for_multi_scale_molecular_dynamics.loss.atom_type_loss_calculator import \
    D3PMLossCalculator
from diffusion_for_multi_scale_molecular_dynamics.loss.coordinates_loss_calculator import (
    MSELossCalculator, WeightedMSELossCalculator)
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL

LOSS_BY_ALGO = dict(mse=MSELossCalculator, weighted_mse=WeightedMSELossCalculator)


def create_loss_calculator(loss_parameters: AXL) -> AXL:
    """Create Loss Calculator.

    This is a factory method to create the loss calculator.

    Args:
        loss_parameters : parameters defining the loss.

    Returns:
        loss_calculator : the loss calculator for atom types, coordinates, lattice in an AXL namedtuple.
    """
    coordinates_algorithm = loss_parameters.X.algorithm
    assert (
        coordinates_algorithm in LOSS_BY_ALGO.keys()
    ), f"Algorithm {coordinates_algorithm} is not implemented. Possible choices are {LOSS_BY_ALGO.keys()}"

    lattice_algorithm = loss_parameters.L.algorithm
    assert (
        lattice_algorithm in LOSS_BY_ALGO.keys()
    ), f"Algorithm {lattice_algorithm} is not implemented. Possible choices are {LOSS_BY_ALGO.keys()}"

    coordinates_loss = LOSS_BY_ALGO[coordinates_algorithm](loss_parameters.X)
    lattice_loss = LOSS_BY_ALGO[lattice_algorithm](
        loss_parameters.L
    )
    atom_loss = D3PMLossCalculator(loss_parameters.A)

    return AXL(
        A=atom_loss,
        X=coordinates_loss,
        L=lattice_loss,
    )
