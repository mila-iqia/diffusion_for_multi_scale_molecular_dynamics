from diffusion_for_multi_scale_molecular_dynamics.loss.atom_type_loss_calculator import \
    D3PMLossCalculator
from diffusion_for_multi_scale_molecular_dynamics.loss.coordinates_loss_calculator import (
    MSELossCalculator, WeightedMSELossCalculator)
from diffusion_for_multi_scale_molecular_dynamics.loss.lattice_loss_calculator import \
    LatticeLossCalculator
from diffusion_for_multi_scale_molecular_dynamics.loss.loss_parameters import \
    LossParameters
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL

LOSS_BY_ALGO = dict(mse=MSELossCalculator, weighted_mse=WeightedMSELossCalculator)


def create_loss_calculator(loss_parameters: LossParameters) -> AXL:
    """Create Loss Calculator.

    This is a factory method to create the loss calculator.

    Args:
        loss_parameters : parameters defining the loss.

    Returns:
        loss_calculator : the loss calculator for atom types, coordinates, lattice in an AXL namedtuple.
    """
    algorithm = loss_parameters.coordinates_algorithm
    assert (
        algorithm in LOSS_BY_ALGO.keys()
    ), f"Algorithm {algorithm} is not implemented. Possible choices are {LOSS_BY_ALGO.keys()}"

    coordinates_loss = LOSS_BY_ALGO[algorithm](loss_parameters)
    lattice_loss = LatticeLossCalculator  # TODO placeholder
    atom_loss = D3PMLossCalculator(loss_parameters)

    return AXL(
        A=atom_loss,
        X=coordinates_loss,
        L=lattice_loss,
    )
