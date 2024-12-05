from diffusion_for_multi_scale_molecular_dynamics.loss.coordinates_loss_calculator import \
    CoordinatesLossCalculator


class LatticeLossCalculator(CoordinatesLossCalculator):
    """Class to calculate the loss for the lattice vectors.

    The loss for the lattice parameters is the same as for the coordinates. We simply inherit from the coordinates loss.
    This is an empty shell for now - we could revisit this to create a different loss for the lattice parameters.
    """

    def __init__(self):
        """Placeholder for now."""
        super().__init__()
