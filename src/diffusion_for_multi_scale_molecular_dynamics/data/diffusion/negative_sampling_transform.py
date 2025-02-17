from typing import Dict

import torch

from diffusion_for_multi_scale_molecular_dynamics.namespace import \
    RELATIVE_COORDINATES
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_relative_coordinates_to_unit_cell


class NegativeSamplingTransform:
    """Transform an atomic configuration into one with 2 atoms on top of each other.."""

    def __init__(self, short_distance: float = 0.001):
        """Negative sampling transform.

        This class creates a method that takes in a batch of dataset data and
        replaces the atom coordinates with a new configuration where two atoms are on top of each other.

        Args:
            spatial_dimension: dimension of space.
        """
        super().__init__()

        assert (
            short_distance < 0.5
        ), f"The distance remaining between two atoms in negative samples should be small. Got {short_distance}."
        self.short_distance = short_distance

    def get_small_displacement(self, num_samples: int, spatial_dimension: int):
        return torch.randn((num_samples, spatial_dimension)) * self.short_distance

    def transform(self, batch: Dict) -> Dict:
        """Transform.

        This method updates the RELATIVE_COORDINATES entry.

        Args:
            batch: dataset data.

        Returns:
            augmented_batch: batch augmented with noised data for score matching.
        """
        assert (
            RELATIVE_COORDINATES in batch
        ), f"The field '{RELATIVE_COORDINATES}' is missing from the input."

        x0 = batch[RELATIVE_COORDINATES]
        shape = x0.shape
        assert len(shape) == 3, (
            f"the shape of the RELATIVE_COORDINATES array should be [batch_size, number_of_atoms, spatial_dimensions]. "
            f"Got shape = {shape}."
        )
        batch_size, n_atoms, spatial_dimension = shape

        assert (
            n_atoms >= 2
        ), f"Need at least 2 atoms to put them on top of each other. Got {n_atoms}."

        batch_idx = torch.arange(batch_size)  # useful for selecting coordinates later

        # select one atom to move in each sample
        moved_atoms_idx = torch.randint(0, n_atoms, (batch_size,))
        # and select a different atom to move it too
        destination_idx = (
            moved_atoms_idx + torch.randint(1, n_atoms, (batch_size,))
        ) % n_atoms

        # the moved atoms should land at the destination plus a small displacement
        small_displacement = self.get_small_displacement(batch_size, spatial_dimension)
        # get the destination coordinates
        destination_positions = x0[
            batch_idx, destination_idx, :
        ]  # batch, spatial_dimension tensor
        # shift those positions by small_displacement - remembering to map back to [0, 1)
        destination_positions = map_relative_coordinates_to_unit_cell(
            destination_positions + small_displacement
        )

        # replace the moved atoms coordinates by the new ones
        x0[batch_idx, moved_atoms_idx] = destination_positions

        batch[RELATIVE_COORDINATES] = x0

        return batch


class RandomizePositionTransform:
    """Transform an atomic configuration into one with 2 atoms on top of each other."""

    def __init__(self):
        return None

    def transform(self, batch: Dict) -> Dict:
        """Transform.

        This method updates the RELATIVE_COORDINATES entry and randomize the values.

        Args:
            batch: dataset data.

        Returns:
            augmented_batch: batch augmented with noised data for score matching.
        """
        assert (
            RELATIVE_COORDINATES in batch
        ), f"The field '{RELATIVE_COORDINATES}' is missing from the input."

        x0 = batch[RELATIVE_COORDINATES]
        new_x0 = torch.rand_like(x0)

        batch[RELATIVE_COORDINATES] = new_x0

        return batch
