"""Utilities.

This module contains helper functions for the notebook tutorials.
"""
import einops
import torch


def compute_total_distance(relative_coordinates: torch.Tensor, reference_relative_coordinates: torch.Tensor) -> float:
    """Compute total distance.

    This method computes the "total distance" between two configurations, accounting for periodicity,
    by comparing coordinates in order.

    Args:
        relative_coordinates: the relative coordinates of a configuration.
            Dimension [number_of_atoms, spatial_dimension]
        reference_relative_coordinates: the reference relative coordinates of a configuration.
            Dimension [number_of_atoms, spatial_dimension]

    Returns:
        Total distance: the total distance between the relative coordinates and the reference, in reduced units.
    """
    raw_displacements = relative_coordinates - reference_relative_coordinates
    augmented_displacements = [raw_displacements - 1.0, raw_displacements, raw_displacements + 1.0]

    squared_displacements = einops.rearrange(augmented_displacements, "c n d -> (n d) c")**2

    total_displacement = torch.sqrt(squared_displacements.min(dim=1).values.sum())
    return total_displacement.item()
