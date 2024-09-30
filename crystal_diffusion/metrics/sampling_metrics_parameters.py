from dataclasses import dataclass


@dataclass(kw_only=True)
class SamplingMetricsParameters:
    """Sampling metrics parameters.

    This dataclass configures what metrics should be computed given that samples have
    been generated.
    """
    compute_energies: bool = False  # should the energies be computed
    compute_structure_factor: bool = False  # should the structure factor (distances distribution) be recorded
    structure_factor_max_distance: float = 10.0  # cutoff for the structure factor
