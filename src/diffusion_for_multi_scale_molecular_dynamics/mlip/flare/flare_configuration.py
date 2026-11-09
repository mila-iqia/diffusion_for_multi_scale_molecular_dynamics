"""Configuration dataclass defining a FLARE sparse Gaussian process."""

from dataclasses import dataclass

from pymatgen.core import Element


@dataclass(kw_only=True)
class FlareConfiguration:
    """Flare Configuration.

    Various parameters defining the FLARE sparce Gaussian process.
    """

    cutoff: float  # neighbor cutoff, in Angstrom

    elements: list[str]  # the elements that can exist.
    n_radial: int  # Number of radial basis functions for the ACE embedding
    lmax: int  # Largest L included in spherical harmonics for the ACE embedding
    variance_type: str

    # Define the initial GP hyperparameters
    initial_sigma: float = 1.00
    initial_sigma_e: float = 1e-2
    initial_sigma_f: float = 1e-3
    initial_sigma_s: float = 1e-1

    def __post_init__(self):
        """Post init."""
        assert self.cutoff > 0.0, "The cutoff should be positive."
        assert len(self.elements) > 0, "The number of elements should be positive."
        assert self.n_radial > 0, "The number of radial basis should be positive."
        assert self.lmax > 0, "The highest angular momentum channel should be positive."

        assert self.variance_type == 'local' or self.variance_type == 'DTC', \
            f"Only 'local' and 'DTC' variance are supported. Got '{self.variance_type}'."

        assert len(set(self.elements)) == len(self.elements), "The elements are not unique!"

        for element in self.elements:
            try:
                Element(element)
            except Exception:
                raise ValueError(f"Expected real elements; got '{element}'.")
