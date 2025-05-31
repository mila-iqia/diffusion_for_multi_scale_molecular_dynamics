import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pymatgen
from flare.bffs.sgp import SGP_Wrapper
from flare.bffs.sgp.calculator import SGP_Calculator
from flare_pp import B2, NormalizedDotProduct

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.ordered_elements import \
    sort_elements_by_atomic_mass
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.base_single_point_calculator import \
    SinglePointCalculation  # noqa


@dataclass(kw_only=True)
class FlareConfiguration:
    """Flare Configuration.

    Various parameters defining the FLARE sparce Gaussian process.
    """
    cutoff: float  # neighbor cutoff, in Angstrom

    elements: list[str]  # the elements that can exist.
    n_radial: int  # Number of radial basis functions for the ACE embedding
    lmax: int   # Largest L included in spherical harmonics for the ACE embedding
    variance_type: str

    # Define the initial GP hyperparameters
    initial_sigma_e: float = 0.01
    initial_sigma_f: float = 0.001
    initial_sigma_s: float = 0.1

    # Maximum number of iterations when optimizing the hyperparameters with BFGS.
    max_bfgs_iterations: int = 100

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
                pymatgen.core.Element(element)
            except Exception:
                raise ValueError(f"Expected real elements; got '{element}'.")


class FlareTrainer:
    """Flare Trainer.

    This class wraps around the  sparse GP in order to only expose the needed methods.
    """

    def __init__(self, flare_configuration: FlareConfiguration):
        """Init method."""
        # We will be very opinionated about certain options.
        n_species = len(flare_configuration.elements)
        species_numbers_map = self._get_species_numbers_map(flare_configuration.elements)

        radial_basis = "chebyshev"  # Radial basis set
        cutoff_name = "quadratic"  # Cutoff function
        radial_hyps = [0, flare_configuration.cutoff]
        cutoff_hyps = []
        descriptor_settings = [n_species,
                               flare_configuration.n_radial,
                               flare_configuration.lmax]

        # Define a B2 object. This object must be long-lived, it must not get out of scope!
        self._B2_descriptor = B2(radial_basis, cutoff_name, radial_hyps, cutoff_hyps, descriptor_settings)

        # The GP class can take a list of descriptors as input, but here we'll use a single descriptor.
        self._descriptor_calculators = [self._B2_descriptor]

        # Define kernel function.
        sigma = 1.0
        power = 2
        self._dot_product_kernel = NormalizedDotProduct(sigma, power)

        # TODO: Consider using the field 'single_atom_energies' if and when we do more serious DFT calculations.
        # The wrapper does not make internal copies of the various input C++ objects like B2, etc...
        # These objects must not get garbage collected; otherwise we get mysterious segfaults.
        self.sgp_model = SGP_Wrapper(kernels=[self._dot_product_kernel],
                                     descriptor_calculators=self._descriptor_calculators,
                                     cutoff=flare_configuration.cutoff,
                                     sigma_e=flare_configuration.initial_sigma_e,
                                     sigma_f=flare_configuration.initial_sigma_f,
                                     sigma_s=flare_configuration.initial_sigma_s,
                                     species_map=species_numbers_map,
                                     variance_type=flare_configuration.variance_type,
                                     energy_training=True,
                                     force_training=True,
                                     stress_training=False,
                                     single_atom_energies=None,
                                     max_iterations=flare_configuration.max_bfgs_iterations,
                                     opt_method="BFGS")

    def add_labelled_structure(self, single_point_calculation: SinglePointCalculation,
                               active_environment_indices: List[int]):
        """Add labelled structure.

        Add to the sparse Gaussian Process (SGP) database.

        Args:
            single_point_calculation: ground truth single-point calculation.
            active_environment_indices: which atomic environment should be added to the SGP active set.
        """
        assert single_point_calculation.uncertainties is None, \
            "Uncertainties are not None! Only ground truth single-point calculation is supported should be added."

        self.sgp_model.update_db(structure=single_point_calculation.structure.to_ase_atoms(),
                                 forces=single_point_calculation.forces,
                                 energy=single_point_calculation.energy,
                                 mode="specific",
                                 custom_range=list(active_environment_indices)
                                 )

    def _get_species_numbers_map(self, list_elements: List[str]) -> Dict[int, int]:
        """Get a map where the key is the atomic number and the value is the integer label."""
        species_numbers_map = dict()
        for idx, symbol in enumerate(sort_elements_by_atomic_mass(list_elements)):
            element = pymatgen.core.Element(symbol)
            species_numbers_map[element.number] = idx
        return species_numbers_map

    def fit_hyperparameters(self):
        """Fit hyperparameters."""
        self.sgp_model.train()

    def write_mapped_model_to_disk(self, mapped_coefficients_directory: Path, version: int):
        """Write mapped model to disk."""
        coeff_filename = f"lmp{version}.flare"
        uncertainty_filename = f"map_unc_{coeff_filename}"
        SGP_Calculator(self.sgp_model, use_mapping=True).build_map(filename=coeff_filename,
                                                                   contributor="Version 0",
                                                                   map_uncertainty=True)
        mapped_coefficients_directory.mkdir(parents=True, exist_ok=True)

        for filename in [coeff_filename, uncertainty_filename]:
            shutil.move(filename, str(mapped_coefficients_directory / filename))
