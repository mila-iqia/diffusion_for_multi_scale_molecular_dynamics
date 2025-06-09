import dataclasses
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pymatgen
from flare.bffs.sgp import SGP_Wrapper
from flare.bffs.sgp.calculator import SGP_Calculator
from flare.utils import NumpyEncoder
from flare_pp import B2, NormalizedDotProduct
from scipy.optimize import OptimizeResult

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.lammps.inputs import \
    sort_elements_by_atomic_mass
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.base_single_point_calculator import \
    SinglePointCalculation  # noqa
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.trainer.flare_hyperparameter_optimizer import \
    FlareHyperparametersOptimizer


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
    initial_sigma: float = 1.00
    initial_sigma_e: float = 0.01
    initial_sigma_f: float = 0.001
    initial_sigma_s: float = 0.1

    # Maximum number of iterations when optimizing the hyperparameters.
    minimization_method: str = "nelder-mead"
    max_iterations: int = 100

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
        self.flare_configuration = flare_configuration
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
        sigma = self.flare_configuration.initial_sigma
        power = 2
        self._dot_product_kernel = NormalizedDotProduct(sigma, power)

        self._minimization_method = flare_configuration.minimization_method

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
                                     max_iterations=flare_configuration.max_iterations,
                                     opt_method=self._minimization_method)

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

    def _get_species_numbers_map(self, list_element_symbols: List[str]) -> Dict[int, int]:
        """Get a map where the key is the atomic number and the value is the integer label."""
        species_numbers_map = dict()

        list_elements = [pymatgen.core.Element(symbol) for symbol in list_element_symbols]
        list_sorted_elements = sort_elements_by_atomic_mass(list_elements)

        for idx, element in enumerate(list_sorted_elements):
            species_numbers_map[element.number] = idx
        return species_numbers_map

    def fit_hyperparameters(self) -> Tuple[OptimizeResult, pd.DataFrame]:
        """Fit hyperparameters.

        This method drives the selection of the sparse GP's hyperparameters, namely the various
        "sigma" parameters.

        Returns:
            optimization_result: the scipy.minimize result object from the HP fitting process.
            history_df: a dataframe containing the negative log likelihood and the various sigma values
                during the optimization iterative process.
        """
        # We hardcode some sensible decisions here to avoid expositing to many obscure choices to the user.
        minimize_options = {"disp": False,  # 'display': the algorithm shouldn't print to terminal.
                            "ftol": 1e-8,
                            "gtol": 1e-8,
                            "maxiter": self.flare_configuration.max_iterations}
        optimizer = FlareHyperparametersOptimizer(method=self._minimization_method, minimize_options=minimize_options)
        optimization_result, history_df = optimizer.train(self.sgp_model)
        return optimization_result, history_df

    def write_mapped_model_to_disk(self, mapped_coefficients_directory: Path, version: int) -> Tuple[Path, Path]:
        """Write mapped model to disk."""
        pair_coeff_filename = f"lmp{version}.flare"
        mapped_uncertainty_filename = f"map_unc_{pair_coeff_filename}"
        SGP_Calculator(self.sgp_model, use_mapping=True).build_map(filename=pair_coeff_filename,
                                                                   contributor="Generated by FlareTrainer",
                                                                   map_uncertainty=True)
        mapped_coefficients_directory.mkdir(parents=True, exist_ok=True)

        pair_coeff_file_path = mapped_coefficients_directory / pair_coeff_filename
        mapped_uncertainty_file_path = mapped_coefficients_directory / mapped_uncertainty_filename

        list_src = [pair_coeff_filename, mapped_uncertainty_filename]
        list_dst = [pair_coeff_file_path, mapped_uncertainty_file_path]

        for src, dst in zip(list_src, list_dst):
            shutil.move(src, str(dst))

        return pair_coeff_file_path, mapped_uncertainty_file_path

    def write_checkpoint_to_disk(self, checkpoint_path: Path):
        """Write checkpoint to disk."""
        sgp_dict = self.sgp_model.as_dict()
        checkpoint_dict = dict(flare_configuration=dataclasses.asdict(self.flare_configuration),
                               sgp_dict=sgp_dict)
        with open(str(checkpoint_path), "w") as fd:
            json.dump(checkpoint_dict, fd, cls=NumpyEncoder)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path):
        """Instantiate a flare trainer from a checkpoint file."""
        with open(str(checkpoint_path), "r") as fd:
            checkpoint_dict = json.loads(fd.readline())

        flare_configuration = FlareConfiguration(**checkpoint_dict["flare_configuration"])

        sgp_dict = checkpoint_dict["sgp_dict"]

        sgp_model, kernels = SGP_Wrapper.from_dict(sgp_dict)

        flare_trainer = cls(flare_configuration=flare_configuration)

        # Overload internals with what was read from disk.
        flare_trainer.sgp_model = sgp_model
        flare_trainer._dot_product_kernel = kernels[0]
        flare_trainer._descriptor_calculators = sgp_model.descriptor_calculators
        flare_trainer._B2_descriptor = flare_trainer._descriptor_calculators[0]

        return flare_trainer
