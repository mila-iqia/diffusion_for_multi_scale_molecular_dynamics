import dataclasses
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pymatgen
from flare.bffs.sgp import SGP_Wrapper
from flare.bffs.sgp._C_flare import B2, NormalizedDotProduct
from flare.bffs.sgp.calculator import SGP_Calculator
from flare.utils import NumpyEncoder
from scipy.optimize import OptimizeResult

from diffusion_for_multi_scale_molecular_dynamics.io.lammps.inputs import \
    sort_symbols_by_atomic_mass
from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.flare import \
    FlarePotential
from diffusion_for_multi_scale_molecular_dynamics.mlip.base_mlip_trainer import \
    BaseMLIPTrainer
from diffusion_for_multi_scale_molecular_dynamics.mlip.flare.flare_configuration import \
    FlareConfiguration
from diffusion_for_multi_scale_molecular_dynamics.mlip.flare.flare_hyperparameter_optimizer import \
    FlareHyperparametersOptimizer
from diffusion_for_multi_scale_molecular_dynamics.oracle.base_single_point_calculator import \
    SinglePointCalculation  # noqa


class FlareTrainer(BaseMLIPTrainer):
    """Flare Trainer.

    This class wraps around the  sparse GP in order to only expose the needed methods.
    """

    def __init__(self, flare_configuration: FlareConfiguration, training_database=None):
        """Init method."""
        super().__init__(training_database)
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
                                     single_atom_energies=None)

    def _add_labelled_structure_to_model(self, single_point_calculation: SinglePointCalculation,
                                         active_environment_indices: List[int]):
        """Add labelled structure.

        Add to the sparse Gaussian Process (SGP) database.

        Args:
            single_point_calculation: ground truth single-point calculation.
            active_environment_indices: which atomic environment should be added to the SGP active set.
        """
        assert single_point_calculation.uncertainties is None, \
            "Uncertainties are not None! Only ground truth single-point calculation is supported should be added."

        self.sgp_model.update_db(structure=single_point_calculation.atoms,
                                 forces=single_point_calculation.forces,
                                 energy=single_point_calculation.energy,
                                 mode="specific",
                                 custom_range=list(active_environment_indices)
                                 )

    def fit(self):
        """Fit the sparse GP (recompute its predictive coefficients) for the current data and hyperparameters."""
        self.sgp_model.sparse_gp.update_matrices_QR()

    def _get_species_numbers_map(self, list_element_symbols: List[str]) -> Dict[int, int]:
        """Get a map where the key is the atomic number and the value is the integer label."""
        species_numbers_map = dict()

        sorted_elements = sort_symbols_by_atomic_mass(list_element_symbols)

        for idx, (symbol, _) in enumerate(sorted_elements):
            species_numbers_map[pymatgen.core.Element(symbol).number] = idx
        return species_numbers_map

    def fit_hyperparameters(self, optimizer: FlareHyperparametersOptimizer) -> Tuple[OptimizeResult, pd.DataFrame]:
        """Fit hyperparameters.

        This method drives the selection of the sparse GP's hyperparameters, namely the various
        "sigma" parameters.

        Args:
            optimizer: FlareHyperparametersOptimizer instance.

        Returns:
            optimization_result: the scipy.minimize result object from the HP fitting process.
            history_df: a dataframe containing the negative log likelihood and the various sigma values
                during the optimization iterative process.
        """
        optimization_result, history_df = optimizer.train(self.sgp_model)
        return optimization_result, history_df

    def write_checkpoint(self, output_directory: Path) -> FlarePotential:
        """Write the SGP checkpoint and the mapped LAMMPS files into output_directory; return the potential."""
        output_directory.mkdir(parents=True, exist_ok=True)
        # Write the SGP checkpoint before mapping: build_map mutates the model, so writing first keeps the
        # checkpoint reload-idempotent.
        self._write_sgp_checkpoint(output_directory / "checkpoint.json")
        return self._write_mapped_potential(output_directory)

    def _write_sgp_checkpoint(self, checkpoint_path: Path):
        """Write the sparse GP state as a json checkpoint."""
        sgp_dict = self.sgp_model.as_dict()
        checkpoint_dict = dict(flare_configuration=dataclasses.asdict(self.flare_configuration),
                               sgp_dict=sgp_dict)
        with open(str(checkpoint_path), "w") as fd:
            json.dump(checkpoint_dict, fd, cls=NumpyEncoder)

    def _write_mapped_potential(self, output_directory: Path) -> FlarePotential:
        """Map the sparse GP to the LAMMPS coefficient files and return the matching FLARE potential."""
        pair_coeff_filename = "lmp.flare"
        mapped_uncertainty_filename = f"map_unc_{pair_coeff_filename}"
        SGP_Calculator(self.sgp_model, use_mapping=True).build_map(filename=pair_coeff_filename,
                                                                   contributor="Generated by FlareTrainer",
                                                                   map_uncertainty=True)
        pair_coeff_file_path = output_directory / pair_coeff_filename
        mapped_uncertainty_file_path = output_directory / mapped_uncertainty_filename

        list_src = [pair_coeff_filename, mapped_uncertainty_filename]
        list_dst = [pair_coeff_file_path, mapped_uncertainty_file_path]

        for src, dst in zip(list_src, list_dst):
            shutil.move(src, str(dst))

        return FlarePotential(pair_coeff_file_path=pair_coeff_file_path,
                              mapped_uncertainty_file_path=mapped_uncertainty_file_path)

    @classmethod
    def load_checkpoint(cls, checkpoint_path: Path, training_database=None):
        """Instantiate a flare trainer from a checkpoint file."""
        with open(str(checkpoint_path), "r") as fd:
            checkpoint_dict = json.loads(fd.readline())

        flare_configuration = FlareConfiguration(**checkpoint_dict["flare_configuration"])

        sgp_dict = checkpoint_dict["sgp_dict"]

        sgp_model, kernels = SGP_Wrapper.from_dict(sgp_dict)

        flare_trainer = cls(flare_configuration=flare_configuration, training_database=training_database)

        # Overload internals with what was read from disk.
        flare_trainer.sgp_model = sgp_model
        flare_trainer._dot_product_kernel = kernels[0]
        flare_trainer._descriptor_calculators = sgp_model.descriptor_calculators
        flare_trainer._B2_descriptor = flare_trainer._descriptor_calculators[0]

        return flare_trainer
