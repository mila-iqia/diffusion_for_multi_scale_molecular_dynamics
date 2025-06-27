from pathlib import Path
from typing import Optional

import numpy as np
from flare.bffs.sgp import SGP_Wrapper
from flare.bffs.sgp.calculator import SGP_Calculator
from pymatgen.core import Structure

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.base_single_point_calculator import (  # noqa
    BaseSinglePointCalculator, SinglePointCalculation)


class FlareSinglePointCalculator(BaseSinglePointCalculator):
    """Wrapper around the internal FLARE calculator class."""

    def __init__(self, sgp_model: SGP_Wrapper):
        """Init method."""
        super().__init__(self)
        self._calculation_type = "flare_sgp"
        self._flare_calculator = SGP_Calculator(sgp_model)
        self._calculation_properties = ["energy", "forces", "stds"]

        self._uncertainty_is_energy = None

        match sgp_model.variance_type:
            case "local":
                self._uncertainty_is_energy = True
            case "DTC":
                self._uncertainty_is_energy = False
            case _:
                raise NotImplementedError("Only local and DTC variance types are implemented. Review input.")

    def calculate(self, structure: Structure, results_path: Optional[Path] = None) -> SinglePointCalculation:
        """Calculate.

        Drive the sparse Gaussian Process calculation.

        Args:
            structure: pymatgen structure.
            results_path: Should be None

        Returns:
            calculation_results: the calculation result.
        """
        assert results_path is None, "The FLARE model has no file results artifact."
        atoms = structure.to_ase_atoms()
        self._flare_calculator.calculate(atoms=atoms, properties=self._calculation_properties)

        energy = self._flare_calculator.results["energy"]
        forces = self._flare_calculator.results["forces"]

        # FLARE's code is convoluted, so it is hard to know exactly what is being computed.
        # Scanning the code flare.bffs.sgp.calculator.SGP_Calculator.predict_on_structure,
        # it seems that the 'stds' array is of the same dimensions as 'forces'. It contains
        # force uncertainty if variance_type = 'DTC', and local energy uncertainty if variance_type = 'local',
        # shoved in the first column.
        flare_stds = self._flare_calculator.results["stds"]

        if self._uncertainty_is_energy:
            # FLARE's code normalizes this to sigma internally. The energy uncertainty is unitless.
            uncertainties = flare_stds[:, 0]
        else:
            # Let's compute the norm of the 'force uncertainties' to get a scalar uncertainty.
            uncertainties = np.linalg.norm(flare_stds, axis=1)

        return SinglePointCalculation(calculation_type=self._calculation_type,
                                      structure=structure,
                                      energy=energy,
                                      forces=forces,
                                      uncertainties=uncertainties)
