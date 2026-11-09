"""Configuration dataclass defining a Moment Tensor Potential."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from pymatgen.core import Element


@dataclass(kw_only=True)
class MtpConfiguration:
    """A Moment Tensor Potential model, trained with MLIP-3 and run through the lammps-mtp-kokkos interface."""

    elements: list[str]

    # Inputs that define the model.
    level: int
    max_dist: float

    # Read back from the template/fitted potential (the level fixes these).
    radial_basis_size: Optional[int] = None
    radial_funcs_count: Optional[int] = None
    alpha_scalar_moments: Optional[int] = None
    species_count: Optional[int] = None

    energy_weight: float = 0.0
    force_weight: float = 0.0
    stress_weight: float = 0.0
    site_en_weight: float = 1.0

    # Parameters passed to the MLIP-3 'mlp train' command.
    training_params: dict = field(
        default_factory=lambda: dict(max_iter=1000, init_params="same", scale_by_force=0.0, bfgs_conv_tol=1e-3)
    )

    # The parameters read from an MTP file and their python type.
    _FILE_PARAMETERS = dict(species_count=int, radial_basis_size=int,
                            radial_funcs_count=int, alpha_scalar_moments=int)

    @property
    def number_of_adjustable_parameters(self) -> int:
        """The number of adjustable MTP parameters."""
        return self.radial_basis_size + self.alpha_scalar_moments + self.species_count

    @classmethod
    def _parse_header(cls, mtp_file_path: Path) -> Dict:
        """Parse the readable header of an MTP file into a dict of the _FILE_PARAMETERS values.

        The MTP file mixes a readable text header with binary data, so it is parsed line by line.
        """
        found = {}
        with open(mtp_file_path, "rb") as file_descriptor:
            for raw_line in file_descriptor:
                key, separator, value = raw_line.decode("latin-1").partition("=")
                key = key.strip()
                if separator and key in cls._FILE_PARAMETERS and key not in found:
                    found[key] = cls._FILE_PARAMETERS[key](value.strip())
                    if len(found) == len(cls._FILE_PARAMETERS):
                        break
        return found

    def read_from_file(self, mtp_file_path: Path) -> None:
        """Read the level-determined parameters back from an MTP file into the configuration."""
        found = self._parse_header(mtp_file_path)
        self.species_count = found["species_count"]
        self.radial_basis_size = found["radial_basis_size"]
        self.radial_funcs_count = found["radial_funcs_count"]
        self.alpha_scalar_moments = found["alpha_scalar_moments"]

    def read_descriptors(self, mtp_file_path: Path) -> Dict[str, int]:
        """Return the MTP basis descriptors that fix the size of the model's coefficient space.

        The basis sizes (radial_basis_size, radial_funcs_count, alpha_scalar_moments) are level-determined and
        read from the (single-species) level template; the species_count comes from the configured elements
        (the templates always report 1), since the radial term scales as species_count squared.
        """
        header = self._parse_header(mtp_file_path)
        return dict(
            species_count=len(self.elements),
            radial_basis_size=header["radial_basis_size"],
            radial_funcs_count=header["radial_funcs_count"],
            alpha_scalar_moments=header["alpha_scalar_moments"],
        )

    def write_to_file(self, mtp_file_path: Path) -> None:
        """Write the configuration's max_dist into an MTP file, leaving the rest (including binary) untouched."""
        with open(mtp_file_path, "rb") as file_descriptor:
            lines = file_descriptor.readlines()

        for index, raw_line in enumerate(lines):
            decoded_line = raw_line.decode("latin-1")
            key, separator, _ = decoded_line.partition("=")
            if separator and key.strip() == "max_dist":
                leading_whitespace = decoded_line[: len(decoded_line) - len(decoded_line.lstrip())]
                lines[index] = f"{leading_whitespace}max_dist = {self.max_dist}\n".encode("latin-1")
                break

        with open(mtp_file_path, "wb") as file_descriptor:
            file_descriptor.writelines(lines)

    def __post_init__(self):
        """Validate the configuration."""
        if len(self.elements) == 0:
            raise ValueError("The list of elements should not be empty.")
        if len(set(self.elements)) != len(self.elements):
            raise ValueError("The elements are not unique!")
        for element in self.elements:
            try:
                Element(element)
            except Exception:
                raise ValueError(f"Expected real elements; got '{element}'.")

        if self.level <= 0:
            raise ValueError("The MTP level should be positive.")
        if self.max_dist <= 0.0:
            raise ValueError("The maximum distance (cutoff) should be positive.")

        weights = dict(energy_weight=self.energy_weight, force_weight=self.force_weight,
                       stress_weight=self.stress_weight, site_en_weight=self.site_en_weight)
        for weight_name, weight_value in weights.items():
            if weight_value < 0.0:
                raise ValueError(f"The {weight_name} should be non-negative.")
