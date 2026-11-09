"""Moment Tensor Potential machine-learning interatomic potential."""

from pathlib import Path
from typing import Dict, Optional, Union

import yaml

from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.mtp import \
    MtpPotential
from diffusion_for_multi_scale_molecular_dynamics.mlip.base_mlip import \
    BaseMLIP
from diffusion_for_multi_scale_molecular_dynamics.mlip.mtp.mtp_configuration import \
    MtpConfiguration
from diffusion_for_multi_scale_molecular_dynamics.mlip.mtp.mtp_trainer import \
    MtpTrainer
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_runner import (
    InProcessLammpsRunner, SubprocessLammpsRunner)

LAMMPS_MTP_KOKKOS_URL = "https://github.com/RichardZJM/lammps-mtp-kokkos"


class MtpMlip(BaseMLIP):
    """Moment Tensor Potential MLIP: trained with MLIP-3 and run through the lammps-mtp-kokkos pair_style."""

    name = "MTP"
    training_program_name = "mlp"

    def __init__(
        self,
        mtp_trainer: MtpTrainer,
        lammps_runner: Union[SubprocessLammpsRunner, InProcessLammpsRunner],
    ):
        """Init method.

        Args:
            mtp_trainer: fits the MTP and exports its LAMMPS potential.
            lammps_runner: runner used to evaluate the deployed potential; must provide the mtp/extrapolation
                pair_style.
        """
        super().__init__(trainer=mtp_trainer, lammps_runner=lammps_runner)
        self._check_lammps_dependency()
        self._set_descriptors()

    def _set_descriptors(self) -> None:
        """Read the level-determined MTP basis descriptors from the template into self.descriptors."""
        self.descriptors = self._trainer.configuration.read_descriptors(self._trainer.template_path)

    def _check_lammps_dependency(self) -> None:
        """Fail early if the LAMMPS runner lacks the mtp/extrapolation pair_style from lammps-mtp-kokkos."""
        try:
            self._lammps_runner.check_dependency(section="Pair styles", to_find="mtp/extrapolation")
        except RuntimeError as error:
            raise RuntimeError(
                f"{error} The 'mtp/extrapolation' pair_style is provided by lammps-mtp-kokkos "
                f"({LAMMPS_MTP_KOKKOS_URL}); build LAMMPS with it using CPU-only or with Kokkos."
            ) from error

    def minimum_number_of_training_environments(self) -> Dict[str, int]:
        """Per-element atomic-environment floor for the D-optimality active set.

        MTP's active set is a single pooled maximal-volume n x n matrix over environments of any species (n is
        the descriptor-space dimension, i.e. the number of trainable coefficients), so the floor is really the
        single total n. To fit the per-element interface, n is spread evenly across the potential's species.
        n combines the level-determined basis sizes (radial_funcs_count, radial_basis_size,
        alpha_scalar_moments) with the species count, the radial term scaling as species_count squared:

            n = species_count**2 * radial_funcs_count * radial_basis_size + alpha_scalar_moments + species_count

        Verified against the level templates: L6/1sp = 22, L8/1sp = 26, L16/1sp = 125, L16/3sp = 383.
        """
        descriptors = self.descriptors
        total_environments = (
            descriptors["species_count"] ** 2 * descriptors["radial_funcs_count"] * descriptors["radial_basis_size"]
            + descriptors["alpha_scalar_moments"]
            + descriptors["species_count"]
        )
        elements = self._trainer.configuration.elements
        per_element = -(-total_environments // len(elements))  # ceil(n / number of species)
        return {element: per_element for element in elements}

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: Path,
        mtp_configuration: MtpConfiguration,
        lammps_runner: Union[SubprocessLammpsRunner, InProcessLammpsRunner] = None,
        mlp_executable_path: Optional[Path] = None,
    ) -> "MtpMlip":
        """Reconstruct an MTP MLIP from a fitted potential file (the configuration must be provided)."""
        mtp_trainer = MtpTrainer.load_checkpoint(
            checkpoint_path, mtp_configuration=mtp_configuration, mlp_executable_path=mlp_executable_path
        )
        return cls(mtp_trainer=mtp_trainer, lammps_runner=lammps_runner)

    def train(self, output_directory: Path) -> None:
        """Train the model, deploy it and write a checkpoint into output_directory."""
        output_directory.mkdir(parents=True, exist_ok=True)

        self._trainer.fit()
        self._deploy(output_directory)  # writes the potential file and caches the MtpPotential
        self._trainer.write_training_log(output_directory)  # captured 'mlp train' output -> training.log

        self._model_file = self.lammps_potential.mtp_file_path
        self.write_state_yaml(output_directory / "state.yaml")

    def load(self, model_directory: Path) -> None:
        """Load the committed MTP model into a runnable potential and warm-start the next fit from it.

        The deployed ``potential.almtp`` drives the dynamics; restoring it into the trainer lets the next fit
        resume from these coefficients (mlp ``--init-params='same'``) rather than the cold level template.
        """
        self._lammps_potential = MtpPotential(mtp_file_path=Path(model_directory) / "potential.almtp")
        self._model_file = self.lammps_potential.mtp_file_path
        self._trainer.restore_from_checkpoint(model_directory)

    def write_state_yaml(self, output_path: Path) -> None:
        """Write a yaml with the current model_file, unc_file, lammps_potential_file and hyperparameters."""
        with open(str(output_path), "w") as file_descriptor:
            yaml.dump(self._state(), file_descriptor)

    def _mtp_parameters(self) -> Dict:
        """The parameters describing the current MTP model."""
        configuration = self._trainer.configuration
        return dict(
            level=configuration.level,
            max_dist=configuration.max_dist,
            radial_basis_size=configuration.radial_basis_size,
            alpha_scalar_moments=configuration.alpha_scalar_moments,
            species_count=configuration.species_count,
            number_of_adjustable_parameters=configuration.number_of_adjustable_parameters,
        )

    def _state(self) -> Dict:
        potential = self._lammps_potential
        model_file = None if self._model_file is None else str(self._model_file)
        # For MTP the model, the uncertainty and the LAMMPS pair-coeff are all the same .almtp file.
        potential_file = None if potential is None else str(potential.mtp_file_path)
        return dict(
            model_file=model_file,
            unc_file=potential_file,
            lammps_potential_file=potential_file,
            hyperparameters=self._mtp_parameters(),
            **self.training_set_state(),
        )
