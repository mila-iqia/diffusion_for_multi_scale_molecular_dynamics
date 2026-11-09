"""FLARE machine-learning interatomic potential."""

from pathlib import Path
from typing import Dict, Optional, Union

import yaml

from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.flare import \
    FlarePotential
from diffusion_for_multi_scale_molecular_dynamics.mlip.base_mlip import \
    BaseMLIP
from diffusion_for_multi_scale_molecular_dynamics.mlip.flare.flare_hyperparameter_optimizer import (
    FlareHyperparametersOptimizer, FlareOptimizerConfiguration)
from diffusion_for_multi_scale_molecular_dynamics.mlip.flare.flare_trainer import \
    FlareTrainer
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_runner import (
    InProcessLammpsRunner, SubprocessLammpsRunner)


class FlareMLIP(BaseMLIP):
    """FLARE machine-learning interatomic potential."""

    name = "FLARE"
    training_program_name = "FLARE"

    def __init__(
        self,
        flare_trainer: FlareTrainer,
        hyperparameter_optimizer: FlareHyperparametersOptimizer,
        lammps_runner: Union[SubprocessLammpsRunner, InProcessLammpsRunner],
    ):
        """Init method.

        Args:
            flare_trainer: wraps the FLARE sparse Gaussian process.
            hyperparameter_optimizer: drives the hyperparameter fitting during train().
            lammps_runner: runner used to evaluate the deployed potential.
        """
        super().__init__(trainer=flare_trainer, lammps_runner=lammps_runner)
        self._hyperparameter_optimizer = hyperparameter_optimizer

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: Path,
        hyperparameter_optimizer: Optional[FlareHyperparametersOptimizer] = None,
        lammps_runner: Union[SubprocessLammpsRunner, InProcessLammpsRunner] = None,
    ) -> "FlareMLIP":
        """Reconstruct a FLARE MLIP from a checkpoint (using an inactive optimizer if none is given)."""
        flare_trainer = FlareTrainer.load_checkpoint(checkpoint_path)
        if hyperparameter_optimizer is None:
            hyperparameter_optimizer = FlareHyperparametersOptimizer(
                FlareOptimizerConfiguration(
                    optimize_sigma=False, optimize_sigma_e=False, optimize_sigma_f=False, optimize_sigma_s=False
                )
            )
        return cls(
            flare_trainer=flare_trainer,
            hyperparameter_optimizer=hyperparameter_optimizer,
            lammps_runner=lammps_runner,
        )

    def train(self, output_directory: Path) -> None:
        """Train the model, deploy it and write a checkpoint into output_directory."""
        output_directory.mkdir(parents=True, exist_ok=True)

        if not self._hyperparameter_optimizer.is_inactive:
            _, history_df = self._trainer.fit_hyperparameters(self._hyperparameter_optimizer)
            history_df.to_pickle(output_directory / "hyperparameter_optimization_log.pkl")

        self._trainer.fit()
        self._deploy(output_directory)  # writes checkpoint.json + mapped files, caches the potential

        self._model_file = output_directory / "checkpoint.json"
        self.write_state_yaml(output_directory / "state.yaml")

    def write_state_yaml(self, output_path: Path) -> None:
        """Write a yaml with the current model_file, unc_file, lammps_potential_file and hyperparameters."""
        with open(str(output_path), "w") as file_descriptor:
            yaml.dump(self._state(), file_descriptor)

    def load(self, model_directory: Path) -> None:
        """Reload a committed FLARE model: the full sparse GP state and the deployed potential.

        FLARE folds data into its sparse GP incrementally (it does not refit from the database), so restoring
        only the deployed potential would lose the training history on a fresh-process restart. The sparse GP
        is therefore reloaded in full from the epoch's ``checkpoint.json`` so that subsequent folding continues
        from all prior data.
        """
        model_directory = Path(model_directory)
        self._trainer = FlareTrainer.load_checkpoint(
            model_directory / "checkpoint.json", training_database=self.training_database
        )
        self._lammps_potential = FlarePotential(
            pair_coeff_file_path=model_directory / "lmp.flare",
            mapped_uncertainty_file_path=model_directory / "map_unc_lmp.flare",
        )
        self._model_file = model_directory / "checkpoint.json"

    def _flare_hyperparameters(self) -> Dict:
        """The current SGP hyperparameters (sigma, sigma_e, sigma_f, sigma_s)."""
        sigma, sigma_e, sigma_f, sigma_s = self._trainer.sgp_model.sparse_gp.hyperparameters
        return dict(
            sigma=float(sigma), sigma_e=float(sigma_e), sigma_f=float(sigma_f), sigma_s=float(sigma_s)
        )

    def _state(self) -> Dict:
        potential = self._lammps_potential
        model_file = None if self._model_file is None else str(self._model_file)
        unc_file = None if potential is None else str(potential.mapped_uncertainty_file_path)
        lammps_potential_file = None if potential is None else str(potential.pair_coeff_file_path)
        hyperparameters = self._flare_hyperparameters()

        return dict(
            model_file=model_file,
            unc_file=unc_file,
            lammps_potential_file=lammps_potential_file,
            hyperparameters=hyperparameters,
            **self.training_set_state(),
        )
