"""GRACE-FS machine-learning interatomic potential."""

import shutil
from pathlib import Path
from typing import Dict, Optional, Union

import yaml

from diffusion_for_multi_scale_molecular_dynamics.io.lammps.potential.grace import \
    GracePotential
from diffusion_for_multi_scale_molecular_dynamics.mlip.base_mlip import \
    BaseMLIP
from diffusion_for_multi_scale_molecular_dynamics.mlip.grace.grace_configuration import \
    GraceConfiguration
from diffusion_for_multi_scale_molecular_dynamics.mlip.grace.grace_trainer import \
    GraceTrainer
from diffusion_for_multi_scale_molecular_dynamics.oracle.base_single_point_calculator import \
    SinglePointCalculation
from diffusion_for_multi_scale_molecular_dynamics.oracle.lammps_runner import (
    InProcessLammpsRunner, SubprocessLammpsRunner)

GRACE_LAMMPS_CLONE = "git clone -b grace --depth=1 https://github.com/yury-lysogorskiy/lammps.git"
GRACEMAKER_CLONE = "git clone https://github.com/ICAMS/grace-tensorpotential.git"
PACE_ACTIVESET_CLONE = "git clone -b feature/grace_fs https://github.com/ICAMS/python-ace.git"


class GraceMlip(BaseMLIP):
    """GRACE-FS MLIP: trained with gracemaker and run through the lammps grace/fs pair_style."""

    name = "GRACE"
    training_program_name = "gracemaker"

    def __init__(
        self,
        grace_trainer: GraceTrainer,
        lammps_runner: Union[SubprocessLammpsRunner, InProcessLammpsRunner],
    ):
        """Init method.

        Args:
            grace_trainer: fits the GRACE-FS model and exports its LAMMPS potential.
            lammps_runner: runner used to evaluate the deployed potential; must provide the grace/fs pair_style.
        """
        super().__init__(trainer=grace_trainer, lammps_runner=lammps_runner)
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Fail early if the LAMMPS grace/fs pair_style or the gracemaker / pace_activeset tools are missing."""
        try:
            self._lammps_runner.check_dependency(section="Pair styles", to_find="grace/fs")
        except RuntimeError as error:
            raise RuntimeError(
                f"{error} The grace/fs pair_style requires a GRACE-enabled LAMMPS build: {GRACE_LAMMPS_CLONE}"
            ) from error

        if shutil.which("gracemaker") is None:
            raise RuntimeError(
                f"gracemaker was not found on PATH; it is required to train a GRACE-FS model. Install it with: "
                f"{GRACEMAKER_CLONE}"
            )
        if shutil.which("pace_activeset") is None:
            raise RuntimeError(
                "pace_activeset was not found on PATH; it builds the GRACE-FS active set (.asi) used for the "
                f"extrapolation grade. Install it with: {PACE_ACTIVESET_CLONE}"
            )

    def train(self, output_directory: Path) -> None:
        """Train the model, deploy it and write a checkpoint into output_directory."""
        output_directory.mkdir(parents=True, exist_ok=True)

        self._trainer.fit()
        self._deploy(output_directory)  # write_checkpoint: builds the active set and caches the GracePotential

        self._model_file = self.lammps_potential.model_file_path
        self.write_state_yaml(output_directory / "state.yaml")

    def write_state_yaml(self, output_path: Path) -> None:
        """Write a yaml with the current model_file, unc_file, lammps_potential_file and hyperparameters."""
        with open(str(output_path), "w") as file_descriptor:
            yaml.dump(self._state(), file_descriptor)

    def load(self, model_directory: Path) -> None:
        """Load the committed GRACE-FS model into a runnable potential and restore its '-rl' warm-start seed.

        GRACE-FS refits from the full training database on the next fit, but the seed folder is restored so
        that fit resumes with gracemaker '-rl': the warm start drastically shortens fitting and must always
        be used, so it has to survive a restart.
        """
        model_directory = Path(model_directory)
        self._trainer.restore_from_checkpoint(model_directory)
        self._lammps_potential = GracePotential(
            model_file_path=model_directory / "model.yaml",
            active_set_file_path=model_directory / "model.asi",
        )
        self._model_file = self.lammps_potential.model_file_path

    def minimum_number_of_training_environments(self) -> Dict[str, int]:
        """Per-element atomic-environment needed for the creation of the D-optimality active set.

        pace_activeset builds a, per element, square matrix with rows=atomic_environments,
        columns=descriptor_functions. This algorithm raises an error if there are fewer rows than columns. This
        function returns a dict of this minimal number of descriptors. The tricky part is, GRACE doesn't have a
        static basis table: the model is a Tf Graph and the only object giving us access to the per-element
        number_of_descriptor_functions is the exported GRACEFSBasisSet:

        Step 1. Build the model specifications dict
        Step 2. get_preset(dict).get_instructions() -> Create the full Directed Acyclic Graph (DAG)
        Step 3. TPModel(instructions) -> A container that will contain the model's weights.
        Step 4. model.build -> Allocate those weights randomly (untrained). (Needed for Step 5)
        Step 5. model.export_to_yaml -> Export to a yaml file
        Step 6. Reload as a GRACEFSBasisSet, which exposes per-element number_of_descriptor_functions

        From Step 2, we could manually calculate the set, but this would be less robust than this process.
        """
        import copy
        import os
        import tempfile

        import tensorflow as tf
        from pyace.grace_fs import GRACEFSBasisSet
        from tensorpotential.potentials import get_preset
        from tensorpotential.tpmodel import TPModel

        configuration = self._trainer.configuration
        element_map = {element: index for index, element in enumerate(sorted(configuration.elements))}
        # Step 1: the model specification. Deep-copy model_kwargs - building the model wraps its lists as
        # TensorFlow ListWrappers, which would corrupt the shared config and break the fit's input.yaml.
        model_specification = dict(
            element_map=element_map, rcut=configuration.cutoff, **copy.deepcopy(configuration.model_kwargs)
        )
        instructions = get_preset(configuration.preset)(**model_specification).get_instructions()

        model = TPModel(instructions)
        model.build(tf.float64, jit_compile=False)

        with tempfile.TemporaryDirectory() as temporary_directory:
            untrained_model_path = os.path.join(temporary_directory, "untrained_fs.yaml")
            model.export_to_yaml(untrained_model_path)
            basis_set = GRACEFSBasisSet(untrained_model_path)
            number_of_functions = list(basis_set.nfuncs)
            element_to_index = dict(basis_set.elements_to_index_map)

        return {element: int(number_of_functions[index]) for element, index in element_to_index.items()}

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: Path,
        grace_configuration: GraceConfiguration,
        initial_configuration: SinglePointCalculation,
        lammps_runner: Union[SubprocessLammpsRunner, InProcessLammpsRunner] = None,
        gracemaker_executable_path: Optional[Path] = None,
        pace_activeset_executable_path: Optional[Path] = None,
    ) -> "GraceMlip":
        """Reconstruct a GRACE-FS MLIP from a checkpoint (the configuration and initial config must be provided)."""
        grace_trainer = GraceTrainer.load_checkpoint(
            checkpoint_path,
            grace_configuration=grace_configuration,
            initial_configuration=initial_configuration,
            gracemaker_executable_path=gracemaker_executable_path,
            pace_activeset_executable_path=pace_activeset_executable_path,
        )
        return cls(grace_trainer=grace_trainer, lammps_runner=lammps_runner)

    def _grace_parameters(self) -> Dict:
        """The parameters describing the current GRACE-FS model."""
        configuration = self._trainer.configuration
        return dict(
            elements=configuration.elements,
            cutoff=configuration.cutoff,
            preset=configuration.preset,
            size=configuration.size,
            seed=configuration.seed,
            target_total_updates=configuration.target_total_updates,
        )

    def _state(self) -> Dict:
        potential = self._lammps_potential
        model_file = None if self._model_file is None else str(self._model_file)
        # The FS model (.yaml) is the LAMMPS pair-coeff; the active set (.asi) provides the extrapolation grade.
        lammps_potential_file = None if potential is None else str(potential.model_file_path)
        unc_file = None if potential is None else str(potential.active_set_file_path)
        return dict(
            model_file=model_file,
            unc_file=unc_file,
            lammps_potential_file=lammps_potential_file,
            hyperparameters=self._grace_parameters(),
            **self.training_set_state(),
        )
