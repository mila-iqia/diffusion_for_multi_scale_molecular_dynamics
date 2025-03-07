"""Moment Tensor Potential model.

This script defines a MTP model in a lightning like manner, with a train() and evaluate() method.
However, it cannot be called as a standard lightning module as it relies on the MLIP-3 library for the model
implementation.
"""

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pyace import (ACEBBasisSet, BBasisConfiguration,
                   aseatoms_to_atomicenvironment)

from diffusion_for_multi_scale_molecular_dynamics.mlip.mtp_utils import (
    crawl_lammps_directory, prepare_mtp_inputs_from_lammps)
from diffusion_for_multi_scale_molecular_dynamics.utils.maxvol import maxvol
from diffusion_for_multi_scale_molecular_dynamics.utils.pyace_utils import \
    compute_B_projections


@dataclass(kw_only=True)
class ACE_arguments:
    """Arguments to train an ACE MLIP with the pyace / pacemaker library."""

    config_path: str  # config yaml for the ACE
    initial_ace: Optional[str] = (
        None  # Define the initial ACE potential file. Default to None
    )
    fitted_ace_savedir: str = (
        "../"  # save directory for the fitted MTP. Defaults to '../' (current wd)
    )
    working_dir: Optional[str] = None
    clear_working_dir: bool = True
    maxvol_cutoff: float = 9.0
    maxvol_tol: float = 1.001
    maxvol_max_iters: int = 300


class ACE_MLIP:
    """ACE with py-ace."""

    def __init__(self, ace_args: ACE_arguments):
        """Modifications to py-ace for a more intuitive integration with our active learning loop.

        Args:
            ace_args: ACE arguments from the class ACEArguments
        """
        assert os.path.exists(
            ace_args.config_path
        ), f"config file not found. Got {ace_args.config_path}"
        self.config_path = ace_args.config_path
        os.makedirs(ace_args.fitted_ace_savedir, exist_ok=True)

        self.initial_ace = ace_args.initial_ace

        self.fitted_ace_savedir = ace_args.fitted_ace_savedir
        if ace_args.working_dir is not None:
            os.makedirs(ace_args.working_dir, exist_ok=True)
        self.working_dir = ace_args.working_dir

        self.clear_working_dir = ace_args.clear_working_dir
        if self.clear_working_dir:
            assert self.fitted_ace_savedir != self.working_dir, (
                "Got instructions to clear the working directory, but it matches the save directory. This would"
                "delete the saved file. Either change the working directory or do not clear it."
            )

        self.maxvol_cutoff = (
            ace_args.maxvol_cutoff
        )  # cutoff to generate atomic environment
        self.maxvol_tol = ace_args.maxvol_tol
        self.maxvol_max_iters = ace_args.maxvol_max_iters
        self.maxvol_epsilon = 1e-16  # small value used in maxvol algorithm

    def evaluate(
        self, dataset: pd.DataFrame, mlip_name: str, mode: str = "eval"
    ) -> Path:
        """Evaluate energies, forces, stresses and MaxVol gamma factor of structures with trained MTP.

        This calls _run_pacemaker with mode="eval" to evaluate MLIP on a dataset.

        Args:
            dataset: dataframe with the following columns:
                ase_atoms: The list of Pymatgen Structure object.
                energy_corrected: List of total energies of each structure in structures list.
                forces: List of (m, 3) forces array of each structure with m atoms in structures list.
                    m can be varied with each single structure case.
            mlip_name: filename for the trained MLIP.
            mode (optional): evaluate or train and evaluate. Defaults to "eval".

        Returns:
            path dataframe with ground truth energies, forces
        """
        assert mode in [
            "eval",
            "train_and_eval",
        ], f"Mode should be eval or train_and_eval. Got {mode}"
        return self._run_pacemaker(dataset, mlip_name, mode=mode)

    @staticmethod
    def prepare_dataset_from_lammps(
        root_data_dir: str,
        atom_dict: Dict[int, str],
        mode: str = "train",
        get_forces: bool = False,
    ) -> pd.DataFrame:
        """Get the LAMMPS in a folder and organize them as inputs for ACE.

        Args:
            root_data_dir: folder to read. Each LAMMPS sample is expected to be in a subfolder.
            atom_dict: map from LAMMPS index to atom name. e.g. {1: 'Si'}
            mode: subset of samples to get. Data from root_data_dir/*mode*/ folders will be parsed. Defaults to train.
            get_forces: whether to get forces from LAMMPS. Defaults to False. TODO implement this
        Returns:
            dataframe usable by pacemaker
        """
        lammps_outputs, thermo_outputs = crawl_lammps_directory(root_data_dir, mode)
        mtp_dataset = prepare_mtp_inputs_from_lammps(
            lammps_outputs, thermo_outputs, atom_dict
        )
        energies = mtp_dataset.energy  # list of length = num structures
        forces = (
            mtp_dataset.forces
        )  # list of ... of length = num structures. Each element is length = num_atoms. Each
        # is length spatial_dimension (num_structures, num_atoms, spatial_dimension)
        # pymatgen Structure can be converted to ASE Atoms classes
        ase_atoms = [x.to_ase_atoms() for x in mtp_dataset.structure]
        num_atoms = [len(x) for x in mtp_dataset.forces]
        ase_dataframe = pd.DataFrame(
            {
                "energy": energies,
                "energy_corrected": energies,  # stupid retro-engineering
                "forces": forces,
                "ase_atoms": ase_atoms,
                "NUMBER_OF_ATOMS": num_atoms,
            }  # some retro-engineering going on here
        )
        return ase_dataframe

    @staticmethod
    def merge_inputs(ace_inputs: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge a list of inputs in a single input.

        Args:
            ace_inputs: list of input dataframes

        Returns:
            merged dataframes
        """
        return pd.concat(ace_inputs)

    def train(self, dataset: pd.DataFrame, mlip_name: str = "ace_fitted.yaml") -> Path:
        """Training method for an ACE MLIP.

        This calls _run_pacemaker with mode="train" to train MLIP on a dataset.

        Args:
            dataset: dataframe dataclass with the following elements:
                ase_atoms: ASE Atoms object which represents an atomic configuration.
                energy_corrected: List of total energies of each structure used as targets to train the model
                forces: List of (m, 3) forces array  with m atoms for each ASE Atoms object.
                    m can vary with each structure.
            mlip_name: filename for the trained ACE. Defaults to ace_fitted.yaml

        Returns:
            fitted_mtp: path to the fitted ACE MLIP
        """
        return self._run_pacemaker(dataset, mlip_name, mode="train")

    def _run_pacemaker(
        self,
        dataset: pd.DataFrame,
        mlip_name: str = "ace_fitted.yaml",
        mode: str = "train",
    ) -> Union[Path, pd.DataFrame]:
        """Generic method calling pacemaker for training or evaluating ACE MLIP.

        Args:
            dataset: dataframe dataclass with the following elements:
                ase_atoms: ASE Atoms object which represents an atomic configuration.
                energy_corrected: List of total energies of each structure used as targets to train the model
                forces: List of (m, 3) forces array  with m atoms for each ASE Atoms object.
                    m can vary with each structure.
            mlip_name: filename for the trained ACE. Defaults to ace_fitted.yaml
        Returns:
            fitted_mtp: path to the fitted ACE MLIP
        """
        init_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp_work_dir:
            data_pkl = Path(tmp_work_dir) / "data_pkl.gzip"
            dataset.to_pickle(data_pkl, compression="gzip", protocol=4)
            output_mlip = Path(self.fitted_ace_savedir) / mlip_name
            os.chdir(tmp_work_dir)
            commands = [
                "pacemaker",
                self.config_path,
                "-o",
                str(output_mlip),
                "-d",
                data_pkl,
            ]
            if mode == "train":
                commands.append("--no-predict")
                if self.initial_ace is not None:
                    commands += ["-ip", self.initial_ace]
            elif mode != "train_and_eval":
                commands += ["--no-fit", "-p", str(output_mlip)]
            with subprocess.Popen(commands, stdout=subprocess.PIPE) as p:
                stdout = p.communicate()[0]
                rc = p.returncode
            if rc != 0:
                error_msg = f"pacemaker exited with return code {rc}"
                msg = stdout.decode("utf-8").split("\n")[:-1]
                try:
                    error_line = next(
                        i for i, m in enumerate(msg) if m.startswith("ERROR")
                    )
                    error_msg += ", ".join(msg[error_line:])
                except Exception:
                    error_msg += msg[-1]
                raise RuntimeError(error_msg)

            if "eval" in mode:  # data is saved in a pickle gzip file in tmp_work_dir
                eval_data = self.process_evaluation_dataframe(
                    os.path.join(tmp_work_dir, "train_pred.pckl.gzip"), str(output_mlip)
                )

            # clean up
            os.chdir(
                init_dir
            )  # tmp_work_dir is deleted. We need to change dir to the original one
            # this creates issues with debugger otherwise
            if self.working_dir is not None:
                shutil.copytree(tmp_work_dir, self.working_dir, dirs_exist_ok=True)

            if self.clear_working_dir:
                clear_commands = ["pacemaker", "-c"]
                with subprocess.Popen(clear_commands, stdout=subprocess.PIPE) as p:
                    _ = p.communicate()[0]
                    rc = p.returncode
                if rc != 0:
                    raise RuntimeError(
                        f"pacemaker clean operation failed with return code {rc}"
                    )
            data_pkl.unlink()

        returned_value = output_mlip if mode == "train" else eval_data
        return returned_value

    def process_evaluation_dataframe(
        self, path_to_eval_data: str, ase_potential: str
    ) -> pd.DataFrame:
        """Process evaluation data.

        Processes an evaluation DataFrame, transforming it into a flattened structure with one row per atom
        instead of one row per sample. This includes associated properties such as positions, forces, energies,
        and additional computed metrics.

        Args:
            path_to_eval_data (str): Path to the evaluation data file, which is expected to be
                a pickled DataFrame with compression enabled.
            ase_potential (str): The path or identifier for the ASE potential, used to retrieve
                associated required properties and to perform necessary computations.

        Returns:
            pd.DataFrame: A concatenated DataFrame containing per-atom properties, including predicted
            and target forces, positions, energy values, and additional computed values (e.g., neighborhood grades).
        """
        eval_data = pd.read_pickle(path_to_eval_data, compression="gzip")
        # flatten the dataframe in a more useful structure with 1 row per atom instead of 1 row per sample
        new_eval_data = []

        # positions are not in that dataframe... because why make things easy when you have a choice
        original_data_df = eval_data["name"][0][:-2]
        original_data_df = pd.read_pickle(original_data_df, compression="gzip")[
            "ase_atoms"
        ]
        maxvol_values = self.get_maxvol(ase_potential, original_data_df)
        cur_atom_idx = 0

        for structure_index, properties in eval_data.iterrows():
            num_atoms = properties["NUMBER_OF_ATOMS"]
            pred_forces = np.array(
                properties["forces_pred"]
            )  # shape: (num_atoms, spatial_dimension)
            real_forces = np.array(properties["forces"])
            # positions = properties["ase_atoms"].get_positions()  # shape: (num_atoms, spatial_dimension)
            positions = original_data_df[structure_index].get_positions()
            pred_energy = [properties["energy_pred"]] * num_atoms  # predicted energy
            real_energy = [
                properties["energy_corrected"]
            ] * num_atoms  # predicted energy
            s_index = [structure_index] * num_atoms
            atom_index = list(range(num_atoms))
            structure_df = pd.DataFrame(
                {
                    "x": positions[:, 0],
                    "y": positions[:, 1],
                    "z": positions[:, 2],
                    "fx": pred_forces[:, 0],
                    "fy": pred_forces[:, 1],
                    "fz": pred_forces[:, 2],
                    "energy": pred_energy,
                    "atom_index": atom_index,
                    "structure_index": s_index,
                    "fx_target": real_forces[:, 0],
                    "fy_target": real_forces[:, 1],
                    "fz_target": real_forces[:, 2],
                    "energy_target": real_energy,
                    "nbh_grades": maxvol_values[
                        cur_atom_idx:cur_atom_idx + num_atoms
                    ],
                }
            )
            cur_atom_idx += num_atoms
            new_eval_data.append(structure_df)

        return pd.concat(new_eval_data)

    def get_maxvol(
        self,
        potential_yaml_path: str,
        structure_data: pd.Series,
    ) -> np.array:
        """Compute maxvol gamma per atom.

        Args:
            potential_yaml_path (str): Path to the YAML file containing potential configuration
                attributes. Used for creating a basis configuration.
            structure_data (pd.Series): Input structural data in the form of a Pandas Series.
                Each entry represents data related to atomic environments.

        Returns:
            maxvol: an array with maxvol gamma factor for each atom.
        """
        bconf = BBasisConfiguration(potential_yaml_path)
        bbasis = ACEBBasisSet(bconf)
        elements_to_index_map = bbasis.elements_to_index_map
        atomic_env_list = structure_data.apply(
            aseatoms_to_atomicenvironment,
            cutoff=self.maxvol_cutoff,
            elements_mapper_dict=elements_to_index_map,
        )
        a_zero_projection_dict, structure_ind_dict = compute_B_projections(
            bbasis, atomic_env_list
        )

        # from pyace.activelearning compute_active_set
        maxvol_gamma_list = []
        for st, cur_a_zero in a_zero_projection_dict.items():
            shape = cur_a_zero.shape
            assert shape[0] < shape[1], (
                f"Insufficient atomic environments to determine active set for species type {st}, "
                f"system is under-determined, projections shape={shape}"
            )
            proj_std = np.std(cur_a_zero, axis=0)
            zero_proj_columns = np.where(proj_std == 0)[0]
            if len(zero_proj_columns) > 0:
                zero_projs = np.zeros((len(cur_a_zero), len(zero_proj_columns)))
                np.fill_diagonal(zero_projs, self.maxvol_epsilon)
                cur_a_zero[:, zero_proj_columns] += zero_projs
            _, maxvol_gamma = maxvol(
                cur_a_zero,
                tol=self.maxvol_tol,
                max_iters=self.maxvol_max_iters,
            )
            maxvol_gamma = np.diagonal(maxvol_gamma)
            maxvol_gamma_list.append(maxvol_gamma)
        # TODO check the mapping between indices and maxvol gamma
        # we might have to stack maxvol_gamma_list ? TBD
        return maxvol_gamma
