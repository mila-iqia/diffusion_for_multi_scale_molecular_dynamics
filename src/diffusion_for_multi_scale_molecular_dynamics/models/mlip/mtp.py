"""Moment Tensor Potential model.

This script defines a MTP model in a lightning like manner, with a train() and evaluate() method.
However, it cannot be called as a standard lightning module as it relies on the MLIP-3 library for the model
implementation.
"""

import itertools
import os
import re
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TextIO, Tuple

import numpy as np
import pandas as pd
from maml.apps.pes import MTPotential
from maml.utils import check_structures_forces_stresses, pool_from
from monty.io import zopen
from monty.tempfile import ScratchDir
from pymatgen.core import Structure

from diffusion_for_multi_scale_molecular_dynamics.mlip.mtp_utils import (
    MTPInputs, concat_mtp_inputs, crawl_lammps_directory,
    prepare_mtp_inputs_from_lammps)


@dataclass(kw_only=True)
class MTPArguments:
    """Arguments to train an MTP with the MLIP3 library."""

    mlip_path: str  # path to MLIP3 library
    name: Optional[str] = None  # MTP
    param: Optional[Dict[Any, Any]] = None
    unfitted_mtp: str = "08.almtp"  # Define the initial mtp file. Default to 08g.amltp
    fitted_mtp_savedir: str = (
        "../"  # save directory for the fitted MTP. Defaults to '../' (current wd)
    )
    max_dist: float = 5  # The actual radial cutoff. Defaults to 5.
    radial_basis_size: int = (
        8  # Relevant to number of radial basis function. Defaults to 8.
    )
    max_iter: int = 1000  # The number of maximum iteration. Defaults to 1000.
    energy_weight: float = 1  # The weight of energy. Defaults to 1
    force_weight: float = 1e-2  # The weight of forces. Defaults to 1e-2
    stress_weight: float = (
        1e-3  # The weight of stresses. Zero-weight can be assigned. Defaults to 1e-3.
    )
    init_params: str = (
        "same"  # how to initialize parameters if a potential was not pre-fitted: "same" or "random".
    )
    scale_by_force: float = (
        0  # If > 0 then configurations near equilibrium get more weight. Defaults to 0.
    )
    bfgs_conv_tol: float = (
        1e-3  # Stop training if error dropped by a factor smaller than this over 50 BFGS iterations.
    )
    weighting: str = (
        "vibration"  # How to weight configuration with different sizes relative to each other.
    )
    # Choose from "vibrations", "molecules" and "structures". Defaults to "vibration".


class MTPWithMLIP3(MTPotential):
    """MTP with MLIP-3."""

    def __init__(self, mtp_args: MTPArguments):
        """Modifications to maml.apps.pes._mtp.MTPotential to be compatible with mlip-3.

        Args:
            mtp_args: MTP arguments from the class MTPArguments
        """
        super().__init__(mtp_args.name, mtp_args.param)
        self.mlp_command = os.path.join(mtp_args.mlip_path, "build", "mlp")
        assert os.path.exists(
            self.mlp_command
        ), "mlp command not found in mlip-3 build folder"
        self.mlp_templates = os.path.join(mtp_args.mlip_path, "MTP_templates")
        assert os.path.exists(
            self.mlp_templates
        ), "MTP templates not found in mlip-3 folder"
        self.fitted_mtp = None
        self.elements = None
        self.mtp_args = mtp_args
        self.savedir = mtp_args.fitted_mtp_savedir
        os.makedirs(self.savedir, exist_ok=True)

    def to_lammps_format(self):
        """Write the trained MTP in a LAMMPS compatible format."""
        # TODO
        # write params write the fitted mtp in a LAMMPS compatible format
        # self.write_param(
        #    fitted_mtp=fitted_mtp,
        #    Abinitio=0,
        #    Driver=1,
        #    Write_cfgs=predict_file,
        #    Database_filename=original_file,
        #    **kwargs,
        # )
        pass

    def evaluate(
        self, dataset: MTPInputs, mlip_name: str = "mtp_fitted.almtp"
    ) -> pd.DataFrame:
        """Evaluate energies, forces, stresses and MaxVol gamma factor of structures with trained MTP.

        Args:
            dataset: MTPInputs dataclass with the following elements:
                structures: The list of Pymatgen Structure object.
                energies: List of total energies of each structure in structures list.
                forces: List of (m, 3) forces array of each structure with m atoms in structures list.
                    m can be varied with each single structure case.
            mlip_name: str : filename for the trained MTP. Defaults to mtp_fitted.almtp

        Returns:
            dataframe with ground truth energies, forces
            dataframe with predicted energies, forces, MaxVol gamma (nbh grades)
        """
        if not mlip_name.endswith(".almtp"):
            mlip_name += ".almtp"
        assert os.path.exists(mlip_name), f"Trained MTP does not exists: {mlip_name}"

        original_file = "original.cfgs"
        predict_file = "predict.cfgs"

        # TODO if forces are not available...
        test_structures, test_forces, _ = check_structures_forces_stresses(
            dataset.structure, dataset.forces, None
        )
        predict_pool = pool_from(test_structures, dataset.energy, test_forces)
        local_mtp_name = "mtp.almtp"

        with ScratchDir(
            "."
        ):  # mlip needs a tmp_work_dir - we will manually copy relevant outputs elsewhere
            # write the structures to evaluate in a mlp compatible format
            original_file = self.write_cfg(original_file, cfg_pool=predict_pool)
            # TODO how to handle when GT is not available
            # df_orig = self.read_cfgs(
            #     original_file, nbh_grade=False
            # )  # read original values as a DataFrame

            # copy the trained mtp in the scratchdir
            shutil.copyfile(mlip_name, os.path.join(os.getcwd(), local_mtp_name))
            # calculate_grade is the method to get the forces, energy & maxvol values
            cmd = [
                self.mlp_command,
                "calculate_grade",
                local_mtp_name,
                original_file,
                predict_file,
            ]
            predict_file += ".0"  # added by mlp...
            stdout, rc = self._call_mlip(cmd)

            # check that MTP was called properly
            if rc != 0:
                error_msg = f"mlp exited with return code {rc}"
                msg = stdout.decode("utf-8").split("\n")[:-1]
                try:
                    error_line = next(
                        i for i, m in enumerate(msg) if m.startswith("ERROR")
                    )
                    error_msg += ", ".join(msg[error_line:])
                except Exception:
                    error_msg += msg[-1]
                raise RuntimeError(error_msg)
            # read the config
            df_predict = self.read_cfgs(predict_file, nbh_grade=True)
        return df_predict

    def read_cfgs(self, filename: str, nbh_grade: bool = False) -> pd.DataFrame:
        """Read mlp output when MaxVol gamma factor is present.

        Args:
            filename: name of mlp output file to be parsed.
            nbh_grade (optional): if True, add the nbh_grades (neighborhood-based approach to determine the MaxVol gamma
                values - see MLIP3 paper) in the resulting dataframe. Defaults to False.

        Returns:
            dataframe with energies, forces, optional nbh grades (MaxVol gamma)
        """

        def formatify(string: str) -> List[float]:
            """Convert string to a list of float."""
            return [float(s) for s in string.split()]

        if not self.elements:
            raise ValueError("No species given.")

        data_pool = []
        with zopen(filename, "rt") as f:
            lines = f.read()

        block_pattern = re.compile("BEGIN_CFG\n(.*?)\nEND_CFG", re.S)
        size_pattern = re.compile("Size\n(.*?)\n SuperCell", re.S | re.I)
        if nbh_grade:
            position_pattern = re.compile("nbh_grades\n(.*?)\n Energy", re.S)
        else:
            position_pattern = re.compile("fz\n(.*?)\n Energy", re.S)
        energy_pattern = re.compile("Energy\n(.*?)\n (?=PlusStress|Stress)", re.S)
        # stress_pattern = re.compile("xy\n(.*?)(?=\n|$)", re.S)  # TODO stress values

        for block in block_pattern.findall(lines):
            d = {"outputs": {}}
            size_str = size_pattern.findall(block)[0]
            size = int(size_str.lstrip())
            position_str = position_pattern.findall(block)[0]
            position = np.array(list(map(formatify, position_str.split("\n"))))
            species = np.array(self.elements)[position[:, 1].astype(np.int64)]
            forces = position[:, 5:8].tolist()

            energy_str = energy_pattern.findall(block)[0]
            energy = float(energy_str.lstrip())
            # TODO add stress
            # stress_str = stress_pattern.findall(block)[0]
            # virial_stress = (np.array(list(map(formatify, stress_str.split()))).reshape(6,).tolist())
            # virial_stress = [virial_stress[self.mtp_stress_order.index(n)] for n in self.vasp_stress_order]
            d["outputs"]["energy"] = energy
            d["num_atoms"] = size
            d["outputs"]["position"] = position[:, 2:5].tolist()
            d["outputs"]["forces"] = forces
            d["outputs"]["species"] = species
            # d["outputs"]["virial_stress"] = virial_stress
            if nbh_grade:
                nbh_grade_values = position[:, 8].tolist()
                d["outputs"]["nbh_grades"] = nbh_grade_values
            data_pool.append(d)

        # originally used convert_docs from maml.utils, but it hard-coded the structure of the dataframe and does not
        # manage nbh_grades; we use our own implementation instead
        df = self.convert_to_dataframe(docs=data_pool)
        return df

    @staticmethod
    def convert_to_dataframe(docs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert a list of docs into DataFrame usable for computing metrics and analysis.

        Modified from maml.utils._data_conversion.py

        Args:
            docs: list of docs generated by mlip-3. Each doc should be structured as a dict.

        Returns:
            A DataFrame with energy, force, and nbh grades (MaxVol factor) per atom and per structure. Energy is
            repeated for each atom and corresponds to the total energy predicted.
        """
        df = defaultdict(list)
        for s_idx, d in enumerate(docs):
            n_atom = d["num_atoms"]
            outputs = d["outputs"]
            pos_arr = np.array(outputs["position"])
            assert (
                n_atom == pos_arr.shape[0]
            ), "Number of positions do not match number of atoms"
            force_arr = np.array(outputs["forces"])
            assert (
                n_atom == force_arr.shape[0]
            ), "Number of forces do not match number of atoms"
            for i, x in enumerate(["x", "y", "z"]):
                df[x] += pos_arr[:, i].tolist()
                df[f"f{x}"] += force_arr[:, i].tolist()
            df["energy"] += [outputs["energy"]] * n_atom  # copy the value to all atoms
            if "nbh_grades" in outputs.keys():
                nbh_grades = outputs["nbh_grades"]
                assert n_atom == len(
                    nbh_grades
                ), "Number of gamma values do not match number of atoms"
                df["nbh_grades"] += nbh_grades
            df["atom_index"] += list(range(n_atom))
            df["structure_index"] += [s_idx] * n_atom
            df["species"] += outputs["species"].tolist()

        df = pd.DataFrame(df)
        return df

    @staticmethod
    def _call_mlip(cmd_list: List[str]) -> Tuple[bytes, int]:
        """Call MLIP library with subprocess.

        Args:
            cmd_list: list of commands & arguments to execute

        Returns:
            stdout: output of the executed commands
            rc: return code of the executed commands
        """
        with subprocess.Popen(cmd_list, stdout=subprocess.PIPE) as p:
            stdout = p.communicate()[0]
            rc = p.returncode
        return stdout, rc

    @staticmethod
    def _call_cmd_to_stdout(cmd: List[str], output_file: TextIO):
        """Call commands with subprocess.POpen and pipe output to a file.

        Args:
            cmd: list of commands & arguments to run
            output_file: name of the file where the stdout is redirected
        """
        with subprocess.Popen(cmd, stdout=output_file) as p:
            p.communicate()[0]

    @staticmethod
    def prepare_dataset_from_lammps(
        root_data_dir: str,
        atom_dict: Dict[int, str],
        mode: str = "train",
        get_forces: bool = True,
    ) -> MTPInputs:
        """Get the LAMMPS in a folder and organize them as inputs for a MTP.

        Args:
            root_data_dir: folder to read. Each LAMMPS sample is expected to be in a subfolder.
            atom_dict: map from LAMMPS index to atom name. e.g. {1: 'Si'}
            mode: subset of samples to get. Data from root_data_dir/*mode*/ folders will be parsed. Defaults to train.
            get_forces: if True, get the forces from the samples. Defaults to True.

        Returns:
            inputs for MTP in the MTPInputs dataclass
        """
        lammps_outputs, thermo_outputs = crawl_lammps_directory(root_data_dir, mode)
        mtp_dataset = prepare_mtp_inputs_from_lammps(
            lammps_outputs, thermo_outputs, atom_dict, get_forces=get_forces
        )
        return mtp_dataset

    @staticmethod
    def prepare_dataset_from_numpy(
        cartesian_positions: np.ndarray,
        box: np.ndarray,
        forces: np.ndarray,
        energy: float,
        atom_type: np.ndarray,
        atom_dict: Dict[int, str] = {1: "Si"},
    ) -> MTPInputs:
        """Convert numpy array variables to a format compatible with MTP.

        Args:
            cartesian_positions: atomic positions in Angstrom as a (n_atom, 3) array.
            box: unit cell description as a (3, 3) array.
            forces: forces on each atom as a (n_atom, 3) array
            energy: energy of the configuration
            atom_type: indices for each atom in the structure as a (n_atom,) array
            atom_dict: map between atom indices and atom types

        Returns:
            data formatted at an input for MTP.
        """
        structure = Structure(
            lattice=box,
            species=[atom_dict[x] for x in atom_type],
            coords=cartesian_positions,
            coords_are_cartesian=True,
        )
        forces = (
            forces.tolist()
        )  # from Nx3 np array to a list of length N where each element is a list of 3 forces
        return MTPInputs(structure=[structure], forces=[forces], energy=[energy])

    @staticmethod
    def merge_inputs(mtp_inputs: List[MTPInputs]) -> MTPInputs:
        """Merge a list of MTPInputs in a single MTPInputs.

        Args:
            mtp_inputs: list of MTPInputs

        Returns:
            merged MTPInputs
        """
        merged_inputs = MTPInputs(structure=[], forces=[], energy=[])
        for x in mtp_inputs:
            merged_inputs = concat_mtp_inputs(merged_inputs, x)
        return merged_inputs

    def train(self, dataset: MTPInputs, mlip_name: str = "mtp_fitted.almtp") -> str:
        """Training data with moment tensor method using MLIP-3.

        Override the base class method.

        Args:
            dataset: MTPInputs dataclass with the following elements:
                structures: The list of Pymatgen Structure object.
                energies: List of total energies of each structure in structures list.
                forces: List of (m, 3) forces array of each structure with m atoms in structures list.
                    m can be varied with each single structure case.
            mlip_name: str : filename for the trained MTP. Defaults to mtp_fitted.almtp

        Returns:
            fitted_mtp: path to the fitted MTP
        """
        train_structures, train_forces, train_stresses = (
            check_structures_forces_stresses(dataset.structure, dataset.forces, None)
        )
        # last argument is for stresses - not used currently
        train_pool = pool_from(train_structures, dataset.energy, train_forces)

        elements = sorted(
            set(itertools.chain(*[struct.species for struct in train_structures]))
        )
        self.elements = [str(element) for element in elements]  # TODO move to __init__

        atoms_filename = "train.cfgs"

        with ScratchDir("."):  # create a tmpdir - deleted afterwards
            atoms_filename = self.write_cfg(
                filename=atoms_filename, cfg_pool=train_pool
            )

            if not self.mtp_args.unfitted_mtp:
                raise RuntimeError("No specific parameter file provided.")
            mtp_file_path = os.path.join(self.mlp_templates, self.mtp_args.unfitted_mtp)
            shutil.copyfile(
                mtp_file_path, os.path.join(os.getcwd(), self.mtp_args.unfitted_mtp)
            )
            commands = [self.mlp_command, "mindist", atoms_filename]
            with open("min_dist", "w") as f:
                self._call_cmd_to_stdout(commands, f)

            # TODO check what min_dist is used for in maml
            # with open("min_dist") as f:
            #    lines = f.readlines()
            # split_symbol = "="  # different for mlip-2 (":") and mlip-3 ("=")
            # min_dist = float(lines[-1].split(split_symbol)[1])

            save_fitted_mtp = mlip_name
            if not save_fitted_mtp.endswith(".almtp"):
                save_fitted_mtp += ".almtp"

            cmds_list = [
                self.mlp_command,
                "train",
                self.mtp_args.unfitted_mtp,
                atoms_filename,
                f"--save_to={save_fitted_mtp}",
                f"--iteration_limit={self.mtp_args.max_iter}",
                "--al_mode=nbh",  # active learning mode - required to get extrapolation grade
                f"--curr-pot-name={self.mtp_args.unfitted_mtp}",
                f"--energy-weight={self.mtp_args.energy_weight}",
                f"--force-weight={self.mtp_args.force_weight}",
                f"--stress-weight={self.mtp_args.stress_weight}",
                f"--init-params={self.mtp_args.init_params}",
                f"--scale-by-force={self.mtp_args.scale_by_force}",
                f"--bfgs-conv-tol={self.mtp_args.bfgs_conv_tol}",
                f"--weighting={self.mtp_args.weighting}",
            ]
            stdout, rc = self._call_mlip(cmds_list)
            if rc != 0:
                error_msg = f"MLP exited with return code {rc}"
                msg = stdout.decode("utf-8").split("\n")[:-1]
                try:
                    error_line = next(
                        i for i, m in enumerate(msg) if m.startswith("ERROR")
                    )
                    error_msg += ", ".join(msg[error_line:])
                except Exception:
                    error_msg += msg[-1]
                raise RuntimeError(error_msg)
            # copy the fitted mtp outside the working directory
            self.fitted_mtp = os.path.join(self.savedir, save_fitted_mtp)
            shutil.copyfile(save_fitted_mtp, self.fitted_mtp)
        return self.fitted_mtp
