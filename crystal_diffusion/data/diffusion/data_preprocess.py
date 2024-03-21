"""Convert results of LAMMPS simulation into dataloader friendly format."""
import logging
import os
import warnings
from typing import List, Optional

import pandas as pd

from crystal_diffusion.data.parse_lammps_outputs import parse_lammps_output

logger = logging.getLogger(__name__)


class LammpsProcessorForDiffusion:
    """Prepare data from LAMMPS for a diffusion model."""

    def __init__(self, raw_data_dir: str, processed_data_dir: str):
        """Read LAMMPS experiment directory and write a processed version to disk.

        Args:
            raw_data_dir: path to LAMMPS runs outputs
            processed_data_dir: path where processed files are saved
        """
        if not os.path.exists(processed_data_dir):
            os.makedirs(processed_data_dir)  # create the dir if it doesn't exist
        self.data_dir = processed_data_dir
        # TODO revisit data splits
        self.train_files = self.prepare_data(raw_data_dir, mode='train')
        self.valid_files = self.prepare_data(raw_data_dir, mode='valid')

    def prepare_data(self, raw_data_dir: str, mode: str = 'train') -> List[str]:
        """Read data in raw_data_dir and write to a parquet file for Datasets.

        Args:
            raw_data_dir: folder where LAMMPS runs are located.
            mode: train, valid or test split

        Returns:
            list of processed dataframe in parquet files
        """
        # TODO split is assumed from data generation. We might revisit this.
        # we assume that raw_data_dir contains subdirectories named train_run_N for N>=1
        # get the list of runs to parse
        assert mode in ['train', 'valid', 'test'], f"Mode should be train, valid or test. Got {mode}."
        list_runs = [d for d in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, d))
                     and d.startswith(f"{mode}_run")]
        list_files = []
        for d in list_runs:
            if f"{d}.parquet" not in os.listdir(self.data_dir):
                df = self.parse_lammps_run(os.path.join(raw_data_dir, d))
                if df is not None:
                    print('hello')
                    df.to_parquet(os.path.join(self.data_dir, f"{d}.parquet"), engine='pyarrow', index=False)
            if f"{d}.parquet" in os.listdir(self.data_dir):
                list_files.append(os.path.join(self.data_dir, f"{d}.parquet"))
        return list_files

    @staticmethod
    def _convert_coords_to_reduced(row: pd.Series) -> List[float]:
        """Convert a dataframe row to reduced coordinates.

        Args:
            row: entry in the dataframe. Should contain box, x, y and z

        Returns:
            x, y and z in reduced coordinates
        """
        x_lim, y_lim, z_lim = row['box']
        coord_red = [coord for triple in zip(row['x'], row['y'], row['z']) for coord in
                     (triple[0] / x_lim, triple[1] / y_lim, triple[2] / z_lim)]
        return coord_red

    def get_x_reduced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a column with reduced x,y, z coordinates.

        Args:
            df: dataframe with atomic positions. Should contain box, x, y and z.

        Returns:
            dataframe with added column of reduced positions [x1, y1, z1, x2, y2, ...]
        """
        df['reduced_position'] = df.apply(lambda x: self._convert_coords_to_reduced(x), axis=1)
        return df

    def parse_lammps_run(self, run_dir: str) -> Optional[pd.DataFrame]:
        """Parse outputs of a LAMMPS run and convert in a dataframe.

        Args:
            run_dir: directory where the outputs are located

        Returns:
            df: dataframe with bounding box, atoms species and coordinates. None if LAMMPS outputs are ambiguous.
        """
        # do something
        # find the LAMMPS dump file and thermo file
        dump_file = [d for d in os.listdir(run_dir) if 'dump' in d]
        if len(dump_file) != 1:
            warnings.warn(f"Found {len(dump_file)} files with dump in the name in {run_dir}. Skipping this run.",
                          UserWarning)
            return None

        thermo_file = [d for d in os.listdir(run_dir) if 'thermo' in d]
        if len(thermo_file) != 1:
            warnings.warn(f"Found {len(thermo_file)} files with thermo in the name in {run_dir}. Skipping this run.",
                          UserWarning)
            return None

        # parse lammps output and store in a dataframe
        df = parse_lammps_output(os.path.join(run_dir, dump_file[0]), os.path.join(run_dir, thermo_file[0]), None)

        # the dataframe contains the following columns: id (list of atom indices), type (list of int representing
        # atom type, x (list of x cartesian coordinates for each atom), y, z, fx (list forces in direction x for each
        # atom), energy (1 float).
        # Each row is a different MD step / usable example for diffusion model
        # TODO consider filtering out samples with large forces and MD steps that are too similar
        # TODO large force and similar are to be defined
        df = df[['type', 'x', 'y', 'z', 'box']]
        df = self.get_x_reduced(df)  # add reduced coordinates
        df['natom'] = df['type'].apply(lambda x: len(x))  # count number of atoms in a structure
        # naive implementation: a list of list which is converted into a 2d array by torch later
        # but a list of list is not ok with the writing on files with parquet
        df['position'] = df.apply(lambda x: [j for i in ['x', 'y', 'z'] for j in x[i]], axis=1)  # position as 3d array
        # position is natom * 3 array
        # TODO unit test to check the order after reshape
        return df[['natom', 'box', 'type', 'position', 'reduced_position']]
