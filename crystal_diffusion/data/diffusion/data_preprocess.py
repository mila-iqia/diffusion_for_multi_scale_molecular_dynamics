"""Convert results of LAMMPS simulation into dataloader friendly format."""
import itertools
import logging
import os
import warnings
from typing import List, Optional, Tuple, Union

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
        for count, d in enumerate(list_runs, 1):
            logging.info(f"Processing run directory {d} ({count} of {len(list_runs)})...")
            if f"{d}.parquet" not in os.listdir(self.data_dir):
                logging.info("     * parquet file is absent. Generating...")
                df = self.parse_lammps_run(os.path.join(raw_data_dir, d))
                if df is not None:
                    logging.info("     * writing parquet file to disk...")
                    df.to_parquet(os.path.join(self.data_dir, f"{d}.parquet"), engine='pyarrow', index=False)
            if f"{d}.parquet" in os.listdir(self.data_dir):
                list_files.append(os.path.join(self.data_dir, f"{d}.parquet"))
        return list_files

    @staticmethod
    def _convert_coords_to_relative(row: pd.Series) -> List[float]:
        """Convert a dataframe row to relative coordinates.

        Args:
            row: entry in the dataframe. Should contain box, x, y and z

        Returns:
            x, y and z in relative (reduced) coordinates
        """
        x_lim, y_lim, z_lim = row['box']
        # Cast the coordinates to float in case they are read in as strings
        coord_red = [coord for triple in zip(row['x'], row['y'], row['z']) for coord in
                     ((float(triple[0]) / x_lim) % 1, (float(triple[1]) / y_lim) % 1, (float(triple[2]) / z_lim) % 1)]
        return coord_red

    def get_x_relative(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a column with relative x,y, z coordinates.

        Args:
            df: dataframe with atomic positions. Should contain box, x, y and z.

        Returns:
            dataframe with added column of relative positions [x1, y1, z1, x2, y2, ...]
        """
        df['relative_positions'] = df.apply(lambda x: self._convert_coords_to_relative(x), axis=1)
        return df

    def get_dump_and_thermo_files(self, run_dir: str) -> Tuple[Union[str, None], Union[str, None]]:
        """Get dump and thermo files.

        Args:
            run_dir : path to run directory.

        Returns:
            dump_file_path, thermo_file_path: full path to data files; return None if there is not exactly
                one data file for each of (dump, thermo).
        """
        # find the LAMMPS dump file and thermo file
        dump_file = [d for d in os.listdir(run_dir) if 'dump' in d]
        if len(dump_file) == 1:
            dump_file_path = os.path.join(run_dir, dump_file[0])
        else:
            warnings.warn(f"Found {len(dump_file)} files with dump in the name in {run_dir}. "
                          f"Expected exactly one.", UserWarning)
            dump_file_path = None

        thermo_file = [d for d in os.listdir(run_dir) if 'thermo' in d]
        if len(thermo_file) == 1:
            thermo_file_path = os.path.join(run_dir, thermo_file[0])
        else:
            warnings.warn(f"Found {len(thermo_file)} files with thermo in the name in {run_dir}. "
                          f"Expected exactly one.", UserWarning)
            thermo_file_path = None

        return dump_file_path, thermo_file_path

    def get_lammps_output(self, run_dir: str):
        """Get lammps output.

        Args:
            run_dir : path to run directory.

        Returns:
            df: dataframe that contains the parsed lammps output. If the input files are missing, return None.
        """
        dump_file_path, thermo_file_path = self.get_dump_and_thermo_files(run_dir)
        if dump_file_path is None or thermo_file_path is None:
            return None

        df = parse_lammps_output(dump_file_path, thermo_file_path, None)
        return df

    def parse_lammps_run(self, run_dir: str) -> Optional[pd.DataFrame]:
        """Parse outputs of a LAMMPS run and convert in a dataframe.

        Args:
            run_dir: directory where the outputs are located

        Returns:
            df: dataframe with bounding box, atoms species and coordinates. None if LAMMPS outputs are ambiguous.
        """
        # parse lammps output and store in a dataframe
        df = self.get_lammps_output(run_dir)
        if df is None:
            warnings.warn("Skipping this run.", UserWarning)
            return None

        # the dataframe contains the following columns: id (list of atom indices), type (list of int representing
        # atom type, x (list of x cartesian coordinates for each atom), y, z, fx (list forces in direction x for each
        # atom), potential_energy (1 float).
        # Each row is a different MD step / usable example for diffusion model
        # TODO consider filtering out samples with large forces and MD steps that are too similar
        # TODO large force and similar are to be defined
        df = df[['type', 'x', 'y', 'z', 'box', 'potential_energy']]
        df = self.get_x_relative(df)  # add relative coordinates
        df['natom'] = df['type'].apply(lambda x: len(x))  # count number of atoms in a structure

        # Parquet cannot handle a list of list; flattening positions.
        df['position'] = df.apply(self._flatten_positions_in_row, axis=1)
        # position is natom * 3 array
        # TODO: position (singular) and relative_positions (plural) are not consistent.
        return df[['natom', 'box', 'type', 'position', 'relative_positions', 'potential_energy']]

    @staticmethod
    def _flatten_positions_in_row(row: pd.Series) -> List[float]:
        """Function to flatten the positions in a dataframe row.

        Args:
            row : a dataframe row that should contain columns x, y, z.

        Returns:
            flattened positions: a list of each element is the flattened coordinate for that row, in C-style.
        """
        list_x = row['x']
        list_y = row['y']
        list_z = row['z']

        flat_positions = list(itertools.chain.from_iterable([[x, y, z] for x, y, z in zip(list_x, list_y, list_z)]))

        return flat_positions
