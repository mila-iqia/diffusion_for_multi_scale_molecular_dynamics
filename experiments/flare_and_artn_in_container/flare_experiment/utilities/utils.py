from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd
import yaml
from ase import Atoms
from yaml import CLoader


def extract_data_from_document(document: dict):
    """Let's make some assumtions on the form of the document."""
    columns = document['keywords']
    data = document['data']
    df = pd.DataFrame(data=data, columns=columns).sort_values(by='id')
    cell = np.array([bounds[1] for bounds in document['box']])
    thermo_dict = parse_thermo_fields(document)
    return df, cell, thermo_dict


def parse_thermo_fields(document: Dict):
    assert 'thermo' in document
    keywords = document['thermo'][0]['keywords']
    data = document['thermo'][1]['data']
    results = {k:v for k,v in zip(keywords, data)}
    return results


def get_atoms(df, cell):
    expected_keywords = {"element", "x", "y", "z"}
    assert expected_keywords.issubset(set(df.columns))
    symbols = df['element'].values
    positions = df[['x', 'y', 'z']].values
    atoms = Atoms(positions=positions, symbols=symbols, cell=cell, pbc=[True, True, True])
    return atoms

def get_forces(df):
    expected_keywords = {"fx", "fy", "fz"}
    assert expected_keywords.issubset(set(df.columns))
    forces = df[['fx', 'fy', 'fz']].values
    return forces

def get_uncertainty(df):
    expected_keywords = {"c_unc"}
    assert expected_keywords.issubset(set(df.columns))
    uncertainties = df["c_unc"].values
    return uncertainties


def parse_lammps_dump(lammps_dump: str):
    """Parse lammps dump."""
    results = defaultdict(list)
    with open(lammps_dump, "r") as stream:
        dump_yaml = yaml.load_all(stream, Loader=CLoader)

        for doc in dump_yaml:  # loop over MD steps
            df, cell, thermo_dict = extract_data_from_document(doc)
            atoms = get_atoms(df, cell)
            results['atoms'].append(atoms)

            forces = get_forces(df)
            results['forces'].append(forces)

            if 'PotEng' in thermo_dict:
                results['energy'].append(thermo_dict['PotEng'])

            if 'c_unc' in df.columns:
                results['uncertainties'].append(get_uncertainty(df))

    return results
