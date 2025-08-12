from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diffusion_for_multi_scale_molecular_dynamics import TOP_DIR
from diffusion_for_multi_scale_molecular_dynamics.generators.sampling_constraint import (
    SamplingConstraint, write_sampling_constraint)

results_top_dir = Path("/Users/brunorousseau/courtois/july26/active_learning/amorphous_silicon")
oracle_path = results_top_dir / "excise_and_repaint_3x3x3/output/run1/campaign_1/round_1/oracle"

output_path = TOP_DIR / "experiments/active_learning/amorphous_silicon/analysis/sampling_constraint.pkl"


if __name__ == '__main__':
    elements = ['Si']
    df = pd.read_pickle(oracle_path / "oracle_single_point_calculations.pkl")
    structure = df.iloc[0]['structure']

    constrained_mask = np.array(structure.site_properties['constrained']).astype(bool)

    constrained_atom_types = torch.zeros(constrained_mask.sum(), dtype=torch.long)

    constrained_relative_coordinates = torch.from_numpy(structure.frac_coords[constrained_mask]).float()

    sampling_constraint = SamplingConstraint(elements=elements,
                                             constrained_relative_coordinates=constrained_relative_coordinates,
                                             constrained_atom_types=constrained_atom_types)

    write_sampling_constraint(sampling_constraint, output_path)
