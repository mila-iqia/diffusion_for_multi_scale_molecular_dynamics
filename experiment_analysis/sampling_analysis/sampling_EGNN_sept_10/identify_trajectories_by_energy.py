import glob
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from crystal_diffusion.analysis import PLOT_STYLE_PATH
from crystal_diffusion.utils.logging_utils import setup_analysis_logger

plt.style.use(PLOT_STYLE_PATH)

logger = logging.getLogger(__name__)
setup_analysis_logger()

results_dir = Path("/Users/bruno/courtois/partial_trajectory_sampling_EGNN_sept_10/partial_samples_EGNN_Sept_10")


tf = 1.0

if __name__ == '__main__':

    list_rows = []
    for pickle_path in glob.glob(str(results_dir / 'diffusion_energies_sample_time=*.pt')):
        energies = torch.load(pickle_path).numpy()
        time = float(pickle_path.split('=')[1].split('.pt')[0])

        for idx, energy in enumerate(energies):
            row = dict(tf=time, trajectory_index=idx, energy=energy)
            list_rows.append(row)

    df = pd.DataFrame(list_rows).sort_values(by=['tf', 'energy'])

    groups = df.groupby('tf')

    sub_df = groups.get_group(tf)
