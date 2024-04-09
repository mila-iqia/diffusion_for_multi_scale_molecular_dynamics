from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from crystal_diffusion.analysis import PLEASANT_FIG_SIZE, PLOT_STYLE_PATH

plt.style.use(PLOT_STYLE_PATH)

current_directory = Path(__file__).parent
pickle_path = current_directory.joinpath('energy_samples.pkl')

if __name__ == '__main__':

    df = pd.read_pickle(pickle_path)

    fig = plt.figure(figsize=PLEASANT_FIG_SIZE)

    small_df = df[df['energy'] < 100.]

    fig.suptitle(f'Sample Energy Distribution\n Total Count = {len(df)}, Count < 100 eV = {len(small_df)}')

    ax1 = fig.add_subplot(111)

    common_params = dict(density=False, bins=50, histtype="stepfilled", alpha=0.25)
    ax1.hist(small_df['energy'], **common_params, label='Samples', color='green')
    ax1.set_xlim([0, 100])
    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('Count')
    plt.show()
