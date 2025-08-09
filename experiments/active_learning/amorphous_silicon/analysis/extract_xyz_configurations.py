from pathlib import Path

import pandas as pd

from diffusion_for_multi_scale_molecular_dynamics.analysis.ovito_utilities.generated_samples_io import \
    write_active_learning_generated_sample

results_top_dir = Path("/Users/brunorousseau/courtois/july26/active_learning/amorphous_silicon")

# oracle_path = results_top_dir / "excise_and_repaint/output/run1/campaign_1/round_1/oracle"
# viz_path = results_top_dir / "visualization/excise_and_repaint/run1/campaign_1/round_1/"

oracle_path = results_top_dir / "excise_and_repaint_3x3x3/output/run1/campaign_1/round_1/oracle"
viz_path = results_top_dir / "visualization/excise_and_repaint_3x3x3/run1/campaign_1/round_1/"

viz_path.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':

    df = pd.read_pickle(oracle_path / "oracle_single_point_calculations.pkl")

    for idx, structure in enumerate(df['structure'].values):
        output_path = viz_path / f"generated_structure_{idx}.xyz"
        write_active_learning_generated_sample(structure, output_path)
