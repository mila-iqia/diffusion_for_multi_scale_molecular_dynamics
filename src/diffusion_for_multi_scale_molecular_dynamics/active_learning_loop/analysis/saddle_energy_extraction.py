import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.artn.artn_outputs import \
    get_saddle_energy


def extract_all_saddle_energies(top_experiment_directory: Path):
    """Extract all saddle energies.

    Args:
        top_experiment_directory (Path): directory where all the runs are located.

    Returns:
        df: a pandas dataframe with all the extracted saddle energies. Failed jobs have a NaN energy.
    """
    campaign_path_pattern = str(top_experiment_directory / "run*/campaign*")

    list_all_campaign_directories = glob.glob(campaign_path_pattern, recursive=True)

    regex_pattern = r".*run(?P<run_id>\d*).*campaign_(?P<campaign_id>\d*)"

    list_rows = []
    for campaign_directory in tqdm(list_all_campaign_directories, "CAMPAIGNS"):
        match = re.search(regex_pattern, campaign_directory)
        run_id = int(match.group('run_id'))
        campaign_id = int(match.group('campaign_id'))

        campaign_path = Path(campaign_directory)

        final_round, threshold = _get_campaign_details(campaign_path)

        saddle_energy = _get_saddle_energy(campaign_path, final_round)

        row = dict(run_id=run_id, campaign_id=campaign_id, final_round=final_round,
                   threshold=threshold, saddle_energy=saddle_energy)
        list_rows.append(row)

    df = pd.DataFrame(list_rows).sort_values(by=["run_id", "campaign_id"])
    df.reset_index(drop=True, inplace=True)
    return df


def _get_campaign_details(campaign_path: Path):
    """Get campaign details."""
    campaign_details_file_path = campaign_path / "campaign_details.yaml"
    if campaign_details_file_path.is_file():
        with open(campaign_path / "campaign_details.yaml", "r") as fd:
            campaign_details = yaml.load(fd, Loader=yaml.FullLoader)

        final_round = campaign_details["final_round"]
        threshold = campaign_details["uncertainty_threshold"]
    else:
        # Something crashed.
        final_round = np.NaN
        threshold = np.NaN

    return final_round, threshold


def _get_saddle_energy(campaign_path: Path, final_round):
    """Get saddle energy."""
    if final_round is np.NaN:
        # something crashed
        return np.NaN

    final_round_dir = campaign_path / f"round_{final_round}"
    artn_output_file = final_round_dir / "lammps_artn/artn.out"
    try:
        with open(artn_output_file, "r") as fd:
            artn_output = fd.read()
            saddle_energy = get_saddle_energy(artn_output)
    except Exception:
        print("Failed to Extract Saddle Energy")
        saddle_energy = np.NaN

    return saddle_energy
