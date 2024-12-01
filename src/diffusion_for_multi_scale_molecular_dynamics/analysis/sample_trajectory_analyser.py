import logging
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import einops
import numpy as np
import torch

from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_scheduler import \
    NoiseScheduler

logger = logging.getLogger(__name__)


class SampleTrajectoryAnalyser:
    """Sample Trajectory Analyser.

    This class reads in a trajectory recording pickle and processes the data to make it easy to analyse.
    """
    def __init__(self, pickle_path: Path, num_classes: int):
        """Init method.

        Args:
            pickle_path:  path to recording pickle.
            num_classes: number of classes (including the MASK class).
        """
        logger.info("Reading data from pickle file.")
        data = torch.load(pickle_path, map_location=torch.device("cpu"))
        logger.info("Done reading data.")

        noise_parameters = NoiseParameters(**data['noise_parameters'])
        sampler = NoiseScheduler(noise_parameters, num_classes=num_classes)
        self.noise, _ = sampler.get_all_sampling_parameters()

        self.time_index_key = 'time_step_index'
        self.axl_keys = ['composition_i', 'composition_im1', 'model_predictions_i']

        self._predictor_data = data["predictor_step"]

        del data

    def extract_axl(self, axl_key: str) -> Tuple[np.ndarray, AXL]:
        """Extract AXL.

        Args:
            axl_key: name of field to be extracted

        Returns:
            time_indices: an array containing the time indices of the AXL.
            axl: the axl described in the axl_key, where the fields have dimension [nsample, ntimes, ...]
        """
        # The recording might have taken place over multiple batches. Combine corresponding compositions.
        assert axl_key in self.axl_keys, f"Unknown axl key '{axl_key}'"
        multiple_batch = defaultdict(list)

        logger.info("Iterating over entries")
        list_time_indices = []
        for entry in self._predictor_data:
            time_index = entry["time_step_index"]
            list_time_indices.append(time_index)
            axl = entry[axl_key]
            multiple_batch[time_index].append(axl)

        time_indices = np.sort(np.unique(np.array(list_time_indices)))

        logger.info("Stacking multiple batch over time")
        list_stacked_axl = []
        for time_index in time_indices:
            list_axl = multiple_batch[time_index]
            stacked_axl = AXL(
                A=torch.vstack([axl.A for axl in list_axl]),
                X=torch.vstack([axl.X for axl in list_axl]),
                L=torch.vstack([axl.L for axl in list_axl]),
            )
            list_stacked_axl.append(stacked_axl)

        logger.info("Rearrange dimensions")
        a = einops.rearrange([axl.A for axl in list_stacked_axl], "time batch ... -> batch time ...")
        x = einops.rearrange([axl.X for axl in list_stacked_axl], "time batch ... -> batch time ...")
        lattice = einops.rearrange([axl.L for axl in list_stacked_axl], "time batch ... -> batch time ...")
        return time_indices, AXL(A=a, X=x, L=lattice)
