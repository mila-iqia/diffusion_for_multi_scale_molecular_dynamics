from collections import defaultdict

import torch


class SampleTrajectory:
    """Sample Trajectory.

    This class aims to record all details of the diffusion sampling process. The goal is to produce
    an artifact that can then be analyzed off-line.
    """

    def __init__(self):
        """Init method."""
        self.data = defaultdict(list)

    def reset(self):
        """Reset data structure."""
        self.data = defaultdict(list)

    def record_unit_cell(self, unit_cell: torch.Tensor):
        """Record unit cell."""
        self.data['unit_cell'] = unit_cell.detach().cpu()

    def record_predictor_step(self, i_index: int, time: float, sigma: float,
                              x_i: torch.Tensor, x_im1: torch.Tensor, scores: torch.Tensor):
        """Record predictor step."""
        self.data['predictor_i_index'].append(i_index)
        self.data['predictor_time'].append(time)
        self.data['predictor_sigma'].append(sigma)
        self.data['predictor_x_i'].append(x_i.detach().cpu())
        self.data['predictor_x_im1'].append(x_im1.detach().cpu())
        self.data['predictor_scores'].append(scores.detach().cpu())

    def record_corrector_step(self, i_index: int, time: float, sigma: float,
                              x_i: torch.Tensor, corrected_x_i: torch.Tensor, scores: torch.Tensor):
        """Record corrector step."""
        self.data['corrector_i_index'].append(i_index)
        self.data['corrector_time'].append(time)
        self.data['corrector_sigma'].append(sigma)
        self.data['corrector_x_i'].append(x_i.detach().cpu())
        self.data['corrector_corrected_x_i'].append(corrected_x_i.detach().cpu())
        self.data['corrector_scores'].append(scores.detach().cpu())

    def write_to_pickle(self, path_to_pickle: str):
        """Write data to pickle file."""
        with open(path_to_pickle, 'wb') as fd:
            torch.save(self.data, fd)

    @staticmethod
    def read_from_pickle(path_to_pickle: str):
        """Read from pickle."""
        with open(path_to_pickle, 'rb') as fd:
            sample_trajectory = SampleTrajectory()
            sample_trajectory.data = torch.load(fd)
        return sample_trajectory
