from collections import defaultdict
from typing import Dict

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

    def write_to_pickle(self, path_to_pickle: str):
        """Write data to pickle file."""
        with open(path_to_pickle, 'wb') as fd:
            torch.save(self.data, fd)


class ODESampleTrajectory(SampleTrajectory):
    """ODE Sample Trajectory.

    This class aims to record all details of the ODE diffusion sampling process. The goal is to produce
    an artifact that can then be analyzed off-line.
    """

    def record_ode_solution(self, times: torch.Tensor, relative_coordinates: torch.Tensor,
                            stats: Dict, status: torch.Tensor):
        """Record ODE solution information."""
        self.data['time'].append(times)
        self.data['stats'].append(stats)
        self.data['status'].append(status)
        self.data['relative_coordinates'].append(relative_coordinates)

    @staticmethod
    def read_from_pickle(path_to_pickle: str):
        """Read from pickle."""
        with open(path_to_pickle, 'rb') as fd:
            sample_trajectory = ODESampleTrajectory()
            sample_trajectory.data = torch.load(fd, map_location=torch.device('cpu'))
        return sample_trajectory


class NoOpODESampleTrajectory(ODESampleTrajectory):
    """A sample trajectory object that performs no operation."""

    def record_unit_cell(self, unit_cell: torch.Tensor):
        """No Op."""
        return

    def record_ode_solution(self, times: torch.Tensor, relative_coordinates: torch.Tensor,
                            stats: Dict, status: torch.Tensor):
        """No Op."""
        return

    def write_to_pickle(self, path_to_pickle: str):
        """No Op."""
        return


class PredictorCorrectorSampleTrajectory(SampleTrajectory):
    """Predictor Corrector Sample Trajectory.

    This class aims to record all details of the predictor-corrector diffusion sampling process. The goal is to produce
    an artifact that can then be analyzed off-line.
    """
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

    @staticmethod
    def read_from_pickle(path_to_pickle: str):
        """Read from pickle."""
        with open(path_to_pickle, 'rb') as fd:
            sample_trajectory = PredictorCorrectorSampleTrajectory()
            sample_trajectory.data = torch.load(fd, map_location=torch.device('cpu'))
        return sample_trajectory


class NoOpPredictorCorrectorSampleTrajectory(PredictorCorrectorSampleTrajectory):
    """A sample trajectory object that performs no operation."""

    def record_unit_cell(self, unit_cell: torch.Tensor):
        """No Op."""
        return

    def record_predictor_step(self, i_index: int, time: float, sigma: float,
                              x_i: torch.Tensor, x_im1: torch.Tensor, scores: torch.Tensor):
        """No Op."""
        return

    def record_corrector_step(self, i_index: int, time: float, sigma: float,
                              x_i: torch.Tensor, corrected_x_i: torch.Tensor, scores: torch.Tensor):
        """No Op."""
        return

    def write_to_pickle(self, path_to_pickle: str):
        """No Op."""
        return
