from collections import defaultdict
from typing import Any, AnyStr, Dict

import einops
import torch

from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, AXL_NAME_DICT)


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
        self.data["unit_cell"] = unit_cell.detach().cpu()

    def standardize_data(self, data: Dict[AnyStr, Any]) -> Dict[AnyStr, Any]:
        """Method to transform the recorded data to a standard form."""
        raise NotImplementedError("Must be implemented in child class.")

    def write_to_pickle(self, path_to_pickle: str):
        """Write standardized data to pickle file."""
        standard_data = self.standardize_data(self.data)
        with open(path_to_pickle, "wb") as fd:
            torch.save(standard_data, fd)


class ODESampleTrajectory(SampleTrajectory):
    """ODE Sample Trajectory.

    This class aims to record all details of the ODE diffusion sampling process. The goal is to produce
    an artifact that can then be analyzed off-line.
    """

    def record_ode_solution(
        self,
        times: torch.Tensor,
        sigmas: torch.Tensor,
        relative_coordinates: torch.Tensor,
        normalized_scores: torch.Tensor,
        stats: Dict,
        status: torch.Tensor,
    ):
        """Record ODE solution information."""
        self.data["time"].append(times)
        self.data["sigma"].append(sigmas)
        self.data["relative_coordinates"].append(relative_coordinates)
        self.data["normalized_scores"].append(normalized_scores)
        self.data["stats"].append(stats)
        self.data["status"].append(status)

    def standardize_data(self, data: Dict[AnyStr, Any]) -> Dict[AnyStr, Any]:
        """Method to transform the recorded data to a standard form."""
        extra_fields = ["stats", "status"]
        standardized_data = dict(
            unit_cell=data["unit_cell"],
            time=data["time"][0],
            sigma=data["sigma"][0],
            relative_coordinates=data["relative_coordinates"][0],
            normalized_scores=data["normalized_scores"][0],
            extra={key: data[key][0] for key in extra_fields},
        )
        return standardized_data


class SDESampleTrajectory(SampleTrajectory):
    """SDE Sample Trajectory.

    This class aims to record all details of the SDE diffusion sampling process. The goal is to produce
    an artifact that can then be analyzed off-line.
    """

    def record_sde_solution(
        self,
        times: torch.Tensor,
        sigmas: torch.Tensor,
        relative_coordinates: torch.Tensor,
        normalized_scores: torch.Tensor,
    ):
        """Record ODE solution information."""
        self.data["time"].append(times)
        self.data["sigma"].append(sigmas)
        self.data["relative_coordinates"].append(relative_coordinates)
        self.data["normalized_scores"].append(normalized_scores)

    def standardize_data(self, data: Dict[AnyStr, Any]) -> Dict[AnyStr, Any]:
        """Method to transform the recorded data to a standard form."""
        standardized_data = dict(
            unit_cell=data["unit_cell"],
            time=data["time"][0],
            sigma=data["sigma"][0],
            relative_coordinates=data["relative_coordinates"][0],
            normalized_scores=data["normalized_scores"][0],
        )
        return standardized_data


class NoOpODESampleTrajectory(ODESampleTrajectory):
    """A sample trajectory object that performs no operation."""

    def record_unit_cell(self, unit_cell: torch.Tensor):
        """No Op."""
        return

    def record_ode_solution(
        self,
        times: torch.Tensor,
        sigmas: torch.Tensor,
        relative_coordinates: torch.Tensor,
        normalized_scores: torch.Tensor,
        stats: Dict,
        status: torch.Tensor,
    ):
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

    def record_predictor_step(
        self,
        i_index: int,
        time: float,
        sigma: float,
        composition_i: AXL,
        composition_im1: AXL,
        model_predictions_i: AXL,
    ):
        """Record predictor step."""
        self.data["predictor_i_index"].append(i_index)
        self.data["predictor_time"].append(time)
        self.data["predictor_sigma"].append(sigma)
        for axl_field, axl_name in AXL_NAME_DICT.items():
            self.data[f"predictor_{axl_name}_i"].append(
                getattr(composition_i, axl_field).detach().cpu()
            )
            self.data[f"predictor_{axl_name}_im1"].append(
                getattr(composition_im1, axl_field).detach().cpu()
            )
            self.data[f"predictor_{axl_name}_model_predictions"].append(
                getattr(model_predictions_i, axl_field).detach().cpu()
            )

    def record_corrector_step(
        self,
        i_index: int,
        time: float,
        sigma: float,
        composition_i: AXL,
        corrected_composition_i: AXL,
        model_predictions_i: AXL,
    ):
        """Record corrector step."""
        self.data["corrector_i_index"].append(i_index)
        self.data["corrector_time"].append(time)
        self.data["corrector_sigma"].append(sigma)
        for axl_field, axl_name in AXL_NAME_DICT.items():
            self.data[f"corrector_{axl_name}_i"].append(
                getattr(composition_i, axl_field).detach().cpu()
            )
            self.data[f"corrector_{axl_name}_corrected_i"].append(
                getattr(corrected_composition_i, axl_field).detach().cpu()
            )
            self.data[f"corrector_{axl_name}_model_predictions"].append(
                getattr(model_predictions_i, axl_field).detach().cpu()
            )

    def standardize_data(self, data: Dict[AnyStr, Any]) -> Dict[AnyStr, Any]:
        """Method to transform the recorded data to a standard form."""
        predictor_relative_coordinates = einops.rearrange(
            torch.stack(data[f"predictor_{AXL_NAME_DICT['X']}_i"]), "t b n d -> b t n d"
        )
        predictor_normalized_scores = einops.rearrange(
            torch.stack(data[f"predictor_{AXL_NAME_DICT['X']}_model_predictions"]),
            "t b n d -> b t n d",
        )

        extra_fields = [
            "predictor_i_index",
            "corrector_i_index",
            "corrector_time",
            "corrector_sigma",
            "corrector_scores",
        ]
        extra_fields += [f"predictor_{v}_i" for v in AXL_NAME_DICT.values()]
        extra_fields += [f"predictor_{v}_im1" for v in AXL_NAME_DICT.values()]
        extra_fields += [f"corrector_{v}_i" for v in AXL_NAME_DICT.values()]
        extra_fields += [f"corrector_{v}_corrected_i" for v in AXL_NAME_DICT.values()]
        extra_fields += [f"corrector_{v}_model_outputs" for v in AXL_NAME_DICT.values()]

        standardized_data = dict(
            unit_cell=data["unit_cell"],
            time=torch.tensor(data["predictor_time"]),
            sigma=torch.tensor(data["predictor_sigma"]),
            relative_coordinates=predictor_relative_coordinates,
            normalized_scores=predictor_normalized_scores,
            extra={key: data[key] for key in extra_fields},
        )
        return standardized_data


class NoOpPredictorCorrectorSampleTrajectory(PredictorCorrectorSampleTrajectory):
    """A sample trajectory object that performs no operation."""

    def record_unit_cell(self, unit_cell: torch.Tensor):
        """No Op."""
        return

    def record_predictor_step(
        self,
        i_index: int,
        time: float,
        sigma: float,
        composition_i: AXL,
        composition_im1: AXL,
        model_predictions_i: AXL,
    ):
        """No Op."""
        return

    def record_corrector_step(
        self,
        i_index: int,
        time: float,
        sigma: float,
        composition_i: AXL,
        corrected_composition_i: AXL,
        model_predictions_i: AXL,
    ):
        """No Op."""
        return

    def write_to_pickle(self, path_to_pickle: str):
        """No Op."""
        return
