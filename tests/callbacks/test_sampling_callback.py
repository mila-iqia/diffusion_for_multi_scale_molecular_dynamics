from unittest.mock import MagicMock

import numpy as np
import pytest
from pytorch_lightning import LightningModule

from crystal_diffusion.callbacks.sampling_callback import (
    DiffusionSamplingCallback, ODESamplingParameters,
    PredictorCorrectorSamplingParameters)
from crystal_diffusion.samplers.variance_sampler import NoiseParameters


@pytest.mark.parametrize("total_time_steps", [1])
@pytest.mark.parametrize("time_delta", [0.1])
@pytest.mark.parametrize("sigma_min", [0.15])
@pytest.mark.parametrize("corrector_step_epsilon", [0.25])
@pytest.mark.parametrize("number_of_samples", [8])
@pytest.mark.parametrize("unit_cell_size", [10])
@pytest.mark.parametrize("lammps_energy", [2])
@pytest.mark.parametrize("spatial_dimension", [3])
@pytest.mark.parametrize("number_of_atoms", [4])
@pytest.mark.parametrize("sample_batchsize", [None, 8, 4])
@pytest.mark.parametrize("record_samples", [True, False])
class TestSamplingCallback:

    @pytest.fixture(params=['predictor_corrector', 'ode'])
    def algorithm(self, request):
        return request.param

    @pytest.fixture()
    def number_of_corrector_steps(self, algorithm):
        if algorithm == 'predictor_corrector':
            return 1
        else:
            return 0

    @pytest.fixture()
    def mock_create_generator(self):
        generator = MagicMock()
        return generator

    @pytest.fixture()
    def mock_create_create_unit_cell(self, number_of_samples):
        unit_cell = np.arange(number_of_samples)  # Dummy unit cell
        return unit_cell

    @pytest.fixture()
    def mock_compute_lammps_energies(self, lammps_energy):
        return np.ones((1,)) * lammps_energy

    @pytest.fixture()
    def noise_parameters(self, total_time_steps, time_delta, sigma_min, corrector_step_epsilon):
        noise_parameters = NoiseParameters(total_time_steps=total_time_steps,
                                           time_delta=time_delta,
                                           sigma_min=sigma_min,
                                           corrector_step_epsilon=corrector_step_epsilon)
        return noise_parameters

    @pytest.fixture()
    def sampling_parameters(self, algorithm, spatial_dimension, number_of_corrector_steps,
                            number_of_atoms, number_of_samples, sample_batchsize, unit_cell_size, record_samples):
        if algorithm == 'predictor_corrector':
            sampling_parameters = (
                PredictorCorrectorSamplingParameters(spatial_dimension=spatial_dimension,
                                                     number_of_corrector_steps=number_of_corrector_steps,
                                                     number_of_atoms=number_of_atoms,
                                                     number_of_samples=number_of_samples,
                                                     sample_batchsize=sample_batchsize,
                                                     cell_dimensions=[unit_cell_size for _ in range(spatial_dimension)],
                                                     record_samples=record_samples))
        elif algorithm == 'ode':
            sampling_parameters = (
                ODESamplingParameters(spatial_dimension=spatial_dimension,
                                      number_of_atoms=number_of_atoms,
                                      number_of_samples=number_of_samples,
                                      sample_batchsize=sample_batchsize,
                                      cell_dimensions=[unit_cell_size for _ in range(spatial_dimension)],
                                      record_samples=record_samples))

        else:
            raise NotImplementedError

        return sampling_parameters

    @pytest.fixture()
    def pl_model(self):
        return MagicMock(spec=LightningModule)

    def test_sample_and_evaluate_energy(self, mocker, mock_compute_lammps_energies, mock_create_generator,
                                        mock_create_create_unit_cell, noise_parameters, sampling_parameters,
                                        pl_model, sample_batchsize, number_of_samples, tmpdir):
        sampling_cb = DiffusionSamplingCallback(
            noise_parameters=noise_parameters,
            sampling_parameters=sampling_parameters,
            output_directory=tmpdir)
        mocker.patch.object(sampling_cb, "_create_generator", return_value=mock_create_generator)
        mocker.patch.object(sampling_cb, "_create_unit_cell", return_value=mock_create_create_unit_cell)
        mocker.patch.object(sampling_cb, "_compute_oracle_energies", return_value=mock_compute_lammps_energies)

        sample_energies = sampling_cb.sample_and_evaluate_energy(pl_model)
        assert isinstance(sample_energies, np.ndarray)
        # each call of compute lammps energy yields a np.array of size 1
        expected_size = int(number_of_samples / sample_batchsize) if sample_batchsize is not None else 1
        assert sample_energies.shape[0] == expected_size
