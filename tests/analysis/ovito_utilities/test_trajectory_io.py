from pathlib import Path

import einops
import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.analysis.ovito_utilities.trajectory_io import (
    create_cif_files, create_xyz_files)
from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL


@pytest.fixture()
def number_of_atoms():
    return 8


@pytest.fixture()
def spatial_dimension():
    return 3


@pytest.fixture()
def number_of_time_steps():
    return 10


@pytest.fixture(params=[1, 5])
def number_of_trajectories(request):
    return request.param


@pytest.fixture()
def trajectory_index(number_of_trajectories):
    if number_of_trajectories == 1:
        return None
    else:
        return number_of_trajectories // 2


@pytest.fixture()
def elements():
    return ['Si']


@pytest.fixture()
def trajectory_axl_compositions(number_of_trajectories, number_of_time_steps, number_of_atoms, spatial_dimension):
    x = torch.rand(number_of_trajectories, number_of_time_steps, number_of_atoms, spatial_dimension).squeeze(0)
    a = torch.ones(number_of_trajectories, number_of_time_steps, number_of_atoms).squeeze(0)
    basis_vectors = einops.repeat(torch.eye(spatial_dimension),
                                  "d1 d2 -> t b d1 d2",
                                  t=number_of_trajectories,
                                  b=number_of_time_steps).squeeze(0)
    return AXL(A=a, X=x, L=basis_vectors)


@pytest.fixture()
def scores(number_of_trajectories, number_of_time_steps, number_of_atoms, spatial_dimension):
    scores = torch.randn(number_of_trajectories, number_of_time_steps, number_of_atoms, spatial_dimension).squeeze(0)
    return scores


@pytest.fixture(params=[True, False])
def atomic_properties(scores, request):
    if request.param:
        return dict(scores=scores)
    else:
        return None


def test_smoke_create_xyz_files(elements, trajectory_index, trajectory_axl_compositions, atomic_properties, tmpdir):
    create_xyz_files(elements,
                     visualization_artifacts_path=Path(tmpdir),
                     trajectory_index=trajectory_index,
                     trajectory_axl_compositions=trajectory_axl_compositions,
                     atomic_properties=atomic_properties)


def test_smoke_create_cif_files(elements, trajectory_index, trajectory_axl_compositions, tmpdir):
    create_cif_files(elements,
                     visualization_artifacts_path=Path(tmpdir),
                     trajectory_index=trajectory_index,
                     trajectory_axl_compositions=trajectory_axl_compositions)
