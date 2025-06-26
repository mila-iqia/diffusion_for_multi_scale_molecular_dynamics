import numpy as np
import pytest

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.atom_selector.threshold_atom_selector import (
    ThresholdAtomSelector, ThresholdAtomSelectorParameters)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.atom_selector.top_k_atom_selector import (
    TopKAtomSelector, TopKAtomSelectorParameters)


@pytest.fixture(params=[1, 3, 5, 12])
def top_k_value(request):
    return request.param


@pytest.fixture(params=[0.1, 0.5, 0.9])
def uncertainty_threshold(request):
    return request.param


@pytest.fixture
def top_k_atom_selector(top_k_value):
    parameters = TopKAtomSelectorParameters(top_k_environment=top_k_value)
    return TopKAtomSelector(parameters)


@pytest.fixture
def threshold_atom_selector(uncertainty_threshold):
    parameters = ThresholdAtomSelectorParameters(uncertainty_threshold=uncertainty_threshold)
    return ThresholdAtomSelector(parameters)


@pytest.fixture
def number_of_atoms():
    return 20


@pytest.fixture
def uncertainty_per_atom(number_of_atoms):
    return np.random.random(number_of_atoms)


@pytest.fixture
def expected_threshold_atom_indices(uncertainty_threshold, uncertainty_per_atom):
    uncertainties, atom_idx = [], []
    for idx, u in enumerate(uncertainty_per_atom):
        if u > uncertainty_threshold:
            atom_idx.append(idx)
            uncertainties.append(u)
    # sort on uncertainty (highest to lowest)
    sorted_pairs = sorted(
        zip(uncertainties, atom_idx), key=lambda item: item[0], reverse=True
    )
    expected_atom_idx = np.array([item[1] for item in sorted_pairs])
    return expected_atom_idx


def test_uncertainty_threshold_atom_selection(expected_threshold_atom_indices,
                                              uncertainty_per_atom, threshold_atom_selector):
    calculated_atom_indices = threshold_atom_selector.select_central_atoms(uncertainty_per_atom)
    assert np.array_equal(calculated_atom_indices, expected_threshold_atom_indices)


@pytest.fixture
def expected_top_k_atom_indices(top_k_value, uncertainty_per_atom):
    uncertainties = uncertainty_per_atom.tolist()
    atom_idx = range(len(uncertainties))

    # sort on uncertainty (highest to lowest)
    _, expected_atom_idx = zip(*sorted(zip(uncertainties, atom_idx), reverse=True))
    expected_atom_idx = expected_atom_idx[:top_k_value]
    return expected_atom_idx


def test_uncertainty_top_k_atom_selection(expected_top_k_atom_indices, uncertainty_per_atom, top_k_atom_selector):
    calculated_atom_indices = top_k_atom_selector.select_central_atoms(uncertainty_per_atom)
    assert np.array_equal(calculated_atom_indices, calculated_atom_indices)
