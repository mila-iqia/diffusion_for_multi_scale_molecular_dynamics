import pytest

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.ordered_elements import \
    sort_elements_by_atomic_mass


@pytest.fixture()
def list_elements():
    return ['Ca', 'Si', 'H', 'C']


@pytest.fixture()
def expected_sorted_list_elements():
    return ['H', 'C', 'Si', 'Ca']


def test_sort_elements_by_atomic_mass(list_elements, expected_sorted_list_elements):
    computed_sorted_list_elements = sort_elements_by_atomic_mass(list_elements)
    assert computed_sorted_list_elements == expected_sorted_list_elements
