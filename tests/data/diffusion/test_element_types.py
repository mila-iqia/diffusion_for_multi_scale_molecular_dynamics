import numpy as np
import pytest

from diffusion_for_multi_scale_molecular_dynamics.data.element_types import (
    NULL_ELEMENT, NULL_ELEMENT_ID, ElementTypes)
from tests.fake_data_utils import generate_random_string


class TestElementTypes:

    @pytest.fixture()
    def num_atom_types(self):
        return 4

    @pytest.fixture
    def unique_elements(self, num_atom_types):
        return [generate_random_string(size=3) for _ in range(num_atom_types)]

    @pytest.fixture
    def bad_element(self):
        return "this_is_a_bad_element"

    @pytest.fixture
    def bad_element_id(self):
        return 9999

    @pytest.fixture
    def element_types(self, unique_elements):
        return ElementTypes(unique_elements)

    def test_number_of_atom_types(self, element_types, num_atom_types):
        assert element_types.number_of_atom_types == num_atom_types

    def test_get_element_id(self, element_types, unique_elements):
        assert element_types.get_element_id(NULL_ELEMENT) == NULL_ELEMENT_ID

        computed_element_ids = [element_types.get_element_id(element) for element in unique_elements]
        assert len(np.unique(computed_element_ids)) == len(unique_elements)

    def test_get_element_id_bad_element(self, element_types, bad_element):
        with pytest.raises(KeyError):
            element_types.get_element_id(bad_element)

    def test_get_element(self, element_types, unique_elements):
        assert element_types.get_element(NULL_ELEMENT_ID) == NULL_ELEMENT

        for element in unique_elements:
            computed_element_id = element_types.get_element_id(element)
            assert element == element_types.get_element(computed_element_id)

    def test_get_element_bad_element_id(self, element_types, bad_element_id):
        with pytest.raises(KeyError):
            element_types.get_element(bad_element_id)
