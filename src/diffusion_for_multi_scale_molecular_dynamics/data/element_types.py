from typing import Dict, List

NULL_ELEMENT = "NULL_ELEMENT_FOR_PADDING"
NULL_ELEMENT_ID = -1


class ElementTypes:
    """Element Types.

    This class manages the relationship between strings that identify elements (Si, Ge, Li, etc...)
    and their integer indices.
    """

    def __init__(self, elements: List[str]):
        """Init method.

        Args:
            elements: list all the elements that could be present in the data.
        """
        self.validate_elements(elements)
        self._elements = sorted(elements)
        self._ids = list(range(len(self._elements)))

        self._element_to_id_map: Dict[str, int] = {
            k: v for k, v in zip(self._elements, self._ids)
        }
        self._id_to_element_map: Dict[int, str] = {
            k: v for k, v in zip(self._ids, self._elements)
        }

        self._element_to_id_map[NULL_ELEMENT] = NULL_ELEMENT_ID
        self._id_to_element_map[NULL_ELEMENT_ID] = NULL_ELEMENT

    @staticmethod
    def validate_elements(elements: List[str]):
        """Validate elements."""
        assert NULL_ELEMENT not in elements, f"The element '{NULL_ELEMENT}' is reserved and should not be used."
        assert len(set(elements)) == len(elements), "Each entry in the elements list should be unique."

    @property
    def number_of_atom_types(self) -> int:
        """Number of atom types."""
        return len(self._elements)

    @property
    def elements(self) -> List[str]:
        """The sorted elements."""
        return self._elements

    @property
    def element_ids(self) -> List[int]:
        """The sorted elements."""
        return self._ids

    def get_element(self, element_id: int) -> str:
        """Get element.

        Args:
            element_id : integer index.

        Returns:
            element: string representing the element
        """
        return self._id_to_element_map[element_id]

    def get_element_id(self, element: str) -> int:
        """Get element id.

        Args:
            element: string representing the element

        Returns:
            element_id : integer index.
        """
        return self._element_to_id_map[element]
