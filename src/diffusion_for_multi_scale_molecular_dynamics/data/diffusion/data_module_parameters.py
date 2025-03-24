from dataclasses import dataclass
from typing import Optional


@dataclass(kw_only=True)
class DataModuleParameters:
    """Base Hyper-parameters for Data Modules."""

    # The data source must be specified in the concrete class.
    data_source = None

    # Either batch_size XOR train_batch_size and valid_batch_size should be specified.
    batch_size: Optional[int] = None
    train_batch_size: Optional[int] = None
    valid_batch_size: Optional[int] = None
    num_workers: int = 0
    max_atom: int = 64
    spatial_dimension: int = 3  # the dimension of Euclidean space where atoms live.
    use_fixed_lattice_parameters: bool = False  # if True, do not noise the lattice parameters and use a fixed box
    elements: list[str]  # the elements that can exist.

    def __post_init__(self):
        """Post init."""
        assert self.data_source is not None, "The data source must be set."

        if self.batch_size is None:
            assert (
                self.valid_batch_size is not None
            ), "If batch_size is None, valid_batch_size must be specified."
            assert (
                self.train_batch_size is not None
            ), "If batch_size is None, train_batch_size must be specified."

        else:
            assert (
                self.valid_batch_size is None
            ), "If batch_size is specified, valid_batch_size must be None."
            assert (
                self.train_batch_size is None
            ), "If batch_size is specified, train_batch_size must be None."
