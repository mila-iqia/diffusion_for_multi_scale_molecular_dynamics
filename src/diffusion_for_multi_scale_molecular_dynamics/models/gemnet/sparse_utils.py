from typing import Optional, Tuple

import torch


try:
    import pyg_lib  # noqa
    WITH_PYG_LIB = True
    WITH_INDEX_SORT = hasattr(pyg_lib.ops, 'index_sort')
except ImportError:
    pyg_lib = object
    WITH_PYG_LIB = False
    WITH_INDEX_SORT = False

try:
    from typing_extensions import Final  # noqa
except ImportError:
    from torch.jit import Final  # noqa


def index_sort(
        inputs: torch.Tensor,
        max_value: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""See pyg-lib documentation for more details:
    https://pyg-lib.readthedocs.io/en/latest/modules/ops.html"""
    if not WITH_INDEX_SORT:  # pragma: no cover
        return inputs.sort()
    return pyg_lib.ops.index_sort(inputs, max_value)