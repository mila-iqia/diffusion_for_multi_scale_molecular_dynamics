import pytest
import torch

from crystal_diffusion.data.diffusion.data_loader import \
    LammpsForDiffusionDataModule


@pytest.fixture
def input_data():
    return {
        'natom': [2],  # batch size of 1
        'box': [[1.0, 1.0, 1.0]],
        'position': [[[1., 2., 3], [4., 5, 6]]],  # for one batch, two atoms, 3D positions
        'type': [[1, 2]]
    }


def test_dataset_transform(input_data):
    result = LammpsForDiffusionDataModule.dataset_transform(input_data)

    # Check keys in result
    assert set(result.keys()) == {'natom', 'positions', 'box', 'type'}

    # Check tensor types and shapes
    assert torch.equal(result['natom'], torch.tensor([2]).long())
    assert result['positions'].shape == (1, 2, 3)  # (batchsize, natom, 3 [since it's 3D])
    assert result['box'].shape == (1, 3)
    assert torch.equal(result['type'], torch.tensor([[1, 2]]).long())

    # Check tensor types explicitly
    assert result['natom'].dtype == torch.long
    assert result['positions'].dtype == torch.float32  # default dtype for torch.as_tensor with float inputs
    assert result['box'].dtype == torch.float32
    assert result['type'].dtype == torch.long
