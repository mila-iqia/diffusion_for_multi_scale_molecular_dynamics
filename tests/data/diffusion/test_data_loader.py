import pytest
import torch

from crystal_diffusion.data.diffusion.data_loader import \
    LammpsForDiffusionDataModule


class TestDiffusionDataLoader:
    @pytest.fixture
    def input_data_to_transform(self):
        return {
            'natom': [2],  # batch size of 1
            'box': [[1.0, 1.0, 1.0]],
            'position': [[1., 2., 3, 4., 5, 6]],  # for one batch, two atoms, 3D positions
            'type': [[1, 2]]
        }
    def test_dataset_transform(self, input_data_to_transform):
        result = LammpsForDiffusionDataModule.dataset_transform(input_data_to_transform)
        # Check keys in result
        assert set(result.keys()) == {'natom', 'position', 'box', 'type'}

        # Check tensor types and shapes
        assert torch.equal(result['natom'], torch.tensor([2]).long())
        assert result['position'].shape == (1, 2, 3)  # (batchsize, natom, 3 [since it's 3D])
        assert result['box'].shape == (1, 3)
        assert torch.equal(result['type'], torch.tensor([[1, 2]]).long())

        # Check tensor types explicitly
        assert result['natom'].dtype == torch.long
        assert result['position'].dtype == torch.float32  # default dtype for torch.as_tensor with float inputs
        assert result['box'].dtype == torch.float32
        assert result['type'].dtype == torch.long

    @pytest.fixture
    def input_data_to_pad(self):
        return {
            'natom': 2,  # batch size of 1
            'box': [1.0, 1.0, 1.0],
            'position': [1., 2., 3, 4., 5, 6],  # for one batch, two atoms, 3D positions
            'type': [1, 2]
        }

    def test_pad_dataset(self, input_data_to_pad):
        max_atom = 5  # Assume we want to pad to a max of 5 atoms
        padded_sample = LammpsForDiffusionDataModule.pad_samples(input_data_to_pad, max_atom)

        # Check if the type and position have been padded correctly
        assert len(padded_sample['type']) == max_atom
        assert padded_sample['position'].shape == torch.Size([max_atom * 3])

        # Check that the padding uses -1 for type
        # 2 atoms in the input_data - last 3 atoms should be type -1
        for k in range(max_atom - 2):
            assert padded_sample['type'].tolist()[-(k + 1)] == -1

        # Check that the padding uses nan for position
        assert torch.isnan(padded_sample['position'][-(max_atom -2) * 3:]).all()
