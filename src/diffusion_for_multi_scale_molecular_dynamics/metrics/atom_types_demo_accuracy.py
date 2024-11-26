import torch


def evaluate_batch_accuracy(atom_types_predictions: torch.LongTensor, num_atom_types: int) -> torch.BoolTensor:
    # atom_typse_predictions: batch_size, num_atoms
    batchsize, num_atoms = atom_types_predictions.shape
    good_samples = torch.arange(0, num_atoms).unsqueeze(0).repeat(batchsize, 1).to(atom_types_predictions.device)
    good_samples = (good_samples + atom_types_predictions[:, 0].unsqueeze(-1)) % num_atom_types
    return torch.all(atom_types_predictions == good_samples, dim=1)
