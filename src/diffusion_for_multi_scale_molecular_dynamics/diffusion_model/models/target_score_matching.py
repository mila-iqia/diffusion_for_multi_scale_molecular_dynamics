"""Target score matching: a coordinate target that uses the forces of the clean configurations.

For a configuration x0 drawn from a Boltzmann distribution at temperature T, the score of the data is
grad log p(x0) = beta F(x0), with beta = 1 / (k_B T). The sigma-normalized coordinate target becomes

    target = kappa sigma beta F(x0) + (1 - kappa) target_dsm,

where target_dsm is the denoising score matching target (-z for a non-wrapped gaussian) and kappa is a mixer giving
the relative importance of the thermal and of the noise contributions. Approximating the thermal vibrations of an
atom as a 3d gaussian of width s, the thermal width of its element,

    kappa = s^2 / (s^2 + sigma^2).

See De Bortoli et al., "Target Score Matching" (arXiv:2402.08667).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import ase.io
import numpy as np
import torch
from ase.calculators.singlepoint import SinglePointCalculator
from ase.units import kB

from diffusion_for_multi_scale_molecular_dynamics.namespace import AXL
from diffusion_for_multi_scale_molecular_dynamics.utils.element_types import \
    ElementTypes


@dataclass(kw_only=True)
class TargetScoreMatchingParameters:
    """Parameters of the target score matching coordinate target, and of the optional bad frames.

    Attributes:
        temperature: temperature of the training data.
        thermal_width: thermal width s for each element.
        minimum_sigma: lowest noise level given to the network for a bad frame.
        maximum_sigma: highest noise level given to the network for a bad frame, which should still be a relatively
            low noise. The noise level of a bad frame is drawn log-uniformly between minimum_sigma and maximum_sigma,
            but the bad frame itself is not noised.
        bad_frames_path: trajectory of bad configurations, each with its forces stored in a SinglePointCalculator.
            Every bad frame is used once per training epoch.
        maximum_force: the norm of each atomic force of a bad frame is clipped to this value, to not train
            exclusively on the configurations with the highest gradient. For an atom whose force is clipped, a
            prediction at least as strong as the target, in the direction of the target, is not penalized.
            Required with bad_frames_path.
        bad_frames_lambda_weight: weight of each bad frame in the loss, relative to a training configuration.
    """

    temperature: float  # K
    thermal_width: Dict[str, float]  # Angstrom
    minimum_sigma: float  # Angstrom
    maximum_sigma: float  # Angstrom
    bad_frames_path: Optional[Union[str, Path]] = None
    maximum_force: Optional[float] = None  # eV/Angstrom
    bad_frames_lambda_weight: float = 1.0

    def __post_init__(self):
        """Post init checks."""
        assert self.temperature > 0.0, "The temperature must be positive."
        assert all(width > 0.0 for width in self.thermal_width.values()), "The thermal widths must be positive."
        assert 0.0 < self.minimum_sigma <= self.maximum_sigma, \
            "The bad frames noise range must satisfy 0 < minimum_sigma <= maximum_sigma."
        if self.bad_frames_path is not None:
            assert self.maximum_force is not None and self.maximum_force > 0.0, \
                "A positive maximum_force is required with bad_frames_path."

    @property
    def beta(self) -> float:
        """Inverse temperature 1 / (k_B T), in 1/eV."""
        return 1.0 / (kB * self.temperature)


def get_thermal_widths_by_atom_type(thermal_width: Dict[str, float], elements: List[str]) -> torch.Tensor:
    """Thermal width of each atom type id, following the (sorted) ids of ElementTypes.

    Args:
        thermal_width: thermal width of each element, in Angstrom.
        elements: the elements of the training data.

    Returns:
        thermal_widths: tensor of dimension [number_of_atom_types].
    """
    element_types = ElementTypes(elements)
    missing_elements = set(element_types.elements) - set(thermal_width.keys())
    assert not missing_elements, f"No thermal width given for the elements {sorted(missing_elements)}."
    return torch.tensor([thermal_width[element] for element in element_types.elements])


def get_target_score_matching_target(
    denoising_score_matching_target: torch.Tensor,
    forces: torch.Tensor,
    atom_types: torch.Tensor,
    sigma: torch.Tensor,
    thermal_widths_by_atom_type: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Mix the force target sigma beta F with the denoising score matching target.

    Args:
        denoising_score_matching_target: sigma-normalized DSM target. Dimension [batch_size, number_of_atoms, 3].
        forces: forces of the clean configurations, in eV/Angstrom. Dimension [batch_size, number_of_atoms, 3].
        atom_types: atom type ids of the clean configurations (non-negative). Dimension [batch_size, number_of_atoms].
        sigma: noise level of each configuration, in Angstrom. Dimension [batch_size].
        thermal_widths_by_atom_type: thermal width of each atom type id, in Angstrom.
        beta: inverse temperature, in 1/eV.

    Returns:
        target: kappa sigma beta F + (1 - kappa) target_dsm. Dimension [batch_size, number_of_atoms, 3].
    """
    thermal_widths = thermal_widths_by_atom_type[atom_types]
    sigma_per_atom = sigma[:, None]
    kappa = (thermal_widths**2 / (thermal_widths**2 + sigma_per_atom**2)).unsqueeze(-1)
    force_target = sigma_per_atom.unsqueeze(-1) * beta * forces
    return kappa * force_target + (1.0 - kappa) * denoising_score_matching_target


def clip_forces(forces: torch.Tensor, maximum_force: float) -> torch.Tensor:
    """Clip the norm of each atomic force to maximum_force, keeping its direction.

    Args:
        forces: forces of dimension [..., 3].
        maximum_force: maximum norm of an atomic force.

    Returns:
        clipped_forces: forces with norms at most maximum_force.
    """
    norms = torch.linalg.norm(forces, dim=-1, keepdim=True)
    return forces * torch.clamp(maximum_force / norms.clamp(min=1e-12), max=1.0)


def get_one_sided_squared_error(
    predictions: torch.Tensor, target: torch.Tensor, saturated_atoms: torch.Tensor
) -> torch.Tensor:
    """Squared error of each atom, one-sided for saturated atoms.

    Calculate the error for the predictions of the network. Saturated atoms are those whose force was bigger than
    maximum_force, before clipping.
    The prediction norm is not penalized if it is at least as strong as the clipped target.
    The prediction direction is penalized as the difference between prediction and the component of prediction
    parallel to the target (a prediction stronger than the target is penalized as if it had the norm of the target).
    Every unsaturated atom has the plain squared error |predictions - target|^2.

    Args:
        predictions: predicted sigma-normalized scores. Dimension [batch_size, number_of_atoms, 3].
        target: target sigma-normalized scores. Dimension [batch_size, number_of_atoms, 3].
        saturated_atoms: atoms whose force is clipped. Dimension [batch_size, number_of_atoms].

    Returns:
        squared_error: squared error of each atom. Dimension [batch_size, number_of_atoms].
    """
    target_norms = torch.linalg.norm(target, dim=-1, keepdim=True)
    target_directions = target / target_norms.clamp(min=1e-12)
    parallel = (predictions * target_directions).sum(dim=-1, keepdim=True)
    perpendicular = predictions - parallel * target_directions
    parallel_error = (target_norms - parallel).clamp(min=0.0)
    prediction_norms = torch.linalg.norm(predictions, dim=-1, keepdim=True)
    perpendicular_scale = (target_norms / prediction_norms.clamp(min=1e-12)).clamp(max=1.0)
    one_sided_error = parallel_error**2 + ((perpendicular_scale * perpendicular) ** 2).sum(dim=-1, keepdim=True)
    plain_error = ((predictions - target) ** 2).sum(dim=-1)
    return torch.where(saturated_atoms, one_sided_error.squeeze(-1), plain_error)


def read_bad_frames(
    bad_frames_path: Union[str, Path], elements: List[str], maximum_force: float
) -> Optional[Dict[str, torch.Tensor]]:
    """Read the bad frames, with their forces clipped to maximum_force.

    Every frame must carry its forces in a SinglePointCalculator and have the same number of atoms, in an
    orthogonal cell.

    Args:
        bad_frames_path: trajectory of bad configurations.
        elements: the elements of the training data.
        maximum_force: the norm of each atomic force is clipped to this value, in eV/Angstrom.

    Returns:
        bad_frames: the composition (AXL of atom types, relative coordinates and lattice parameters), the clipped
            forces and the saturated atoms (whose force is clipped) of every frame, or None if the file is missing
            or empty.
    """
    bad_frames_path = Path(bad_frames_path)
    if not bad_frames_path.is_file() or bad_frames_path.stat().st_size == 0:
        return None
    frames = ase.io.read(bad_frames_path, index=":")
    if len(frames) == 0:
        return None

    number_of_atoms = {len(atoms) for atoms in frames}
    assert len(number_of_atoms) == 1, f"All bad frames must have the same number of atoms. Got {number_of_atoms}."
    for atoms in frames:
        assert isinstance(atoms.calc, SinglePointCalculator), \
            "Every bad frame must carry its forces in a SinglePointCalculator."
        assert np.allclose(atoms.cell.angles(), 90.0), "The bad frames must have orthogonal cells."

    element_types = ElementTypes(elements)
    atom_types = [[element_types.get_element_id(symbol) for symbol in atoms.get_chemical_symbols()]
                  for atoms in frames]
    lattice_parameters = [list(atoms.cell.lengths()) + [0.0, 0.0, 0.0] for atoms in frames]

    relative_coordinates = np.ascontiguousarray(np.stack([atoms.get_scaled_positions(wrap=True) for atoms in frames]))
    composition = AXL(
        A=torch.tensor(atom_types, dtype=torch.long),
        X=torch.tensor(relative_coordinates, dtype=torch.float32),
        L=torch.tensor(lattice_parameters, dtype=torch.float32),
    )
    forces = torch.tensor(np.ascontiguousarray(np.stack([atoms.get_forces() for atoms in frames])), dtype=torch.float32)
    saturated_atoms = torch.linalg.norm(forces, dim=-1) > maximum_force
    return dict(composition=composition, forces=clip_forces(forces, maximum_force), saturated_atoms=saturated_atoms)
