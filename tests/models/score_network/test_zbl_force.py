from __future__ import annotations

from pathlib import Path

import pytest
import torch

from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.repulsive_force.zbl_force import (
    ZBLForce, ZBLForceParameters)


def _tests_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if p.name == "tests":
            return p
    raise RuntimeError("Could not locate tests/ directory from __file__.")


def read_lammps_forces(fn):
    """Read the forces contained in data/models/repulsion_score."""
    with open(fn, "r") as f:
        lines = f.readlines()
    forces = []
    start_read = False
    for line in lines:
        if line.strip().startswith("ITEM: ATOMS"):
            start_read = True
            continue
        if start_read:
            forces.append([float(x) for x in line.split()[-3:]])
    return torch.tensor(forces, dtype=torch.float32, device="cpu")


class TestZBLForce:
    @pytest.fixture()
    def cartesian_positions(self):
        pos = [[[2.0722, 6.8944, 4.1256],
               [3.5923, 1.7706, 3.1422],
               [5.6965, 5.8560, 1.6302],
               [1.1049, 5.1537, 6.9432],
               [5.3291, 0.0841, 0.0237],
               [6.9115, 6.2579, 6.8071],
               [0.9318, 3.6683, 3.3170],
               [4.9636, 2.1880, 6.4710],
               [5.9502, 0.8173, 0.6279],
               [5.7841, 2.8497, 3.4793],
               [4.6890, 1.1964, 1.4846],
               [2.4854, 1.0776, 4.6584],
               [2.7329, 4.4024, 6.6043],
               [2.2355, 6.9130, 2.1594],
               [0.7413, 5.7150, 2.4136],
               [0.9003, 2.8371, 3.7896],
               [5.9723, 1.1048, 6.9392],
               [2.1095, 1.8250, 0.0095],
               [1.2997, 1.5873, 2.0805],
               [3.6191, 4.1745, 2.1670],
               [5.9250, 1.3836, 3.8350],
               [1.6283, 2.3922, 5.5159],
               [2.0133, 0.2571, 6.3284],
               [0.0960, 2.0796, 3.2575],
               [3.5847, 3.7682, 3.8876],
               [6.1058, 5.0023, 5.3827],
               [1.8365, 2.1038, 3.7810],
               [0.3789, 2.4084, 4.3938],
               [6.4045, 6.2634, 0.3362],
               [3.1300, 4.4086, 0.7728],
               [2.8324, 1.7593, 6.0379],
               [3.5303, 4.5427, 4.7823]]]
        return torch.tensor(pos, dtype=torch.float32, device="cpu")

    @pytest.fixture()
    def basis_vectors(self):
        vec = [[[7.2200, 0.0000, 0.0000],
                [0.0000, 7.2200, 0.0000],
                [0.0000, 0.0000, 7.2200]]]
        return torch.tensor(vec, dtype=torch.float32, device="cpu")

    def test_masked_type_32(self, cartesian_positions, basis_vectors):
        """Test for ZBLRepulsionScore using masked_atom type of 14.5"""
        zbl_parameters = ZBLForceParameters(
            radial_cutoff=2.19293,
            inner_radius_fraction=0.5552844824048191,
            element_list=["Si", "P"],
        )
        zbl_force = ZBLForce(zbl_parameters)

        # elements_index is a random list containing both elements + masked type
        elements_index = [2, 2, 0, 2, 0, 2, 0, 0, 1, 0, 1, 0, 1, 2, 2, 1,
                          2, 0, 1, 0, 2, 1, 2, 0, 0, 1, 2, 1, 1, 1, 2, 0]
        A = torch.tensor([elements_index], dtype=torch.long)
        calculated_forces = zbl_force.get_cartesian_forces(A, cartesian_positions, basis_vectors)

        # Type is torch.Tensor
        assert isinstance(calculated_forces, torch.Tensor)
        # Shape = (1, 32, 3)
        assert calculated_forces.shape == (1, 32, 3)
        # Sum = 0
        assert torch.allclose(calculated_forces.sum(dim=[1, 2]), torch.zeros(calculated_forces.shape[0]),
                              atol=1e-4)

    def test_zbl_forces_match_lammps_sige(self, cartesian_positions, basis_vectors):
        """Test the forces of ZBLRepulsionScore with precomputed ones from LAMMPS."""
        zbl_parameters = ZBLForceParameters(
            radial_cutoff=2.19293,
            inner_radius_fraction=0.5552844824048191,
            element_list=["Si", "Ge"],
        )
        zbl_force = ZBLForce(zbl_parameters)

        elements_index = [[1, 0] * 16]
        A = torch.tensor(elements_index, dtype=torch.long)

        torch_forces = zbl_force.get_cartesian_forces(A, cartesian_positions, basis_vectors)[0]
        dump_path = _tests_root() / "reference_files" / "models" / "repulsion_score" / "SiGe.dump"
        lammps_forces = read_lammps_forces(dump_path)

        # Shape
        assert torch_forces.shape == lammps_forces.shape == (32, 3)
        # Sum = 0
        assert torch.allclose(torch_forces.sum(dim=[0, 1]), torch.tensor([0.]), atol=1e-3)
        # Same forces as lammps
        assert torch.allclose(torch_forces, lammps_forces, atol=1e-3)
