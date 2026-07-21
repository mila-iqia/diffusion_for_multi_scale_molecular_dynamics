from __future__ import annotations

from dataclasses import dataclass

import torch
from ase.data import atomic_numbers
from ase.units import J, _e, _eps0, m

from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.repulsive_force.repulsive_force import (
    RepulsiveForce, RepulsiveForceParameters)


@dataclass(kw_only=True)
class ZBLForceParameters(RepulsiveForceParameters):
    """Specific Hyper-parameters for ZBL forces.

    Args:
        radial_cutoff: distance where the ZBL interaction is fully switched off (Angstrom).
        element_list: ordered list of element symbols used to map atom type indices A -> atomic number.
        inner_radius_fraction: inner_radius = fraction * radial_cutoff, where the switching polynomial starts.
    """
    architecture: str = "zbl"
    element_list: list[str]
    inner_radius_fraction: float = 0.5  # inner_radius = inner_radius_fraction * radial_cutoff


class ZBLForce(RepulsiveForce):
    """Ziegler-Biersack-Littmark interatomic potential to get an analytical repulsion score based on LAMMPS ZBL."""

    def __init__(self, hyper_params: ZBLForceParameters):
        """Initialize the ZBL analytical repulsion model which calculates forces and gives an analytical score."""
        super().__init__(hyper_params)
        self.inner_radius = self.radial_cutoff * hyper_params.inner_radius_fraction

        # How does ZBL should deal with masked atom type ? For now, it will be the average Z over types.
        Z_of_index = [atomic_numbers[s] for s in hyper_params.element_list]
        Z_of_index.append(sum(Z_of_index) / len(Z_of_index))  # Adding masked element
        self.register_buffer("index_to_atomic_numbers", torch.tensor(Z_of_index, dtype=torch.float32))

        # eps0 in C^2/(J*m) and _e in C
        self.register_buffer(
            "prefactor",  # e^2/(4pi epsilon_0) in eV ang
            torch.tensor(_e**2 / (4 * torch.pi * _eps0) * J * m)
        )
        self.calc_abc()

    def get_cartesian_forces(self, A, cartesian_positions, basis_vectors):
        """Calculate forces using ZBL interatomic potential as used in LAMMPS.

        Uses torch.autograd for efficient calculations of the forces.

        Args :
            A: atom type indices [nconf, natoms], used to index index_to_atomic_numbers
            cartesian_positions: atomic positions in Angstrom [nconf, natoms, 3]
            basis_vectors: cell vectors in Angstrom [nconf, 3, 3]

        Returns:
            forces: per-atom forces [nconf, natoms, 3]
        """
        with torch.enable_grad():
            # We'll need the grad for forces
            cartesian_positions = cartesian_positions.clone().detach().requires_grad_(True)
            adj, distances = self.get_atomic_distances(cartesian_positions, basis_vectors)
            src, dst = adj.adjacency_matrix
            b = adj.edge_batch_indices

            # Keep only unique pairs (src < dst) to avoid double counting
            unique_mask = src < dst
            b_unique = b[unique_mask]
            src_unique = src[unique_mask]
            dst_unique = dst[unique_mask]
            r_ij = distances[unique_mask]

            idx_i = A[b_unique, src_unique]
            idx_j = A[b_unique, dst_unique]

            E_ij = self.zbl_energy(r_ij, idx_i, idx_j)
            E_total = E_ij.sum()
            grad_E, = torch.autograd.grad(
                E_total,
                cartesian_positions,
                create_graph=False,
                retain_graph=False
            )
            forces = -grad_E
        return forces

    def zbl_energy(self, r_ij: torch.Tensor, idx_i: torch.Tensor, idx_j: torch.Tensor):
        """Calculates the pairwise energy using ZBL interatomic potential as used in LAMMPS.

        E_ij = 1/(4 pi epsilon_0) (Z_i Z_j e^2)/r_ij phi(r_ij/a) + S(r_ij)
        a = 0.46850/ (Zi^0.23 + Zj^0.23)  -> in angstrom
        phi(x) = 0.18175e^(-3.19980x) + 0.50986e^(-0.94229x) + 0.28022e^(-0.40290x) + 0.02817e^(-0.20162x)
        S(r_ij) -> See the function zbl_S
        """
        Zi = self.index_to_atomic_numbers[idx_i]
        Zj = self.index_to_atomic_numbers[idx_j]
        a = 0.46850 / (Zi.pow(0.23) + Zj.pow(0.23))  # ang
        phi = self.zbl_phi(r_ij / a)

        S = self.zbl_S(r_ij, idx_i, idx_j)
        E_ij = self.prefactor * (Zi * Zj) / r_ij * phi + S  # in eV

        return E_ij

    def calc_abc(self):
        """Calculate A, B and C of the switching function and cache it into self.abc for every pair Zi, Zj.

        See zbl_S for more information about how A, B, C are used.

        E(r)  = (1/(4*pi*eps0)) * Zi*Zj*e^2/r * phi(r/a)
        E'(r) = (1/(4*pi*eps0)) * Zi*Zj*e^2 *
                ( phi'(r/a)/(a*r) - phi(r/a)/r^2 )
        E''(r)= (1/(4*pi*eps0)) * Zi*Zj*e^2 *
                ( phi''(r/a)/(a^2*r) - 2*phi'(r/a)/(a*r^2) + 2*phi(r/a)/r^3 )

        A = (-3*E'(rc) + (rc-ri)*E''(rc) )/(rc-ri)^2
        B = ( 2*E'(rc) - (rc-ri)*E''(rc) )/(rc-ri)^3
        C = -E(rc) + 0.5*(rc-ri)*E'(rc) - (1/12)*(rc-ri)^2*E''(rc)
        """
        rc = self.radial_cutoff
        ri = self.inner_radius
        Zi = self.index_to_atomic_numbers[:, None]
        Zj = self.index_to_atomic_numbers[None, :]

        # Compute E, E', E'' evaluated at rc.
        a = 0.46850 / (Zi.pow(0.23) + Zj.pow(0.23))
        Erc = self.prefactor * (Zi * Zj) / rc * self.zbl_phi(rc / a)
        Eprc = self.prefactor * (Zi * Zj) * (self.zbl_phiprime(rc / a) / (a * rc) - self.zbl_phi(rc / a) / rc**2)
        Epprc = self.prefactor * (Zi * Zj) * (
            self.zbl_phiprimeprime(rc / a) / (a * a * rc)
            - 2 * self.zbl_phiprime(rc / a) / (a * rc * rc)
            + 2 * self.zbl_phi(rc / a) / (rc**3))
        A = (-3 * Eprc + (rc - ri) * Epprc) / (rc - ri)**2
        B = (2 * Eprc - (rc - ri) * Epprc) / (rc - ri)**3
        C = -Erc + 1 / 2 * (rc - ri) * Eprc - 1 / 12 * (rc - ri)**2 * Epprc
        self.register_buffer("abc", torch.stack((A, B, C), dim=-1))

    def zbl_S(self, r, idx_i, idx_j):
        """Calculate the Switching function S(r) based on Lammps ZBL.

        The switching function is comprised of an inner radius ri where the effect of S(r) appears
        and an outer radius rc where the whole ZBL becomes zero.
        The ZBL contains three different regions :
            1. S(r) = C, for r < ri
            2. S(r) = A/3 (r-ri)**3 + B/4 (r-ri)**4 + C, for ri < r < rc
            3. S(r) = 0, for r > rc
        These regions satisfies two conditions :
            1. S(ri)=C, S'(ri)=0, S''(ri)=0, by construction (Note that the constant C doesn't change the dynamic).
            2. S(rc)=-E(rc), S'(rc)=-E'(rc), S''(rc)=-E''(rc), which are respected through A, B and C.
        A, B, C, E(rc), E'(rc) and E''(rc) are all computed in calc_abc.

        Args :
             r_ij : Distance between all atoms within self.radial_cutoff of each other.
                    IMPORTANT : The function will fail if given distance bigger than self.radial_cutoff.
             idx_i : index of atom i in index_to_atomic_numbers
             idx_j : index of atom j in index_to_atomic_numbers

        Returns:
            Tensor of same shape as r containing the switching contribution.
        """
        A = self.abc[:, :, 0]
        B = self.abc[:, :, 1]
        C = self.abc[:, :, 2]
        ri = self.inner_radius

        # Inner region : r <= ri
        S_inner = C[idx_i, idx_j]

        # Middle region : ri < r < rc
        S_mid = A[idx_i, idx_j] / 3 * (r - ri)**3 + B[idx_i, idx_j] / 4 * (r - ri)**4 + C[idx_i, idx_j]
        S = torch.where(r > ri, S_mid, S_inner)
        return S

    def zbl_phi(self, x):
        """Calculate the term phi in ZBL.

        phi(x) = 0.18175e^(-3.19980x) + 0.50986e^(-0.94229x) + 0.28022e^(-0.40290x) + 0.02817e^(-0.20162x)

        Args : x = r_ij / a
        """
        return (
            0.18175 * torch.exp(-3.19980 * x)
            + 0.50986 * torch.exp(-0.94229 * x)
            + 0.28022 * torch.exp(-0.40290 * x)
            + 0.02817 * torch.exp(-0.20162 * x)
        )

    def zbl_phiprime(self, x):
        """Calculate the first derivative of phi in ZBL w.r.t. x.

        Args : x = r_ij / a
        """
        return (
            0.18175 * -3.19980 * torch.exp(-3.19980 * x)
            + 0.50986 * -0.94229 * torch.exp(-0.94229 * x)
            + 0.28022 * -0.40290 * torch.exp(-0.40290 * x)
            + 0.02817 * -0.20162 * torch.exp(-0.20162 * x)
        )

    def zbl_phiprimeprime(self, x):
        """Calculate the second derivative of phi in ZBL w.r.t. x.

        Args : x = r_ij / a
        """
        return (
            0.18175 * 3.19980**2 * torch.exp(-3.19980 * x)
            + 0.50986 * 0.94229**2 * torch.exp(-0.94229 * x)
            + 0.28022 * 0.40290**2 * torch.exp(-0.40290 * x)
            + 0.02817 * 0.20162**2 * torch.exp(-0.20162 * x)
        )
