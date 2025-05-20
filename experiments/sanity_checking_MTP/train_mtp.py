from pathlib import Path

from maml.apps.pes import MTPotential

from src.diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.data.lammps import extract_structures_forces_and_energies_from_dump

lammps_dump_path = Path("/home/user/diffusion_for_multi_scale_molecular_dynamics/experiments/sanity_checking_MTP/artn_trajectory/dump.yaml")

if __name__ == '__main__':

    list_structures, list_forces, list_energies  = extract_structures_forces_and_energies_from_dump(lammps_dump_path)


    # The MTP Potential object from MAML implemments the parsing of the MTP config files.

    structure = list_structures[0]
    forces = list_forces[0]
    energy = list_energies[0]

    dummy_virial_stress = 6 * [0]

    mtp_potential = MTPotential()
    lines = mtp_potential._line_up(structure, energy, forces, virial_stress=dummy_virial_stress)