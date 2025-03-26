from pathlib import Path

from diffusion_for_multi_scale_molecular_dynamics.utils.reference_configurations import \
    create_equilibrium_sige_structure

if __name__ == "__main__":
    output_file_path = Path(__file__).parent / "equilibrium_sige.cif"
    structure = create_equilibrium_sige_structure()
    structure.to(output_file_path)
