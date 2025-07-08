from pathlib import Path

from pymatgen.core import Structure

from diffusion_for_multi_scale_molecular_dynamics.analysis.ovito_utilities.xyz_utils import \
    generate_xyz_text


def write_active_learning_generated_sample(structure: Structure, output_path: Path):
    """Write active learning generated samples.

    This method assumes that the input structure has 'constrained' and 'forces' site properties.

    Args:
        structure (Structure): Structure object.
        output_path (Path): Path where to write the xyz output.

    Returns:
        None
    """
    properties_dim = dict(constrained=1, forces=3)
    site_properties = list(properties_dim.keys())

    for key in site_properties:
        assert key in structure.site_properties, f"The input structure is mising the site property {key}"

    xyz_text = generate_xyz_text(structure, site_properties, properties_dim)

    with open(output_path, 'w') as fd:
        fd.write(xyz_text)
