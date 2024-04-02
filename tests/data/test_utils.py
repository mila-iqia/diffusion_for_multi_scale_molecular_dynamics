import pytest
import yaml

from crystal_diffusion.data.utils import crop_lammps_yaml

# Sample data for dump and thermo YAML files
DUMP_YAML_CONTENT = """
- step: 0
  data: {...}
---
- step: 1
  data: {...}
---
- step: 2
  data: {...}
"""
THERMO_YAML_CONTENT = """
data:
  - {...}
  - {...}
  - {...}
"""


@pytest.fixture
def dump_file(tmpdir):
    file = tmpdir.join("lammps_dump.yaml")
    file.write(DUMP_YAML_CONTENT)
    return str(file)


@pytest.fixture
def thermo_file(tmpdir):
    file = tmpdir.join("lammps_thermo.yaml")
    file.write(THERMO_YAML_CONTENT)
    return str(file)


def test_crop_lammps_yaml(dump_file, thermo_file):
    crop_step = 1
    # Call the function with the path to the temporary files
    cropped_dump, cropped_thermo = crop_lammps_yaml(dump_file, thermo_file, crop_step)

    # Load the content to assert correctness
    with open(dump_file) as f:
        dump_yaml_content = list(yaml.safe_load_all(f))

    with open(thermo_file) as f:
        thermo_yaml_content = yaml.safe_load(f)

    # Verify the function output
    assert len(cropped_dump) == len(dump_yaml_content) - crop_step
    assert len(cropped_thermo['data']) == len(thermo_yaml_content['data']) - crop_step

    # Testing exception for too large crop_step
    with pytest.raises(ValueError):
        crop_lammps_yaml(dump_file, thermo_file, 4)
