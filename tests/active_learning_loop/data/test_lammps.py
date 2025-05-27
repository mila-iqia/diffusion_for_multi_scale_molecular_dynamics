import numpy as np
import pytest
import yaml

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.data.lammps import \
    extract_structures_forces_and_energies_from_dump
from tests.fake_data_utils import (create_dump_single_record,
                                   generate_fake_configuration)


@pytest.fixture(scope="module", autouse=True)
def set_seed():
    """Set the random seed."""
    np.random.seed(435345234)


@pytest.fixture()
def spatial_dimension():
    return 3


@pytest.fixture()
def unique_elements():
    # Since pymatgen is used to parse the elements, they cannot be random strings.
    return ["H", "Si", "Ca", "O", "Ge"]


@pytest.fixture()
def number_of_configurations():
    return 8


@pytest.fixture()
def configurations(spatial_dimension, unique_elements, number_of_configurations):

    list_configs = []

    list_number_of_atoms = np.random.randint(low=4, high=16, size=number_of_configurations)

    for number_of_atoms in list_number_of_atoms:
        config = generate_fake_configuration(spatial_dimension, number_of_atoms, unique_elements)
        list_configs.append(config)

    return list_configs


@pytest.fixture()
def yaml_documents(configurations):
    docs = []
    for timestep, configuration in enumerate(configurations):
        doc = create_dump_single_record(configuration, timestep)
        # Inject the thermo data in the document.
        doc['thermo'] = [dict(keywords=["Step", "PotEng"]), dict(data=[timestep, configuration.potential_energy])]
        docs.append(doc)

    return docs


@pytest.fixture()
def lammps_dump_path(yaml_documents, tmp_path):
    output_path = tmp_path / "dump.yaml"
    with open(output_path, "w") as yaml_file:
        yaml.dump_all(yaml_documents, yaml_file)
    return output_path


def test_extract_structures_forces_and_energies_from_dump(lammps_dump_path, configurations):

    list_structures, list_forces, list_energies = extract_structures_forces_and_energies_from_dump(lammps_dump_path)

    for structure, forces, energy, configuration in zip(list_structures, list_forces, list_energies, configurations):
        np.testing.assert_almost_equal(energy, configuration.potential_energy)
        np.testing.assert_almost_equal(forces, configuration.cartesian_forces)
        np.testing.assert_almost_equal(structure.cart_coords, configuration.cartesian_positions)

        assert np.all([str(site.specie) for site in structure.sites] == configuration.elements)
        np.testing.assert_almost_equal(structure.lattice.matrix, np.diag(configuration.cell_dimensions))
