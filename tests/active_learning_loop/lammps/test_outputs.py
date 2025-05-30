from typing import Dict

import numpy as np
import pytest
import yaml

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.lammps.outputs import \
    extract_all_fields_from_dump
from tests.fake_data_utils import (create_dump_single_record,
                                   generate_fake_configuration)


def inject_uncertainty_field_in_yaml_document(
    doc: Dict, uncertainties: np.array
) -> Dict:
    doc["keywords"].append("c_uncertainty")
    old_data = doc["data"]
    new_data = []
    for row, u in zip(old_data, uncertainties):
        row.append(u)
        new_data.append(row)
    doc["data"] = new_data
    return doc


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


@pytest.fixture(params=[True, False])
def use_uncertainty(request):
    return request.param


@pytest.fixture()
def number_of_configurations():
    return 8


@pytest.fixture()
def configurations(spatial_dimension, unique_elements, number_of_configurations):

    list_configs = []

    list_number_of_atoms = np.random.randint(
        low=4, high=16, size=number_of_configurations
    )

    for number_of_atoms in list_number_of_atoms:
        config = generate_fake_configuration(
            spatial_dimension, number_of_atoms, unique_elements
        )
        list_configs.append(config)

    return list_configs


@pytest.fixture()
def list_expected_uncertainties(configurations):
    list_uncertainties = []
    for configuration in configurations:
        number_of_atoms = len(configuration.ids)
        uncertainties = np.random.rand(number_of_atoms)
        list_uncertainties.append(uncertainties)

    return list_uncertainties


@pytest.fixture()
def yaml_documents(configurations, use_uncertainty, list_expected_uncertainties):
    docs = []
    for timestep, (configuration, uncertainties) in enumerate(
        zip(configurations, list_expected_uncertainties)
    ):
        doc = create_dump_single_record(configuration, timestep)
        # Inject the thermo data in the document.
        doc["thermo"] = [
            dict(keywords=["Step", "PotEng"]),
            dict(data=[timestep, configuration.potential_energy]),
        ]

        if use_uncertainty:
            doc = inject_uncertainty_field_in_yaml_document(doc, uncertainties)

        docs.append(doc)

    return docs


@pytest.fixture()
def lammps_dump_path(yaml_documents, tmp_path):
    output_path = tmp_path / "dump.yaml"
    with open(output_path, "w") as yaml_file:
        yaml.dump_all(yaml_documents, yaml_file)
    return output_path


def test_extract_structures_forces_and_energies_from_dump(
    lammps_dump_path, configurations, use_uncertainty, list_expected_uncertainties
):

    list_structures, list_forces, list_energies, list_uncertainties = (
        extract_all_fields_from_dump(lammps_dump_path)
    )

    for structure, forces, energy, configuration in zip(
        list_structures, list_forces, list_energies, configurations
    ):
        np.testing.assert_almost_equal(energy, configuration.potential_energy)
        np.testing.assert_almost_equal(forces, configuration.cartesian_forces)
        np.testing.assert_almost_equal(
            structure.cart_coords, configuration.cartesian_positions
        )

        assert np.all(
            [str(site.specie) for site in structure.sites] == configuration.elements
        )
        np.testing.assert_almost_equal(
            structure.lattice.matrix, np.diag(configuration.cell_dimensions)
        )

    for computed_uncertainties, expected_uncertainties in zip(
        list_uncertainties, list_expected_uncertainties
    ):
        if use_uncertainty:
            np.testing.assert_almost_equal(
                computed_uncertainties, expected_uncertainties
            )
        else:
            assert computed_uncertainties is None
