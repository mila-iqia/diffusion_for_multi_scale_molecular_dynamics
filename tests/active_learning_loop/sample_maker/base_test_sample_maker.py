import numpy as np
import pytest
from pymatgen.core import Structure

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.atom_selector.threshold_atom_selector import (
    ThresholdAtomSelector, ThresholdAtomSelectorParameters)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.atom_selector.top_k_atom_selector import (
    TopKAtomSelector, TopKAtomSelectorParameters)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.excisor.excisor_factory import (
    create_excisor, create_excisor_parameters)
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.structure_converter import \
    StructureConverter
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import \
    map_numpy_unit_cell_to_lattice_parameters
from tests.active_learning_loop.conftest import BaseTestAxlStructure


class BaseTestSampleMaker(BaseTestAxlStructure):

    @pytest.fixture()
    def sample_maker(self, **kwargs):
        raise NotImplementedError("Must be implemented in subclass")

    @pytest.fixture()
    def top_k_value(self):
        return 4

    @pytest.fixture()
    def uncertainty_threshold(self):
        return 0.001

    @pytest.fixture
    def top_k_atom_selector(self, top_k_value):
        parameters = TopKAtomSelectorParameters(top_k_environment=top_k_value)
        return TopKAtomSelector(parameters)

    @pytest.fixture
    def threshold_atom_selector(self, uncertainty_threshold):
        parameters = ThresholdAtomSelectorParameters(uncertainty_threshold=uncertainty_threshold)
        return ThresholdAtomSelector(parameters)

    @pytest.fixture(params=["threshold", "top_k"])
    def atom_selector_algorithm(self, request):
        return request.param

    @pytest.fixture()
    def atom_selector(self, atom_selector_algorithm, threshold_atom_selector, top_k_atom_selector):
        match atom_selector_algorithm:
            case "threshold":
                return threshold_atom_selector
            case "top_k":
                return top_k_atom_selector
            case _:
                raise NotImplementedError("Non-existent atom_selector algorithm")

    @pytest.fixture(params=['uncertainty_threshold', 'excise_top_k_environment'])
    def excision_strategy(self, request):
        return request.param

    @pytest.fixture
    def uncertainty_per_atom(self, number_of_atoms, uncertainty_threshold):
        return 2 * np.random.rand(number_of_atoms) * uncertainty_threshold


class BaseTestExciseSampleMaker(BaseTestSampleMaker):
    @pytest.fixture(params=["nearest_neighbors", "spherical_cutoff"])
    def excisor_algorithm(self, request):
        return request.param

    @pytest.fixture()
    def spatial_dimension(self):
        # Force the spatial dimension to be 3 so that we can use pymatgen tools to conduct the tests.
        return 3

    @pytest.fixture(params=[1, 3])
    def number_of_samples_per_substructure(self, request):
        return request.param

    @pytest.fixture()
    def number_of_neighbors(self):
        return 4

    @pytest.fixture()
    def radial_cutoff(self):
        return 7.32

    @pytest.fixture(params=["fixed", "noop"])
    def sample_box_strategy(self, request):
        return request.param

    @pytest.fixture()
    def sample_box_size(self, sample_box_strategy, spatial_dimension):
        match sample_box_strategy:
            case "noop":
                return None
            case "fixed":
                return list(18.0 + 0.1 * np.random.rand(spatial_dimension))
            case _:
                raise NotImplementedError("Non-existent sample_box_strategy")

    @pytest.fixture()
    def excisor_parameters(self, excisor_algorithm, number_of_neighbors, radial_cutoff):
        excisor_dictionary = dict(algorithm=excisor_algorithm)

        match excisor_algorithm:
            case "nearest_neighbors":
                excisor_dictionary["number_of_neighbors"] = number_of_neighbors
            case "spherical_cutoff":
                excisor_dictionary["radial_cutoff"] = radial_cutoff
            case _:
                raise NotImplementedError("Unknown excisor algorithm.")

        return create_excisor_parameters(excisor_dictionary)

    @pytest.fixture()
    def excisor(self, excisor_parameters):
        return create_excisor(excisor_parameters)

    @pytest.fixture()
    def reference_pymatgen_excised_substructures_and_indices(self, element_list, uncertainty_per_atom,
                                                             atom_selector, excisor, structure_axl):
        # We will use pymatgen to see if the created sample structures match expectations.
        structure_converter = StructureConverter(element_list)
        selected_atom_indices = atom_selector.select_central_atoms(uncertainty_per_atom)

        list_reference_pymatgen_excised_substructures = []
        list_excised_atom_indices = []
        for atom_index in selected_atom_indices:
            excised_substructure_axl, excised_atom_index = excisor._excise_one_environment(structure_axl, atom_index)
            struct = structure_converter.convert_axl_to_structure(excised_substructure_axl)
            list_reference_pymatgen_excised_substructures.append(struct)
            list_excised_atom_indices.append(excised_atom_index)

        return list_reference_pymatgen_excised_substructures, list_excised_atom_indices

    @pytest.fixture()
    def make_samples_output(self, structure_axl, uncertainty_per_atom, sample_maker):
        # This fixture is where we call the method we actually want to test.
        # We will conduct various checks based on the outputs of the method.
        list_sample_axl_structures, list_active_environment_indices, list_sample_infos = (
            sample_maker.make_samples(structure_axl, uncertainty_per_atom))
        return list_sample_axl_structures, list_active_environment_indices, list_sample_infos

    @pytest.fixture()
    def list_sample_axl_structures(self, make_samples_output):
        return make_samples_output[0]

    @pytest.fixture()
    def list_active_environment_indices(self, make_samples_output):
        return make_samples_output[1]

    @pytest.fixture()
    def list_sample_infos(self, make_samples_output):
        return make_samples_output[2]

    def test_sample_lattice_parameters(self, list_sample_axl_structures, sample_box_strategy,
                                       sample_box_size, lattice_parameters):
        match sample_box_strategy:
            case "noop":
                reference_lattice_parameters = lattice_parameters
            case "fixed":
                unit_cell = np.diag(np.array(sample_box_size))
                reference_lattice_parameters = map_numpy_unit_cell_to_lattice_parameters(unit_cell)
            case _:
                raise NotImplementedError("Unknown sampling box making strategy.")

        for axl_structure in list_sample_axl_structures:
            sample_lattice_parameters = axl_structure.L
            np.testing.assert_allclose(sample_lattice_parameters, reference_lattice_parameters)

    @pytest.fixture()
    def calculated_pymatgen_sample_structures_and_indices(self, list_sample_axl_structures,
                                                          list_active_environment_indices, element_list):
        # We will use pymatgen to see if the created sample structures match expectations.
        # Convert the sampled axl structures to pymatgen for simpler testing.
        structure_converter = StructureConverter(element_list)

        list_calculated_pymatgen_sample_structures = []
        for axl_substructure in list_sample_axl_structures:
            structure = structure_converter.convert_axl_to_structure(axl_substructure)
            list_calculated_pymatgen_sample_structures.append(structure)

        return list_calculated_pymatgen_sample_structures, list_active_environment_indices

    def test_excised_environments_are_present(
        self,
        calculated_pymatgen_sample_structures_and_indices,
        reference_pymatgen_excised_substructures_and_indices,
        number_of_samples_per_substructure
    ):
        # The goal of this test is to check that the sample maker is indeed putting the expected
        # excised environments inside the new sample boxes. There may be other atoms present in the samples,
        # but the excised environment should be present, and its central atom should have the right index.
        # We'll leverage pymatgen to perform the relevant checks.

        # we'll use this to check if two positions are the same. We don't want this tolerance to be too small;
        # we can expect errors on the order of float machine epsilon if there is casting between float32 and float64.
        same_position_tolerance = 1.0e-5

        list_reference_pymatgen_excised_substructures, list_excised_atom_indices = (
            reference_pymatgen_excised_substructures_and_indices)

        list_calculated_pymatgen_sample_structures, list_of_active_environment_index_arrays = (
            calculated_pymatgen_sample_structures_and_indices)

        # For each sample structure, the active indices are actually stored as a numpy array. Since excision
        # presupposes a single active atom at the heart of the environment, these index arrays should
        # all contain a single element.
        # Let's test that this is indeed the case and extract the relevant index.
        list_active_indices = []
        for index_array in list_of_active_environment_index_arrays:
            assert len(index_array) == 1
            active_index = index_array[0]
            list_active_indices.append(active_index)

        # Each reference substructure should be found exactly "number_of_samples_per_substructure" times.
        # We'll test that this is indeed the case.
        found_count = np.zeros(len(list_reference_pymatgen_excised_substructures), dtype=int)

        for ref_idx, (reference_structure, excised_index) in (
                enumerate(zip(list_reference_pymatgen_excised_substructures, list_excised_atom_indices))):
            # The 'reference_structure' should contain an excised environment, namely a central atom identified by
            # the 'excised_index', and other nearby atoms that form the environment. This should be in the original
            # unit cell.
            reference_active_site = reference_structure.sites[excised_index]
            reference_lattice = reference_structure.lattice

            for sample_structure, active_index in zip(list_calculated_pymatgen_sample_structures, list_active_indices):
                # A sample structure *should* contain an excised environment in a unit cell that can either
                # be the original one or a fixed one specified by the user. There may also be more atoms
                # depending on the repainting algorithm used.

                # Check that the "active sites" have the same species; otherwise, a match is impossible.
                if sample_structure[active_index].species != reference_active_site.species:
                    continue

                # Since the sample maker may change the unit cell, we have to transform the lattice to match.
                candidate_structure = Structure(species=sample_structure.species,
                                                lattice=reference_lattice,  # The original unit cell
                                                coords=sample_structure.cart_coords,
                                                coords_are_cartesian=True)

                if len(reference_structure) > len(candidate_structure):
                    # Clearly, if the reference structure has more atoms than the sample, there cannot be a match.
                    continue

                # The excised environment may be translated in the sample structure.
                # The candidate_structure will next be translated in order to ensure that the active atom in the
                # reference structure and the active atom in the candidate structure coincide.
                sample_active_site = candidate_structure.sites[active_index]
                translation = reference_active_site.coords - sample_active_site.coords

                # The translation is in-place
                candidate_structure.translate_sites(indices=np.arange(len(candidate_structure)),
                                                    vector=translation,
                                                    frac_coords=False,
                                                    to_unit_cell=True)

                # can we find every reference site in the candidate structure?
                list_found_reference_sites = []
                for ref_site in reference_structure.sites:
                    site_is_found = False
                    for candidate_site in candidate_structure.sites:
                        species_are_the_same = candidate_site.species == ref_site.species
                        positions_are_the_same = ref_site.distance(candidate_site) < same_position_tolerance
                        if species_are_the_same and positions_are_the_same:
                            site_is_found = True
                            break
                    list_found_reference_sites.append(site_is_found)

                if np.all(list_found_reference_sites):
                    # If we made it this far, then there is an exact structural match between the reference structure
                    # and the sample structure. The active sites are indeed the same, up to a translation.
                    found_count[ref_idx] += 1

        expected_count = number_of_samples_per_substructure * np.ones_like(found_count, dtype=int)
        np.testing.assert_array_equal(found_count, expected_count)
