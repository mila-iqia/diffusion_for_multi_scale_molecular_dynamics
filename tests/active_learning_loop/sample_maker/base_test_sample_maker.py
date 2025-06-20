import einops
import numpy as np
import pytest
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from scipy.optimize import linear_sum_assignment

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
            case "none":
                return None
            case "fixed":
                return list(15.0 + np.random.rand(spatial_dimension))

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
    def samples_and_indices(self, structure_axl, uncertainty_per_atom, sample_maker):
        # This fixture is where we call the method we actually want to test.
        # We will conduct various checks based on the outputs of the method.
        list_sample_axl_structures, list_active_environment_indices, _ = (
            sample_maker.make_samples(structure_axl, uncertainty_per_atom))
        return list_sample_axl_structures, list_active_environment_indices

    def test_sample_lattice_parameters(self, samples_and_indices, sample_box_strategy,
                                       sample_box_size, lattice_parameters):
        list_samples_axl_structure, _ = samples_and_indices

        match sample_box_strategy:
            case "noop":
                reference_lattice_parameters = lattice_parameters
            case "fixed":
                unit_cell = np.diag(np.array(sample_box_size))
                reference_lattice_parameters = map_numpy_unit_cell_to_lattice_parameters(unit_cell)
            case _:
                raise NotImplementedError("Unknown sampling box making strategy.")

        for axl_structure in list_samples_axl_structure:
            sample_lattice_parameters = axl_structure.L
            np.testing.assert_allclose(sample_lattice_parameters, reference_lattice_parameters)

    @pytest.fixture()
    def calculated_pymatgen_sample_structures_and_indices(self, samples_and_indices, element_list):
        # We will use pymatgen to see if the created sample structures match expectations.
        # Convert the sampled axl structures to pymatgen for simpler testing.
        structure_converter = StructureConverter(element_list)
        list_sample_axl_structures, list_of_active_environment_index_arrays = samples_and_indices

        list_calculated_pymatgen_sample_structures = []
        for axl_substructure in list_sample_axl_structures:
            structure = structure_converter.convert_axl_to_structure(axl_substructure)
            list_calculated_pymatgen_sample_structures.append(structure)

        return list_calculated_pymatgen_sample_structures, list_of_active_environment_index_arrays

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

        # Let's use pymatgen to do the heavy lifting of matching substructures.
        structure_matcher = StructureMatcher(primitive_cell=False,
                                             scale=False,
                                             angle_tol=0.001,
                                             ltol=1e-8,
                                             stol=1e-8)

        # Each reference substructure should be found exactly "number_of_samples_per_substructure" times.
        # We'll test that this is indeed the case.
        found_count = np.zeros(len(list_reference_pymatgen_excised_substructures), dtype=int)

        for ref_idx, (subset_structure, excised_index) in (
                enumerate(zip(list_reference_pymatgen_excised_substructures, list_excised_atom_indices))):
            # The 'reference_structure' should contain an excised environment, namely a central atom identified by
            # the 'excised_index', and other nearby atoms that form the environment. This should be in the original
            # unit cell.
            reference_active_site = subset_structure.sites[excised_index]
            reference_lattice = subset_structure.lattice

            for sample_structure, active_index in zip(list_calculated_pymatgen_sample_structures, list_active_indices):
                # A sample structure *should* contain an excised environment in a unit cell that can either
                # be the original one or a fixed one specified by the user. There may also be more atoms
                # depending on the repainting algorithm used.

                # Check that the "active sites" have the same species; otherwise, a match is impossible.
                if sample_structure[active_index].species != reference_active_site.species:
                    continue

                # Since the sample maker may change the unit cell, we have to transform the lattice to match.
                superset_structure = Structure(species=sample_structure.species,
                                               lattice=reference_lattice,  # The original unit cell
                                               coords=sample_structure.cart_coords,
                                               coords_are_cartesian=True)

                if len(subset_structure) > len(superset_structure):
                    # Clearly, if the reference environment has more atoms than the sample, there cannot be a match.
                    continue

                # The excised environment may be translated in the superset structure; structure_matcher
                # can deal with that. The output variable "matching" is either "None" if there is no match,
                # or the indices of the matching atoms.
                matching = structure_matcher.get_mapping(superset=superset_structure,
                                                         subset=subset_structure)
                if matching is None:
                    continue

                # Next, attempt an exact match by accounting for the potential translation between the
                # active atom in the reference and the sample.
                sample_active_site = superset_structure.sites[active_index]

                # build a candidate for an exact structural match by making sure that the active atom in the
                # reference structure and the active atom in the sample structure coincide.
                translation = reference_active_site.coords - sample_active_site.coords

                # Now that the translation is known, the active atoms will be REMOVED from the
                # structures to be compared. This is to avoid various tedious edge cases where symmetry can
                # lead to false positive matches.
                reference = subset_structure.copy()
                reference.remove_sites([reference.index(reference_active_site)])

                sample_matching_sites = [superset_structure.sites[match_idx] for match_idx in matching]
                candidate = Structure(
                    species=[site.species for site in sample_matching_sites],
                    lattice=reference_lattice,
                    coords=[site.coords for site in sample_matching_sites],
                    to_unit_cell=True,
                    coords_are_cartesian=True)

                candidate.remove_sites([candidate.index(sample_active_site)])
                # Translation is in-place
                candidate.translate_sites(indices=np.arange(len(candidate)),
                                          vector=translation,
                                          frac_coords=False,
                                          to_unit_cell=True)

                # We next want to check if there is an exact match. We CANNOT use structure_matcher.get_rms_dist
                # because it can perform translations internally to align input structures. We cannot
                # assume that the atoms are in the same order between reference and candidate: we have to
                # solve an assignment problem. Note that this is why we had to remove the active atoms:
                # they must be assigned to each other, the linear assignment algorithm shouldn't have the freedom
                # to assign them differently.
                candidate_x = candidate.frac_coords
                ref_x = reference.frac_coords
                number_of_atoms = len(candidate_x)
                assert number_of_atoms != 0, \
                    ("The test assumes that an environment is not composed of a single atom. "
                     "This has now occurred in the test because of unfortunate random numbers."
                     ">>> Modify parameters to avoid this! Hint: increase the radial cutoff!")

                matrix1 = einops.repeat(candidate_x,
                                        "natoms1 spatial -> natoms1 natoms2 spatial",
                                        natoms2=number_of_atoms)
                matrix2 = einops.repeat(ref_x,
                                        "natoms2 spatial -> natoms1 natoms2 spatial",
                                        natoms1=number_of_atoms)
                distance_cost = np.linalg.norm(matrix1 - matrix2, axis=2)

                row_ind, col_ind = linear_sum_assignment(distance_cost)
                minimum_distance = distance_cost[row_ind, col_ind].sum()

                if minimum_distance < 1.0e-8:
                    # If we made it this far, then there is an exact structural match between the reference structure
                    # and the sample structure. The active sites are indeed the same, up to a translation.
                    found_count[ref_idx] += 1

        expected_count = number_of_samples_per_substructure * np.ones_like(found_count, dtype=int)
        np.testing.assert_array_equal(found_count, expected_count)
