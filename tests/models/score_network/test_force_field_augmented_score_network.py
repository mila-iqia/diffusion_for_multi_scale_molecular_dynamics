from unittest.mock import patch

import pytest
import torch
from ase import Atoms

from diffusion_for_multi_scale_molecular_dynamics.generators.langevin_generator import \
    LangevinGenerator
from diffusion_for_multi_scale_molecular_dynamics.generators.predictor_corrector_axl_generator import \
    PredictorCorrectorSamplingParameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.egnn_score_network import (
    EGNNScoreNetwork, EGNNScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.force_field_augmented_score_network import (
    ForceFieldAugmentedScoreNetwork, ForceFieldAugmentedScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.mlp_score_network import (
    MLPScoreNetwork, MLPScoreNetworkParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.repulsive_force.harmonic_force import \
    HarmonicForceParameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.repulsive_force.zbl_force import \
    ZBLForceParameters
from diffusion_for_multi_scale_molecular_dynamics.namespace import (
    AXL, CARTESIAN_FORCES, NOISE, NOISY_AXL_COMPOSITION, NUMBER_OF_ATOMS, TIME,
    UNIT_CELL)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_scheduler import \
    NoiseScheduler
from diffusion_for_multi_scale_molecular_dynamics.utils.basis_transformations import (
    get_positions_from_coordinates,
    map_lattice_parameters_to_unit_cell_vectors)
from diffusion_for_multi_scale_molecular_dynamics.utils.neighbors import \
    get_periodic_adjacency_information
from tests.models.score_network.base_test_score_network import \
    BaseTestScoreNetwork


@pytest.mark.parametrize("number_of_atoms", [4, 8, 16])
@pytest.mark.parametrize("radial_cutoff", [1.5, 2.0, 2.5])
class TestForceFieldAugmentedScoreNetworkHarmonic(BaseTestScoreNetwork):
    @pytest.fixture()
    def score_network(
        self, number_of_atoms, spatial_dimension, num_atom_types
    ):
        # Generate an arbitrary MLP-based score network.
        score_network_parameters = MLPScoreNetworkParameters(
            spatial_dimension=spatial_dimension,
            number_of_atoms=number_of_atoms,
            num_atom_types=num_atom_types,
            relative_coordinates_embedding_dimensions_size=6,
            noise_embedding_dimensions_size=6,
            time_embedding_dimensions_size=6,
            atom_type_embedding_dimensions_size=12,
            lattice_parameters_embedding_dimensions_size=6,
            n_hidden_dimensions=2,
            hidden_dimensions_size=16,
        )
        return MLPScoreNetwork(score_network_parameters)

    @pytest.fixture()
    def harmonic_force_parameters(self, radial_cutoff):
        return HarmonicForceParameters(radial_cutoff=radial_cutoff, strength=1.0)

    @pytest.fixture()
    def force_field_augmented_score_network(
        self, score_network, harmonic_force_parameters
    ):
        force_field_parameters = ForceFieldAugmentedScoreNetworkParameters(
            repulsive_force_parameters=harmonic_force_parameters,
        )
        augmented_score_network = ForceFieldAugmentedScoreNetwork(
            score_network=score_network,
            force_field_parameters=force_field_parameters,
        )
        return augmented_score_network

    @pytest.fixture
    def times(self, batch_size):
        times = torch.rand(batch_size, 1)
        return times

    @pytest.fixture()
    def basis_vectors(self, batch_size, spatial_dimension):
        # orthogonal boxes with dimensions between 5 and 10.
        orthogonal_boxes = torch.stack(
            [
                torch.diag(5.0 + 5.0 * torch.rand(spatial_dimension))
                for _ in range(batch_size)
            ]
        )
        # add a bit of noise to make the vectors not quite orthogonal
        basis_vectors = orthogonal_boxes + 0.1 * torch.randn(
            batch_size, spatial_dimension, spatial_dimension
        )
        return basis_vectors

    @pytest.fixture
    def lattice_parameters(self, batch_size, spatial_dimension, basis_vectors):
        lattice_params = torch.zeros(batch_size, int(spatial_dimension * (spatial_dimension + 1) / 2))
        lattice_params[:, :spatial_dimension] = torch.diagonal(basis_vectors, dim1=-2, dim2=-1)
        return lattice_params

    @pytest.fixture
    def relative_coordinates(
        self, batch_size, number_of_atoms, spatial_dimension, basis_vectors
    ):
        relative_coordinates = torch.rand(
            batch_size, number_of_atoms, spatial_dimension
        )
        return relative_coordinates

    @pytest.fixture
    def cartesian_forces(
        self, batch_size, number_of_atoms, spatial_dimension, basis_vectors
    ):
        cartesian_forces = torch.rand(batch_size, number_of_atoms, spatial_dimension)
        return cartesian_forces

    @pytest.fixture
    def noises(self, batch_size):
        return torch.rand(batch_size, 1)

    @pytest.fixture()
    def batch(
        self,
        relative_coordinates,
        atom_types,
        cartesian_forces,
        times,
        noises,
        basis_vectors,
        lattice_parameters,
        number_of_atoms,
        batch_size,
    ):
        return {
            NOISY_AXL_COMPOSITION: AXL(
                A=atom_types,
                X=relative_coordinates,
                L=lattice_parameters,
            ),
            TIME: times,
            UNIT_CELL: basis_vectors,  # TODO remove this
            NOISE: noises,
            CARTESIAN_FORCES: cartesian_forces,
            NUMBER_OF_ATOMS: torch.full((batch_size,), number_of_atoms, dtype=torch.long),
        }

    @pytest.fixture
    def number_of_edges(self):
        return 128

    @pytest.fixture
    def fake_cartesian_displacements(self, number_of_edges, spatial_dimension):
        return torch.rand(number_of_edges, spatial_dimension)

    def test_get_cartesian_pseudo_forces_contributions(
        self,
        force_field_augmented_score_network,
        harmonic_force_parameters,
        fake_cartesian_displacements,
    ):
        s = harmonic_force_parameters.strength
        r0 = harmonic_force_parameters.radial_cutoff

        expected_contributions = (
            force_field_augmented_score_network.repulsive_force._get_cartesian_pseudo_forces_contributions(
                fake_cartesian_displacements
            )
        )

        for r, expected_contribution in zip(
            fake_cartesian_displacements, expected_contributions
        ):
            r_norm = torch.linalg.norm(r)

            r_hat = r / r_norm
            computed_contribution = 2.0 * s * (r_norm - r0) * r_hat
            torch.testing.assert_allclose(expected_contribution, computed_contribution)

    def test_get_cartesian_pseudo_forces(
        self, batch, harmonic_force_parameters, force_field_augmented_score_network
    ):
        composition_i = batch[NOISY_AXL_COMPOSITION]
        basis_vectors = map_lattice_parameters_to_unit_cell_vectors(composition_i.L)
        cartesian_positions = get_positions_from_coordinates(composition_i.X, basis_vectors)

        adj_info = get_periodic_adjacency_information(
            cartesian_positions,
            basis_vectors,
            radial_cutoff=harmonic_force_parameters.radial_cutoff,
        )
        cartesian_displacements = (
            force_field_augmented_score_network.repulsive_force._get_cartesian_displacements(
                adj_info, cartesian_positions, basis_vectors
            )
        )
        cartesian_pseudo_force_contributions = (
            force_field_augmented_score_network.repulsive_force._get_cartesian_pseudo_forces_contributions(
                cartesian_displacements
            )
        )

        computed_cartesian_pseudo_forces = (
            force_field_augmented_score_network.repulsive_force._get_cartesian_pseudo_forces(
                cartesian_pseudo_force_contributions, adj_info, cartesian_positions
            )
        )

        # Compute the expected value by explicitly looping over indices, effectively checking that
        # the 'torch.scatter_add' is used correctly.
        expected_cartesian_pseudo_forces = torch.zeros_like(
            computed_cartesian_pseudo_forces
        )
        batch_indices = adj_info.edge_batch_indices
        source_indices, _ = adj_info.adjacency_matrix
        for batch_idx, src_idx, cont in zip(
            batch_indices, source_indices, cartesian_pseudo_force_contributions
        ):
            expected_cartesian_pseudo_forces[batch_idx, src_idx] += cont

        torch.testing.assert_allclose(
            computed_cartesian_pseudo_forces, expected_cartesian_pseudo_forces
        )

    def test_augmented_scores(
        self, batch, score_network, harmonic_force_parameters, force_field_augmented_score_network
    ):
        force_directions, force_importance = force_field_augmented_score_network.get_force_score_from_batch(
            batch
        )
        assert force_directions.sum() < 1e-4

        force_field_augmented_score_network(batch)
        # TODO : Add more test to make sure the updated scores make sense


def test_specific_scenario_sanity_check():
    """Test a specific scenario.

    It is very easy to have the forces point in the wrong direction. Here we check explicitly that
    the computed forces points AWAY from the neighbors.
    """
    spatial_dimension = 3
    harmonic_force_parameters = HarmonicForceParameters(radial_cutoff=0.4, strength=1)

    force_field_parameters = ForceFieldAugmentedScoreNetworkParameters(
        repulsive_force_parameters=harmonic_force_parameters,
        force_activation_scale=100.,
        use_for_training=False,
    )
    force_field_score_network = ForceFieldAugmentedScoreNetwork(
        score_network=None,
        force_field_parameters=force_field_parameters,
    )

    # Put two atoms on a straight line
    relative_coordinates = torch.tensor([[[0.35, 0.5, 0.0], [0.65, 0.5, 0.0]]])
    atom_types = torch.zeros_like(relative_coordinates[..., 0])
    lattice_parameters = torch.ones(1, 6)
    lattice_parameters[:, 3:] = 0

    basis_vectors = map_lattice_parameters_to_unit_cell_vectors(lattice_parameters)
    cartesian_positions = get_positions_from_coordinates(relative_coordinates, basis_vectors)

    forces = force_field_score_network.repulsive_force.get_cartesian_forces(
        A=atom_types,
        cartesian_positions=cartesian_positions,
        basis_vectors=basis_vectors
    )

    force_on_atom1 = forces[0, 0]
    force_on_atom2 = forces[0, 1]

    assert force_on_atom1[0] < 0.0
    assert force_on_atom2[0] > 0.0

    torch.testing.assert_allclose(force_on_atom1[1:], torch.zeros(2))
    torch.testing.assert_allclose(force_on_atom2[1:], torch.zeros(2))
    torch.testing.assert_allclose(
        force_on_atom1 + force_on_atom2, torch.zeros(spatial_dimension)
    )


class TestForceFieldAugmentedScoreNetworkZBL(BaseTestScoreNetwork):
    @pytest.fixture()
    def atomic_positions_Si32(self):
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
               [4.9723, 1.1048, 6.9392],
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
        return pos

    @pytest.fixture()
    def number_of_samples_Si32(self, atomic_positions_Si32):
        return len(atomic_positions_Si32)

    @pytest.fixture()
    def number_of_atoms_Si32(self, atomic_positions_Si32):
        return len(atomic_positions_Si32[0])

    @pytest.fixture()
    def basis_vectors_Si32(self):
        vec = [[[7.2200, 0.0000, 0.0000],
                [0.0000, 7.2200, 0.0000],
                [0.0000, 0.0000, 7.2200]]]
        return torch.tensor(vec, dtype=torch.float32, device="cpu")

    @pytest.fixture()
    def element_list_Si32(self):
        return ["Si"]

    @pytest.fixture()
    def number_of_elements_Si32(self, element_list_Si32):
        return len(element_list_Si32)

    @pytest.fixture()
    def times_Si32(self):
        return [1e-5, 0.5, 1.]

    @pytest.fixture()
    def atoms_Si32(self, element_list_Si32, number_of_atoms_Si32, atomic_positions_Si32, basis_vectors_Si32):
        atoms = Atoms(
            symbols=element_list_Si32 * number_of_atoms_Si32,
            positions=atomic_positions_Si32[0],
            cell=basis_vectors_Si32[0],
            pbc=True
        )
        return atoms

    @pytest.fixture()
    def lattice_parameters_Si32(self, basis_vectors_Si32):
        lattice_parameters = torch.tensor([[basis_vectors_Si32[0, 0, 0], basis_vectors_Si32[0, 1, 1],
                                            basis_vectors_Si32[0, 2, 2], basis_vectors_Si32[0, 1, 2],
                                            basis_vectors_Si32[0, 0, 2], basis_vectors_Si32[0, 0, 1]]])
        return lattice_parameters

    @pytest.fixture()
    def reduced_positions_Si32(self, atoms_Si32, basis_vectors_Si32):
        return (torch.from_numpy(atoms_Si32.get_scaled_positions(wrap=True))[None, ...]).to(basis_vectors_Si32)

    @pytest.fixture()
    def noise_parameters_Si32(self):
        noise_parameters = NoiseParameters(
            total_time_steps=3,
            sigma_min_cart=0.005,
            sigma_max_cart=2.0,
            schedule_type="exponential",
        )
        return noise_parameters

    @pytest.fixture()
    def score_network_parameters_Si32(self):
        score_network_parameters = EGNNScoreNetworkParameters(
            num_atom_types=1,
            number_of_bloch_wave_shells=1,
            message_n_hidden_dimensions=2,
            message_hidden_dimensions_size=64,
            node_n_hidden_dimensions=2,
            node_hidden_dimensions_size=64,
            coordinate_n_hidden_dimensions=2,
            coordinate_hidden_dimensions_size=64,
            residual=True,
            attention=False,
            normalize=True,
            tanh=True,
            coords_agg="mean",
            message_agg="mean",
            n_layers=4,
            edges="radial_cutoff",
            radial_cutoff=5.)
        return score_network_parameters

    @pytest.fixture()
    def sampling_parameters_Si32(self, number_of_samples_Si32, number_of_atoms_Si32, basis_vectors_Si32):
        sampling_parameters = PredictorCorrectorSamplingParameters(number_of_samples=number_of_samples_Si32,
                                                                   spatial_dimension=3,
                                                                   number_of_corrector_steps=1,
                                                                   num_atom_types=1,
                                                                   number_of_atoms=number_of_atoms_Si32,
                                                                   use_fixed_lattice_parameters=True,
                                                                   cell_dimensions=basis_vectors_Si32[0],
                                                                   record_samples=True)
        return sampling_parameters

    @pytest.fixture()
    def composition_Si32(self, number_of_atoms_Si32, reduced_positions_Si32, lattice_parameters_Si32):
        A = torch.tensor([[1] * number_of_atoms_Si32])
        return AXL(A=A, X=reduced_positions_Si32, L=lattice_parameters_Si32)

    def test_Si_forcefield_ZBL(self, number_of_samples_Si32, number_of_atoms_Si32, number_of_elements_Si32,
                               basis_vectors_Si32, element_list_Si32, times_Si32,
                               score_network_parameters_Si32, sampling_parameters_Si32, noise_parameters_Si32,
                               composition_Si32):
        """Test for ZBLRepulsionScore using masked_atom type of 14.5"""
        # 1. Prepare the objects for the test
        noise_sched = NoiseScheduler(noise_parameters_Si32, num_classes=number_of_elements_Si32 + 1)
        noise, _ = noise_sched.get_all_sampling_parameters()
        sigmas = noise.sigma
        forces = torch.zeros([number_of_atoms_Si32, 3]).to(basis_vectors_Si32)

        zbl_parameters = ZBLForceParameters(
            radial_cutoff=2.19293,
            inner_radius_fraction=0.5552844824048191,
            element_list=element_list_Si32,
        )

        score_network = EGNNScoreNetwork(score_network_parameters_Si32)
        force_field_parameters = ForceFieldAugmentedScoreNetworkParameters(
            repulsive_force_parameters=zbl_parameters,
            force_activation_scale=100.,
            use_for_training=False,
        )
        model = ForceFieldAugmentedScoreNetwork(
            score_network=score_network,
            force_field_parameters=force_field_parameters,
        )
        model.eval()  # Set model.training to False
        zbl_force = model.repulsive_force

        fake_model_output = AXL(
            A=torch.zeros([number_of_samples_Si32,
                           number_of_atoms_Si32,
                           number_of_elements_Si32]).to(basis_vectors_Si32),
            X=torch.zeros([number_of_samples_Si32,
                           number_of_atoms_Si32,
                           3]).to(basis_vectors_Si32),
            L=torch.zeros([1, 6]).to(basis_vectors_Si32)
        )  # We want the EGNN model to return a zeros

        generator = LangevinGenerator(noise_parameters=noise_parameters_Si32,
                                      sampling_parameters=sampling_parameters_Si32,
                                      axl_network=model)

        # 2. Calculate the scores and make an updated structure
        force_scores, updated_structs = [], []
        g2_squared_test = [1e-4, 1e-5, 1e-6]  # The g2_i decreases with t->0. Override to get predictability
        with patch.object(model._score_network, "forward", return_value=fake_model_output):
            for i in range(len(times_Si32)):
                time_tensor = (times_Si32[i] * torch.ones(number_of_samples_Si32, 1)).to(composition_Si32.X)
                sigma_tensor = sigmas[i] * torch.ones_like(time_tensor)
                batch = {
                    NOISY_AXL_COMPOSITION: composition_Si32,
                    TIME: time_tensor,
                    NOISE: sigma_tensor,
                    CARTESIAN_FORCES: forces,
                }

                force_score = model(batch)
                force_scores.append(force_score)

                sigma_i = noise_sched._sigma_array[i]

                g2i_updated_structs = []
                for g2_i in g2_squared_test:
                    score_weight = g2_i * torch.ones_like(noise_sched._g_squared_array[i])
                    gaussian_noise = torch.zeros_like(noise_sched._g_array[i])  # No gaussian noise
                    z_noise = torch.zeros_like(force_score.X)

                    g2i_updated_structs.append(
                        generator._relative_coordinates_update(
                            relative_coordinates=composition_Si32.X,
                            sigma_normalized_scores=force_score.X,
                            sigma_cart=sigma_i,
                            score_weight=score_weight,
                            gaussian_noise_weight=gaussian_noise,
                            z=z_noise,
                        )
                    )
                updated_structs.append(g2i_updated_structs)

        # 3. Verify the results
        # Note : We do small displacements and assume forces should decrease with bigger g2_i or smaller t.

        # 3.1 Verify the maximal force decreases more with bigger g2_i
        for g2i_updated_structs in updated_structs:
            force_g2i_small = zbl_force.get_cartesian_forces(
                composition_Si32.A,
                g2i_updated_structs[0],
                basis_vectors_Si32
            )
            force_g2i_smaller = zbl_force.get_cartesian_forces(
                composition_Si32.A,
                g2i_updated_structs[1],
                basis_vectors_Si32
            )
            force_g2i_smallest = zbl_force.get_cartesian_forces(
                composition_Si32.A,
                g2i_updated_structs[2],
                basis_vectors_Si32
            )

            assert force_g2i_small.abs().max() <= force_g2i_smaller.abs().max()
            assert force_g2i_smaller.abs().max() <= force_g2i_smallest.abs().max()

        # 3.2 Verify that the maximal force decreases more with t->0 (only true because every g2_i is equal)
        force_zerotime = zbl_force.get_cartesian_forces(composition_Si32.A, updated_structs[0][0], basis_vectors_Si32)
        force_halftime = zbl_force.get_cartesian_forces(composition_Si32.A, updated_structs[1][0], basis_vectors_Si32)
        force_Ttime = zbl_force.get_cartesian_forces(composition_Si32.A, updated_structs[2][0], basis_vectors_Si32)

        assert force_zerotime.abs().max() <= force_halftime.abs().max()
        assert force_halftime.abs().max() <= force_Ttime.abs().max()

        # 3.3 Verify that ZBL force didn't change the struct at t=T (gaussian noise was also removed for this test)
        Ttime_struct = updated_structs[2][0]
        assert torch.allclose(Ttime_struct, composition_Si32.X, atol=1e-4)

        # 3.4 Verify that the minimal interatomic distances is bigger in the updated_struct
        initial_adj, initial_dist = zbl_force.get_atomic_distances(composition_Si32.X, basis_vectors_Si32)
        zerotime_adj, zerotime_dist = zbl_force.get_atomic_distances(updated_structs[0][0], basis_vectors_Si32)
        halftime_adj, halftime_dist = zbl_force.get_atomic_distances(updated_structs[1][0], basis_vectors_Si32)

        # Here, we need to filter out atoms outside rcut (dist=-1) with a mask
        initial_min_dist = initial_dist.masked_fill(initial_dist < 0, float("inf")).min()
        zerotime_min_dist = zerotime_dist.masked_fill(zerotime_dist < 0, float("inf")).min()
        halftime_min_dist = halftime_dist.masked_fill(halftime_dist < 0, float("inf")).min()

        assert zerotime_min_dist >= halftime_min_dist
        assert halftime_min_dist >= initial_min_dist

    def test_ZBL_with_no_forces(self, number_of_samples_Si32, number_of_atoms_Si32, number_of_elements_Si32,
                                basis_vectors_Si32, element_list_Si32, score_network_parameters_Si32,
                                sampling_parameters_Si32, noise_parameters_Si32, composition_Si32):
        """Smoke test for ForceFieldAugmentedScoreNetwork + ZBL with no interacting atoms."""
        # 1. Create the object
        time = 1e-4
        sigma = 5e-3
        g2_i = 1e-4
        noise_sched = NoiseScheduler(noise_parameters_Si32, num_classes=number_of_elements_Si32 + 1)
        noise, _ = noise_sched.get_all_sampling_parameters()
        forces = torch.zeros([number_of_atoms_Si32, 3]).to(basis_vectors_Si32)

        zbl_parameters = ZBLForceParameters(
            radial_cutoff=1e-4,  # Tiny radial_cutoff so there's no interacting pairs
            inner_radius_fraction=0.5,
            element_list=element_list_Si32,
        )

        score_network = EGNNScoreNetwork(score_network_parameters_Si32)
        force_field_parameters = ForceFieldAugmentedScoreNetworkParameters(
            repulsive_force_parameters=zbl_parameters,
            force_activation_scale=100.0,
            use_for_training=False,
        )
        model = ForceFieldAugmentedScoreNetwork(
            score_network=score_network,
            force_field_parameters=force_field_parameters,
        )
        model.eval()

        generator = LangevinGenerator(noise_parameters=noise_parameters_Si32,
                                      sampling_parameters=sampling_parameters_Si32,
                                      axl_network=model)

        # Patch EGNN to output zeros so any non-zero would have to come from ZBL
        fake_model_output = AXL(
            A=torch.zeros(
                (number_of_samples_Si32, number_of_atoms_Si32, number_of_elements_Si32),
                device=basis_vectors_Si32.device,
                dtype=basis_vectors_Si32.dtype,
            ),
            X=torch.zeros(
                (number_of_samples_Si32, number_of_atoms_Si32, 3),
                device=basis_vectors_Si32.device,
                dtype=basis_vectors_Si32.dtype,
            ),
            L=torch.zeros_like(composition_Si32.L),
        )

        # 2. Do the calculations
        with patch.object(model._score_network, "forward", return_value=fake_model_output):
            time_tensor = (time * torch.ones(number_of_samples_Si32, 1)).to(composition_Si32.X)
            sigma_tensor = sigma * torch.ones_like(time_tensor)
            batch = {
                NOISY_AXL_COMPOSITION: composition_Si32,
                TIME: time_tensor,
                NOISE: sigma_tensor,
                CARTESIAN_FORCES: forces,
            }

            force_score = model(batch)
            score_weight = g2_i * torch.ones_like(noise_sched._g_squared_array)
            gaussian_noise = torch.zeros_like(noise_sched._g_array)  # No gaussian noise
            z_noise = torch.zeros_like(force_score.X)

            updated_struct = generator._relative_coordinates_update(
                relative_coordinates=composition_Si32.X,
                sigma_normalized_scores=force_score.X,
                sigma_cart=sigma,
                score_weight=score_weight,
                gaussian_noise_weight=gaussian_noise,
                z=z_noise,
            )

        # 3. Assert everything works as expected
        # 3.1 The force_score should filled with 0.
        assert torch.allclose(force_score.X, torch.zeros_like(force_score.X), atol=1e-4)

        # 3.2 The updated_struct should be identical to the initial one
        assert torch.allclose(updated_struct, composition_Si32.X, atol=1e-4)
