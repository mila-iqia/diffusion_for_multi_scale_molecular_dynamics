import lightning as pl
import pytest

from diffusion_for_multi_scale_molecular_dynamics.loss.loss_parameters import \
    create_loss_parameters
from diffusion_for_multi_scale_molecular_dynamics.models.axl_diffusion_lightning_model import (
    AXLDiffusionLightningModel, AXLDiffusionParameters)
from diffusion_for_multi_scale_molecular_dynamics.models.optimizer import \
    create_optimizer_parameters
from diffusion_for_multi_scale_molecular_dynamics.models.score_networks.mlp_score_network import \
    MLPScoreNetworkParameters

try:
    import flare_pp  # noqa
except ImportError:
    pytest.skip("Skipping FLARE tests:  optional FLARE dependencies not installed.", allow_module_level=True)

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.configuration_parsing import \
    get_sample_maker_from_configuration
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.excise_and_noop_sample_maker import \
    ExciseAndNoOpSampleMaker  # noqa
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.excise_and_random_sample_maker import \
    ExciseAndRandomSampleMaker  # noqa
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.excise_and_repaint_sample_maker import \
    ExciseAndRepaintSampleMaker  # noqa
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.no_op_sample_maker import \
    NoOpSampleMaker


@pytest.fixture()
def element_list():
    return ['Ca', 'Si', 'H', 'C']


@pytest.fixture(params=[('noop', 'noop'),
                        ('nearest_neighbors', 'excise_and_repaint'),
                        ('spherical_cutoff', 'excise_and_repaint'),
                        ('nearest_neighbors', 'excise_and_random'),
                        ('spherical_cutoff', 'excise_and_random'),
                        ('nearest_neighbors', 'excise_and_noop'),
                        ('spherical_cutoff', 'excise_and_noop')])
def excision_and_sampler_maker_algorithm_combination(request):
    return request.param


@pytest.fixture
def noise_dictionary():
    return dict(total_time_steps=10,
                sigma_min=0.0001,
                sigma_max=0.2,
                schedule_type='linear',
                corrector_step_epsilon=2.5e-8)


@pytest.fixture
def repaint_generator_dictionary():
    return dict(number_of_atoms=64,
                number_of_corrector_steps=2,
                one_atom_type_transition_per_step=False,
                atom_type_greedy_sampling=False,
                atom_type_transition_in_corrector=False,
                record_samples=False)


@pytest.fixture
def excision_algorithm(excision_and_sampler_maker_algorithm_combination):
    return excision_and_sampler_maker_algorithm_combination[0]


@pytest.fixture
def sampling_algorithm(excision_and_sampler_maker_algorithm_combination):
    return excision_and_sampler_maker_algorithm_combination[1]


@pytest.fixture
def axl_diffusion_lightning_model(element_list):
    score_network_parameters = MLPScoreNetworkParameters(number_of_atoms=64,
                                                         num_atom_types=len(element_list),
                                                         n_hidden_dimensions=8,
                                                         hidden_dimensions_size=8,
                                                         noise_embedding_dimensions_size=8,
                                                         relative_coordinates_embedding_dimensions_size=8,
                                                         time_embedding_dimensions_size=8,
                                                         atom_type_embedding_dimensions_size=8,
                                                         lattice_parameters_embedding_dimensions_size=8)

    loss_parameters = create_loss_parameters(dict(coordinates_algorithm="mse"))
    optimizer_parameters = create_optimizer_parameters(dict(name="adam", learning_rate=0.001))

    parameters = AXLDiffusionParameters(score_network_parameters=score_network_parameters,
                                        loss_parameters=loss_parameters,
                                        optimizer_parameters=optimizer_parameters)

    return AXLDiffusionLightningModel(parameters)


@pytest.fixture()
def path_to_score_network_checkpoint(sampling_algorithm, axl_diffusion_lightning_model, tmp_path):

    if sampling_algorithm == 'excise_and_repaint':
        path = tmp_path / "fake_score_network.ckpt"
        # We have to create these convoluted objects to save a checkpoint.
        trainer = pl.Trainer()
        trainer.strategy.connect(axl_diffusion_lightning_model)
        trainer.save_checkpoint(path)
        return path
    else:
        return None


@pytest.fixture()
def excision_dictionary(excision_algorithm):
    excision_dict = dict(algorithm=excision_algorithm)
    match excision_algorithm:
        case 'noop':
            pass
        case 'nearest_neighbors':
            excision_dict['number_of_neighbors'] = 16
        case 'spherical_cutoff':
            excision_dict['radial_cutoff'] = 2.0
        case _:
            raise NotImplementedError("Unknown excision case.")

    return excision_dict


@pytest.fixture()
def sample_maker_dictionary(sampling_algorithm):
    sample_maker_dict = dict(algorithm=sampling_algorithm)
    sample_box_size = [10.1, 11.2, 12.3]

    match sampling_algorithm:
        case 'noop':
            pass
        case 'excise_and_repaint':
            sample_maker_dict['sample_box_strategy'] = 'fixed'
            sample_maker_dict['sample_box_size'] = sample_box_size
        case 'excise_and_random':
            sample_maker_dict['total_number_of_atoms'] = 64
            sample_maker_dict['random_coordinates_algorithm'] = "true_random"
            sample_maker_dict['max_attempts'] = 4
            sample_maker_dict['minimal_interatomic_distance'] = 1.0
            sample_maker_dict['sample_box_size'] = sample_box_size
        case 'excise_and_noop':
            sample_maker_dict['sample_box_size'] = sample_box_size
        case _:
            raise NotImplementedError("Unknown sampling case.")
    return sample_maker_dict


@pytest.fixture()
def sampling_dictionary(sampling_algorithm,
                        excision_dictionary,
                        sample_maker_dictionary,
                        noise_dictionary,
                        repaint_generator_dictionary):
    sampling_dict = sample_maker_dictionary.copy()

    if sampling_algorithm != 'noop':
        sampling_dict['excision'] = excision_dictionary

    if sampling_algorithm == 'excise_and_repaint':
        sampling_dict['noise'] = noise_dictionary
        sampling_dict['repaint_generator'] = repaint_generator_dictionary

    return sampling_dict


@pytest.fixture()
def expected_sampler_maker_class(sampling_dictionary):
    algorithm = sampling_dictionary['algorithm']
    match algorithm:
        case 'noop':
            return NoOpSampleMaker
        case "excise_and_repaint":
            return ExciseAndRepaintSampleMaker
        case "excise_and_random":
            return ExciseAndRandomSampleMaker
        case "excise_and_noop":
            return ExciseAndNoOpSampleMaker
        case _:
            raise NotImplementedError("Unknown sampling case.")


def test_get_sample_maker_from_configuration(sampling_dictionary,
                                             element_list,
                                             expected_sampler_maker_class,
                                             path_to_score_network_checkpoint):

    uncertainty_threshold = 0.001
    sample_maker = get_sample_maker_from_configuration(sampling_dictionary,
                                                       uncertainty_threshold,
                                                       element_list,
                                                       path_to_score_network_checkpoint)
    assert type(sample_maker) is expected_sampler_maker_class
