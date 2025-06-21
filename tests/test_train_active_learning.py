import pytest

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.excise_and_noop_sample_maker import \
    ExciseAndNoOpSampleMaker  # noqa
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.excise_and_random_sample_maker import \
    ExciseAndRandomSampleMaker  # noqa
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.excise_and_repaint_sample_maker import \
    ExciseAndRepaintSampleMaker  # noqa
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.no_op_sample_maker import \
    NoOpSampleMaker
from diffusion_for_multi_scale_molecular_dynamics.train_active_learning import \
    get_sample_maker_from_configuration


@pytest.fixture()
def element_list():
    return ['Ca', 'Si', 'H', 'C']


@pytest.fixture(params=[('noop', 'noop'),
                        ('nearest_neighbors', 'excise_and_random'),
                        ('spherical_cutoff', 'excise_and_random'),
                        ('nearest_neighbors', 'excise_and_noop'),
                        ('spherical_cutoff', 'excise_and_noop')])
def excision_and_sampler_maker_algorithm_combination(request):
    return request.param


@pytest.fixture
def excision_algorithm(excision_and_sampler_maker_algorithm_combination):
    return excision_and_sampler_maker_algorithm_combination[0]


@pytest.fixture
def sampling_algorithm(excision_and_sampler_maker_algorithm_combination):
    return excision_and_sampler_maker_algorithm_combination[1]


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


# TODO: Implement something for excise_and_repaint
@pytest.fixture()
def sampling_dictionary(sampling_algorithm, excision_dictionary):
    sampling_dict = dict(algorithm=sampling_algorithm)
    sampling_dict['excision'] = excision_dictionary
    match sampling_algorithm:
        case 'noop':
            pass
        case 'excise_and_random':
            sampling_dict['total_number_of_atoms'] = 64
            sampling_dict['random_coordinates_algorithm'] = "true_random"
            sampling_dict['max_attempts'] = 4
            sampling_dict['minimal_interatomic_distance'] = 1.0
            sampling_dict['sample_box_size'] = [10.1, 11.2, 12.3]
        case 'excise_and_noop':
            sampling_dict['sample_box_size'] = [10.1, 11.2, 12.3]
        case _:
            raise NotImplementedError("Unknown sampling case.")
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


def test_get_sample_maker_from_configuration(sampling_dictionary, element_list, expected_sampler_maker_class):
    uncertainty_threshold = 0.001
    sample_maker = get_sample_maker_from_configuration(sampling_dictionary, uncertainty_threshold, element_list)
    assert type(sample_maker) is expected_sampler_maker_class
