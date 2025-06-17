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


@pytest.fixture(params=['uncertainty_threshold', 'excise_top_k_environment'])
def excision_criterion(request):
    return request.param


@pytest.fixture()
def excision_criterion_parameter(excision_criterion):
    match excision_criterion:
        case 'uncertainty_threshold':
            return 0.001
        case 'excise_top_k_environment':
            return 8
        case _:
            raise NotImplementedError("Unknown excision criterion")


@pytest.fixture(params=['noop', 'nearest_neighbors', 'spherical_cutoff'])
def excision_dictionary(request, excision_criterion, excision_criterion_parameter):
    algorithm = request.param
    excision_dict = dict(algorithm=algorithm)
    match algorithm:
        case 'noop':
            pass
        case 'nearest_neighbors':
            excision_dict['number_of_neighbors'] = 16
            excision_dict[excision_criterion] = excision_criterion_parameter
        case 'spherical_cutoff':
            excision_dict[excision_criterion] = excision_criterion_parameter
            excision_dict['radial_cutoff'] = 2.0
        case _:
            raise NotImplementedError("Unknown excision case.")

    return excision_dict


# TODO: Implement something for excise_and_repaint
@pytest.fixture(params=['noop', 'excise_and_random', 'excise_and_noop'])
def sampling_dictionary(request, excision_dictionary):
    algorithm = request.param
    sampling_dict = dict(algorithm=algorithm)
    sampling_dict['excision'] = excision_dictionary
    match algorithm:
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
    sample_maker = get_sample_maker_from_configuration(sampling_dictionary, element_list)
    assert type(sample_maker) is expected_sampler_maker_class
