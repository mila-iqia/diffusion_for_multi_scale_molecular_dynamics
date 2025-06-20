import numpy as np
import pytest

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.sample_maker.excise_and_noop_sample_maker import (  # noqa
    ExciseAndNoOpSampleMaker, ExciseAndNoOpSampleMakerArguments)
from tests.active_learning_loop.sample_maker.base_test_sample_maker import \
    BaseTestExciseSampleMaker


class TestExciseAndNoOpSampleMaker(BaseTestExciseSampleMaker):
    @pytest.fixture(scope="class", autouse=True)
    def set_random_seed(self):
        np.random.seed(34345345)

    @pytest.fixture()
    def sample_maker_arguments(self, element_list, sample_box_strategy,
                               sample_box_size, number_of_samples_per_substructure):
        return ExciseAndNoOpSampleMakerArguments(element_list=element_list,
                                                 sample_box_strategy=sample_box_strategy,
                                                 sample_box_size=sample_box_size,
                                                 number_of_samples_per_substructure=number_of_samples_per_substructure)

    @pytest.fixture()
    def sample_maker(self, sample_maker_arguments, atom_selector, excisor, element_list):
        return ExciseAndNoOpSampleMaker(
            sample_maker_arguments=sample_maker_arguments,
            atom_selector=atom_selector,
            environment_excisor=excisor)
