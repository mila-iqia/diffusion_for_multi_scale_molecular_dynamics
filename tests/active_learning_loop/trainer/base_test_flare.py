import numpy as np
import pytest

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.base_single_point_calculator import \
    SinglePointCalculation  # noqa
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.trainer.flare_trainer import \
    FlareTrainer  # noqa
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.trainer.flare_trainer import \
    FlareConfiguration


class BaseTestFlare:

    def instantiate_flare_trainer(self, flare_configuration, labelled_structure, active_environment_indices):
        flare_trainer = FlareTrainer(flare_configuration)
        flare_trainer.add_labelled_structure(labelled_structure,
                                             active_environment_indices=active_environment_indices)
        return flare_trainer

    @pytest.fixture()
    def flare_configuration(self, list_element_symbols):
        return FlareConfiguration(cutoff=5.0,
                                  elements=list_element_symbols,
                                  n_radial=8,
                                  lmax=3,
                                  variance_type='local')

    @pytest.fixture()
    def labelled_structure(self, structure):
        number_of_atoms = len(structure)
        forces = np.random.rand(number_of_atoms, 3)
        energy = np.random.rand()
        return SinglePointCalculation(calculation_type='dummy_test',
                                      structure=structure,
                                      forces=forces,
                                      energy=energy)

    @pytest.fixture()
    def active_environment_indices(self, structure):
        return list(np.random.choice(np.arange(len(structure)), 4))
