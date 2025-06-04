import importlib.util

import numpy as np
import pytest

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.base_single_point_calculator import \
    SinglePointCalculation  # noqa
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.flare_single_point_calculator import \
    FlareSinglePointCalculator  # noqa
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.trainer.flare_trainer import \
    FlareTrainer  # noqa
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.trainer.flare_trainer import \
    FlareConfiguration


@pytest.mark.skipif(importlib.util.find_spec("flare_pp") is None, reason="FLARE is not installed")
class TestFlareTrainer:

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

    def test_smoke_test_flare_trainer(self, flare_configuration, labelled_structure, active_environment_indices):
        # Test that we can create a flare trainer and add a structure to it.
        self.instantiate_flare_trainer(flare_configuration, labelled_structure, active_environment_indices)

    def test_checkpoint_flare_trainer(self, flare_configuration, labelled_structure,
                                      active_environment_indices, tmp_path):
        # Test that we can create a flare trainer and add a structure to it.
        flare_trainer = self.instantiate_flare_trainer(flare_configuration,
                                                       labelled_structure,
                                                       active_environment_indices)

        checkpoint_path = tmp_path / "checkpoint_flare_for_test.json"
        flare_trainer.write_checkpoint_to_disk(checkpoint_path)
        assert checkpoint_path.is_file()

        read_flare_trainer = FlareTrainer.from_checkpoint(checkpoint_path)

        calculator1 = FlareSinglePointCalculator(flare_trainer.sgp_model)
        calculator2 = FlareSinglePointCalculator(read_flare_trainer.sgp_model)

        calculation1 = calculator1.calculate(labelled_structure.structure)
        calculation2 = calculator2.calculate(labelled_structure.structure)

        np.testing.assert_allclose(calculation1.forces, calculation2.forces)
        np.testing.assert_allclose(calculation1.energy, calculation2.energy)
        np.testing.assert_allclose(calculation1.uncertainties, calculation2.uncertainties)
