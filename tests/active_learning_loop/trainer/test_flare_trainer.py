import numpy as np

from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.single_point_calculators.flare_single_point_calculator import \
    FlareSinglePointCalculator  # noqa
from diffusion_for_multi_scale_molecular_dynamics.active_learning_loop.trainer.flare_trainer import \
    FlareTrainer  # noqa
from tests.active_learning_loop.trainer.base_test_flare import BaseTestFlare


class TestFlareTrainer(BaseTestFlare):

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
