from unittest.mock import patch

from patches.fixed_position_data_module import load_fixed_position_data_module
from patches.identity_relative_coordinates_langevin_generator import \
    instantiate_identity_relative_coordinates_generator

from diffusion_for_multi_scale_molecular_dynamics.train_diffusion import \
    main as train_diffusion_main

if __name__ == "__main__":
    # We must patch 'where the class is looked up', not where it is defined.
    # See: https://docs.python.org/3/library/unittest.mock.html#where-to-patch

    # Patch the data module loading to always use the same relative coordinates.
    target1 = "diffusion_for_multi_scale_molecular_dynamics.train_diffusion.load_data_module"

    # Patch the generator to never change the relative coordinates"
    target2 = ("diffusion_for_multi_scale_molecular_dynamics.models."
               "axl_diffusion_lightning_model.instantiate_generator")

    with (
        patch(target=target1, new=load_fixed_position_data_module),
        patch(target=target2, new=instantiate_identity_relative_coordinates_generator),
    ):
        train_diffusion_main()
