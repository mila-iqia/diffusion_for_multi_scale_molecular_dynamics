import sys  # noqa
from unittest.mock import patch  # noqa

from diffusion_for_multi_scale_molecular_dynamics import ROOT_DIR  # noqa

sys.path.append(str(ROOT_DIR / "../experiments/atom_types_only_experiments/patches"))

from patches.fixed_position_data_loader import FixedPositionDataModule  # noqa
from patches.identity_noiser import IdentityNoiser  # noqa
from patches.identity_relative_coordinates_langevin_generator import \
    IdentityRelativeCoordinatesUpdateLangevinGenerator  # noqa

from diffusion_for_multi_scale_molecular_dynamics.train_diffusion import \
    main as train_diffusion_main  # noqa

if __name__ == "__main__":
    # We must patch 'where the class is looked up', not where it is defined.
    # See: https://docs.python.org/3/library/unittest.mock.html#where-to-patch

    # Patch the dataloader to always use the same atomic relative coordinates.
    target1 = "diffusion_for_multi_scale_molecular_dynamics.train_diffusion.LammpsForDiffusionDataModule"

    # Patch the noiser to never change the relative coordinates"
    target2 = ("diffusion_for_multi_scale_molecular_dynamics.models."
               "axl_diffusion_lightning_model.RelativeCoordinatesNoiser")

    # Patch the generator to never change the relative coordinates"
    target3 = "diffusion_for_multi_scale_molecular_dynamics.generators.instantiate_generator.LangevinGenerator"

    with (
        patch(target=target1, new=FixedPositionDataModule),
        patch(target=target2, new=IdentityNoiser),
        patch(target=target3, new=IdentityRelativeCoordinatesUpdateLangevinGenerator),
    ):
        train_diffusion_main()
