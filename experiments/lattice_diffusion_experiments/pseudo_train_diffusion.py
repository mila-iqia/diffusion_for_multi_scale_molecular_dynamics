import sys  # noqa
from pathlib import Path
from typing import Optional
from unittest.mock import patch  # noqa

from patches.identity_noiser import RelativeCoordinatesIdentityNoiser  # noqa
from patches.lattice_demo_dataloader import LatticeDemoDataModule  # noqa
from patches.lattice_demo_dataloader import LatticeDemoParameters  # noqa

from diffusion_for_multi_scale_molecular_dynamics import ROOT_DIR  # noqa
from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.data_loader import \
    LammpsLoaderParameters
from diffusion_for_multi_scale_molecular_dynamics.train_diffusion import \
    main as train_diffusion_main  # noqa

sys.path.append(str(ROOT_DIR / "../experiments/lattice_diffusion_experiments/patches"))

LATTICE_AVERAGE = [2.0, 3.0, 4.0]
LATTICE_STD = [0.2, 0.3, 0.4]
NUM_TRAIN_SAMPLES = 102400
NUM_VALID_SAMPLES = 51200

EXP_PATH = Path("/Users/simonblackburn/projects/courtois2024/experiments/"
                + "lattice_diffusion_experiments/experiments_with_tanh")
CONFIG_PATH = str(EXP_PATH / "config_mlp.yaml")
OUTPUT_PATH = str(EXP_PATH / "output")


def lattice_dm_wrapper(
    lammps_run_dir: str,
    processed_dataset_dir: str,
    hyper_params: LammpsLoaderParameters,
    working_cache_dir: Optional[str] = None,
):
    """This replaces the arguments from the normal datamodule to the experiment one."""
    lattice_dm_data = LatticeDemoParameters(
        batch_size=hyper_params.batch_size,
        num_workers=hyper_params.num_workers,
        max_atom=hyper_params.max_atom,
        spatial_dimension=hyper_params.spatial_dimension,
        use_physical_positions=False,
        num_atom_types=len(hyper_params.elements),
        lattice_averages=LATTICE_AVERAGE,
        lattice_stddev=LATTICE_STD,
        train_num_samples=NUM_TRAIN_SAMPLES,
        valid_num_samples=NUM_VALID_SAMPLES,
    )
    datamodule = LatticeDemoDataModule(lattice_dm_data)
    return datamodule


if __name__ == "__main__":
    # We must patch 'where the class is looked up', not where it is defined.
    # See: https://docs.python.org/3/library/unittest.mock.html#where-to-patch

    # Patch the dataloader to use the lattice diffusion experimental one.
    target_dm = "diffusion_for_multi_scale_molecular_dynamics.train_diffusion.LammpsForDiffusionDataModule"

    # Patch the noiser to never change the relative coordinates"
    target_x_noiser = (
        "diffusion_for_multi_scale_molecular_dynamics.models."
        "axl_diffusion_lightning_model.RelativeCoordinatesNoiser"
    )

    args = [
        "--accelerator",
        "cpu",
        "--config",
        str(CONFIG_PATH),
        "--data",
        "./",
        "--processed_datadir",
        "./",
        "--dataset_working_dir",
        "./",
        "--output",
        str(OUTPUT_PATH),
    ]

    with (
        patch(target=target_dm, new=lattice_dm_wrapper),
        patch(target=target_x_noiser, new=RelativeCoordinatesIdentityNoiser),
        # patch(target=target_a_noiser, new=AtomTypesIdentityNoiser),
    ):
        train_diffusion_main(args)
