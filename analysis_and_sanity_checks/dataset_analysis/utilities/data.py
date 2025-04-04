from diffusion_for_multi_scale_molecular_dynamics import DATA_DIR
from diffusion_for_multi_scale_molecular_dynamics.data.diffusion.lammps_for_diffusion_data_module import (
    LammpsDataModuleParameters, LammpsForDiffusionDataModule)
from diffusion_for_multi_scale_molecular_dynamics.noise_schedulers.noise_parameters import \
    NoiseParameters


def get_data_module(dataset_name: str):
    """Convenience method to get the data module."""
    match dataset_name:
        case "Si_diffusion_1x1x1":
            max_atom = 8
        case "Si_diffusion_2x2x2":
            max_atom = 64
        case "Si_diffusion_3x3x3":
            max_atom = 216
        case _:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

    lammps_run_dir = DATA_DIR / dataset_name
    processed_dataset_dir = lammps_run_dir / "processed"

    cache_dir = lammps_run_dir / "cache"

    data_params = LammpsDataModuleParameters(batch_size=1024,
                                             max_atom=max_atom,
                                             noise_parameters=NoiseParameters(total_time_steps=1),
                                             use_optimal_transport=False,
                                             use_fixed_lattice_parameters=True,
                                             elements=['Si'])
    datamodule = LammpsForDiffusionDataModule(
        lammps_run_dir=lammps_run_dir,
        processed_dataset_dir=processed_dataset_dir,
        hyper_params=data_params,
        working_cache_dir=cache_dir,
    )
    datamodule.setup()

    return datamodule
