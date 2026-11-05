# README

# Tutorials

These notebooks are a guided tour of the code base. Each one follows one or two of the scripts in `examples/`,
and explains the choices behind them step by step.

| Tutorial | Examples | Topic |
| --- | --- | --- |
| 01 | `01_activelearning_mtp_md_sw_noop.py`, `02_activelearning_mtp_artn_sw_noop.py` | Active learning of a MTP potential with a dynamic driver using MD or ARTn and an oracle that labels using Stillinger-Weber. |
| 02 | `03_train_diffusion_model.py`, `04_generate_and_repaint.py` | Training a diffusion model on Si8 then generating new cells from it. |
| 03 | `05_activelearning_mtp_md_abinit_repaint.py` | Active learning with the Abinit oracle and a diffusion model.  |

> [!NOTE]
> The tutorials call external programs: LAMMPS with ARTn plugin and MLIP-3, the MLIP-3 executable (mlp) and Abinit.
> 
> 
> To make things easier, we suggest you define the following
> environment variables (or add them to your `~/.bashrc` or `~/.zshrc`):
> 
> ```bash
> export DFMSMD_LAMMPS_PATH=Replace_with_absolute_path_to_lmp
> export DFMSMD_MLP_PATH=Replace_with_absolute_path_to_mlp
> export DFMSMD_ARTN_LIBRARY_PLUGIN_PATH=Replace_with_absolute_path_to_libartn-lmp.so
> export DFMSMD_ABINIT_COMMAND="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 abinit"
> export DFMSMD_ABINIT_MPI_TASKS=4
> ```
> 
> Then launch Jupyter from that same terminal: a notebook only sees the environment variables of the shell that started Jupyter. If you prefer not to use environment variables, replace the `None` in the `get_setting(None, "DFMSMD_...")` lines of the examples by your paths.
> 

Each tutorial only needs some of these variables:

| Variable | Description | Tutorials |
| --- | --- | --- |
| `DFMSMD_LAMMPS_PATH` | A LAMMPS executable with the `mtp/extrapolation` pair style (lammps-mtp-kokkos), built with the PLUGIN package for ARTn. | 01, 03 |
| `DFMSMD_MLP_PATH` | The `mlp` executable of MLIP-3, which fits the MTP. | 01, 03 |
| `DFMSMD_ARTN_LIBRARY_PLUGIN_PATH` | The compiled ARTn plugin (`libartn-lmp.so`). | 01 |
| `DFMSMD_ABINIT_COMMAND` | The command that runs Abinit. Optional, by default one thread per MPI task. | 03 |
| `DFMSMD_ABINIT_MPI_TASKS` | The number of MPI tasks of Abinit, usually the number of physical cores. Optional, 4 by default. | 03 |

Tutorial 02 needs none of them, only the LAMMPS python binding (`import lammps`), which computes the Stillinger-Weber energies and forces in-process.
