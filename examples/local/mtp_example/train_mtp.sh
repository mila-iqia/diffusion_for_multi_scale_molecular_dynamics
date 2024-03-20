#!/bin/bash

# Change this path to the root folder of the repo
ROOT_DIR="${HOME}/ic-collab/courtois_collab/crystal_diffusion"

MLIP_PATH="${ROOT_DIR}/mlip-3"
SAVE_DIR="${ROOT_DIR}/debug_mlip3"
LAMMPS_YAML="${ROOT_DIR}/examples/local/mtp_example/dump.si-300-1.yaml"
LAMMPS_THERMO="${ROOT_DIR}/examples/local/mtp_example/thermo_log.yaml"

python crystal_diffusion/train_mtp.py \
    --lammps_yaml $LAMMPS_YAML \
    --lammps_thermo $LAMMPS_THERMO \
    --mlip_dir $MLIP_PATH \
    --output_dir $SAVE_DIR
