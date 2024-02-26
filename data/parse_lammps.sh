#!/bin/bash

EXP_DIR="lammps_scripts/Si/si-custom/"
DUMP_FILENAME="dump.si-300-1.yaml"
THERMO_FILENAME="thermo_log.yaml"
OUTPUT_NAME="demo.parquet"

python crystal_diffusion/data/parse_lammps_outputs.py \
    --dump_file  ${EXP_DIR}/${DUMP_FILENAME} \
    --thermo_file ${EXP_DIR}/${THERMO_FILENAME} \
    --output_name ${EXP_DIR}/${OUTPUT_NAME}
