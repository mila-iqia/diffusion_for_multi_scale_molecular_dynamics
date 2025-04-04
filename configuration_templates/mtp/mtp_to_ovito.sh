#!/bin/bash

MTP_PREDICTION=./experiments/mtp_find_region/predictions.csv
LAMMPS_OUTPUT=./experiments/mtp_find_region/dump.si-10000-4.yaml
OVITO_OUTPUT=./test_si_structure_ovito.xyz

python crystal_diffusion/analysis/ovito_visualisation.py \
    --prediction_file $MTP_PREDICTION \
    --lammps_output $LAMMPS_OUTPUT \
    --output_name $OVITO_OUTPUT