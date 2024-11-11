#!/bin/bash

source ../data_generation_functions.sh

TEMPERATURE=300
BOX_SIZE=1
MAX_ATOM=8
STEP=10000
CROP=10000
NTRAIN_RUN=10
NVALID_RUN=5

SW_PATH="../stillinger_weber_coefficients/Si.sw"
IN_PATH="in.Si.lammps"

create_data_function $TEMPERATURE $BOX_SIZE $MAX_ATOM $STEP $CROP $NTRAIN_RUN $NVALID_RUN $SW_PATH $IN_PATH
