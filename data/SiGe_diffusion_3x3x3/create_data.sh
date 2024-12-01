#!/bin/bash

source ../data_generation_functions.sh

TEMPERATURE=300
BOX_SIZE=3
STEP=10000
CROP=10000
NTRAIN_RUN=10
NVALID_RUN=5

SW_PATH="../stillinger_weber_coefficients/SiGe.sw"
IN_PATH="in.SiGe.lammps"
CONFIG_PATH="config.yaml"

create_data_function $TEMPERATURE $BOX_SIZE $STEP $CROP $NTRAIN_RUN $NVALID_RUN $SW_PATH $IN_PATH $CONFIG_PATH
