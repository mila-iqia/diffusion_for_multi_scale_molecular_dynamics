#!/bin/bash

# This example assumes that the dataset 'si_diffusion_small' is present locally in the DATA folder.
# It is also assumed that the user has a Comet account for logging experiments.

CONFIG=../../config_files/diffusion/config_diffusion_mace.yaml
DATA_DIR=../../../data/si_diffusion_1x1x1
PROCESSED_DATA=${DATA_DIR}/processed
DATA_WORK_DIR=./tmp_work_dir/
OUTPUT=output

python ../../../crystal_diffusion/train_diffusion.py \
    --config $CONFIG \
    --data $DATA_DIR \
    --processed_datadir $PROCESSED_DATA \
    --dataset_working_dir $DATA_WORK_DIR \
    --output $OUTPUT
