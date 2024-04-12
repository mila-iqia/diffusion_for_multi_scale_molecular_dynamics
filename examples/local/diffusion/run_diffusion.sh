#!/bin/bash

CONFIG=config_diffusion.yaml
DATA_DIR=../../data/si_diffusion_v1
PROCESSED_DATA=${DATA_DIR}/processed
DATA_WORK_DIR=./tmp_work_dir/
OUTPUT=debug

python ../../crystal_diffusion/train_diffusion.py \
    --config $CONFIG \
    --data $DATA_DIR \
    --processed_datadir $PROCESSED_DATA \
    --dataset_working_dir $DATA_WORK_DIR \
    --output $OUTPUT
