#!/bin/bash

# This example assumes that the dataset 'Si_diffusion_1x1x1' is present locally in the DATA folder.

CONFIG=../../config_files/diffusion/config_diffusion_mlp.yaml
DATA_DIR=../../../data/Si_diffusion_1x1x1
PROCESSED_DATA=${DATA_DIR}/processed
DATA_WORK_DIR=${DATA_DIR}/cache/

OUTPUT=output/run1

python ../../../src/diffusion_for_multi_scale_molecular_dynamics/train_diffusion.py \
    --accelerator "cpu" \
    --config $CONFIG \
    --data $DATA_DIR \
    --processed_datadir $PROCESSED_DATA \
    --dataset_working_dir $DATA_WORK_DIR \
    --output $OUTPUT #> log.txt 2>&1 
