#!/bin/bash

export OMP_PATH="/opt/homebrew/opt/libomp/include/"
export PYTORCH_ENABLE_MPS_FALLBACK=1

CONFIG=config.yaml
DATA_DIR=./
PROCESSED_DATA=${DATA_DIR}
DATA_WORK_DIR=${DATA_DIR}

OUTPUT=./output/run1

python ../pseudo_train_diffusion.py \
    --accelerator "cpu" \
    --config $CONFIG \
    --data $DATA_DIR \
    --processed_datadir $PROCESSED_DATA \
    --dataset_working_dir $DATA_WORK_DIR \
    --output $OUTPUT # > log.txt 2>&1 
