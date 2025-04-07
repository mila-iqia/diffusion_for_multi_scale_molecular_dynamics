#!/bin/bash

export OMP_PATH="/opt/homebrew/opt/libomp/include/"
export PYTORCH_ENABLE_MPS_FALLBACK=1

OUTPUT=./output/run1

BASE_CONFIG=../base_config.yaml
SPECIFIC_CONFIG=specific_config.yaml
CONFIG=config.yaml

# Build the config
cp $BASE_CONFIG $CONFIG
cat $SPECIFIC_CONFIG >> $CONFIG

train_diffusion --accelerator "cpu" --config $CONFIG --output $OUTPUT 
