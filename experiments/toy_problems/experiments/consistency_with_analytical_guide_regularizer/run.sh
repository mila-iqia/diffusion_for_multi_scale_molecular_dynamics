#!/bin/bash

export OMP_PATH="/opt/homebrew/opt/libomp/include/"
export PYTORCH_ENABLE_MPS_FALLBACK=1

CONFIG=config.yaml
OUTPUT=./output/run1

train_diffusion --accelerator "cpu" --config $CONFIG --output $OUTPUT 
