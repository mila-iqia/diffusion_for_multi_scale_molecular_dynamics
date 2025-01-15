#!/bin/bash

export OMP_PATH="/opt/homebrew/opt/libomp/include/"
export PYTORCH_ENABLE_MPS_FALLBACK=1


CONFIG=config.yaml

OUTPUT=./output/run1

SRC=/Users/brunorousseau/PycharmProjects/diffusion_for_multi_scale_molecular_dynamics/src/diffusion_for_multi_scale_molecular_dynamics


python ${SRC}/train_diffusion.py --accelerator "cpu" --config $CONFIG --output $OUTPUT 
