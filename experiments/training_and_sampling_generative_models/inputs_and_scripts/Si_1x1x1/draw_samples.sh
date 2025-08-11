#!/bin/bash

## optional key to track gpu usage
#SBATCH --wckey=courtois2024_extension
#SBATCH --partition=staff-amlrt
#SBATCH --job-name=sample_si_egnn_1x1x1
#SBATCH --output=logs/%x__%j.out
#SBATCH --error=logs/%x__%j.err
## cpu / gpu / memory selection
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00

module purge
module load python/3.10
module load cuda/12.3.2

REPO_ROOT=$HOME/repositories/diffusion_for_multi_scale_molecular_dynamics/

## Activate your venv
source $REPO_ROOT/.venv/bin/activate

# Make sure Huggingface's Datasets library doesn't try to connect to the internet.
# see : https://huggingface.co/docs/datasets/loading#offline
export HF_DATASETS_OFFLINE=1

REPO_DIR=$REPO_ROOT/src/diffusion_for_multi_scale_molecular_dynamics

EXPERIMENT_DIR=$SCRATCH/experiments/july26_si_egnn_1x1x1/

TRAINING_DIR=$EXPERIMENT_DIR/run1/
CHECKPOINT=$TRAINING_DIR/output/best_model/best_model-epoch\=039-step\=125040.ckpt

SAMPLING_DIR=$EXPERIMENT_DIR/samples_from_run1/

CONFIG1=$SAMPLING_DIR/config_sample_T\=1000.yaml
CONFIG2=$SAMPLING_DIR/config_sample_T\=10000.yaml


OUTPUT1=$SAMPLING_DIR/sample_output_T\=1000
OUTPUT2=$SAMPLING_DIR/sample_output_T\=10000

## work in scratch for easy access
## train_diffusion path assumes we are launching from the examples/narval/diffusion folder
srun python $REPO_DIR/sample_diffusion.py --config $CONFIG1 \
                                          --checkpoint $CHECKPOINT \
                                          --output $OUTPUT1 > script_output1.txt 2>&1

srun python $REPO_DIR/sample_diffusion.py --config $CONFIG2 \
                                          --checkpoint $CHECKPOINT \
                                          --output $OUTPUT2 > script_output2.txt 2>&1
