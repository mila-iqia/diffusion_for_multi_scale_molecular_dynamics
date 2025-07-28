#!/bin/bash

## optional key to track gpu usage
#SBATCH --wckey=courtois2024_extension
#SBATCH --partition=staff-amlrt
#SBATCH --job-name=sige_egnn_2x2x2
#SBATCH --output=logs/%x__%j.out
#SBATCH --error=logs/%x__%j.err
## cpu / gpu / memory selection
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

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

DATA_DIR=$SCRATCH/data/SiGe_diffusion_2x2x2
PROCESSED_DATA=$DATA_DIR/processed
DATA_WORK_DIR=$DATA_DIR/cache

EXPERIMENT_DIR=$SCRATCH/experiments/july26_sige_egnn_2x2x2/run1

CONFIG=$EXPERIMENT_DIR/config_diffusion_egnn.yaml
OUTPUT=$EXPERIMENT_DIR/output

## work in scratch for easy access

srun python $REPO_DIR/train_diffusion.py --config $CONFIG \
                                         --data $DATA_DIR \
                                         --processed_datadir $PROCESSED_DATA \
                                         --dataset_working_dir $DATA_WORK_DIR \
                                         --output $OUTPUT > script_output.txt 2>&1

