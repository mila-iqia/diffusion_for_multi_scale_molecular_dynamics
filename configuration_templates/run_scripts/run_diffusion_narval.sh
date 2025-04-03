#!/bin/bash

#====================================================================================
#  Example running script for the Narval cluster
#  ---------------------------------------------
#
#  This example assumes that:
#   1. The code has been installed with virutalenv in environment 'crystal_diffusion' 
#   2. The comet api key is located in ~/.comet.config 
#   3. The dataset si_diffusion_1x1x1 is available and located at $HOME/data/si_diffusion_1x1x1
#   4. The experiment dir is actually $HOME/experiments/example, ie this script and 
#      the corresponding config file has been moved outside the repo.
#      
#====================================================================================

## link to Normand Mousseau account
#SBATCH --account=def-mousseau
## optional key to track gpu usage
#SBATCH --wckey=crystal_diffusion_phase1
#SBATCH --job-name=crystal_diffusion
#SBATCH --output=logs/%x__%j.out
#SBATCH --error=logs/%x__%j.err
## cpu / gpu / memory selection
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00

module purge
module load httpproxy # necessary for Comet logging
module load python/3.11.5
module load arrow
module load cuda
module load scipy-stack

## Activate your venv
source $HOME/crystal_diffusion/bin/activate


# Make sure Huggingface's Datasets library doesn't try to connect to the internet.
# see : https://huggingface.co/docs/datasets/loading#offline
export HF_DATASETS_OFFLINE=1

EXPERIMENT_DIR=$HOME/experiments/example

DATA_DIR=$HOME/data/si_diffusion_1x1x1
PROCESSED_DATA=$DATA_DIR/processed

REPO_DIR=$HOME/diffusion_for_multi_scale_molecular_dynamics/crystal_diffusion

DATA_WORK_DIR=$EXPERIMENT_DIR/tmp_work_dir

CONFIG=$EXPERIMENT_DIR/config_diffusion.yaml

## work in scratch for easy access
OUTPUT=$EXPERIMENT_DIR/output

## train_diffusion path assumes we are launching from the examples/narval/diffusion folder
srun python $REPO_DIR/train_diffusion.py --config $CONFIG \
                                         --data $DATA_DIR \
                                         --processed_datadir $PROCESSED_DATA \
                                         --dataset_working_dir $DATA_WORK_DIR \
                                         --output $OUTPUT > script_output.txt 2>&1
