#!/bin/bash

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
#SBATCH --time=01:00:00

## Get optional email alerts
##SBATCH --mail-type=all
##SBATCH --mail-user=first.last@mila.quebec

module load python/3.11.5
module load arrow
module load cuda
module load scipy-stack

## Activate your venv
source /home/blackbur/crystal_diffusion/bin/activate

export ORION_DB_ADDRESS='orion_db.pkl'
export ORION_DB_TYPE='pickleddb'

ROOT_DIR=../../../
CONFIG=config_diffusion.yaml
DATA_DIR=${ROOT_DIR}/data/si_diffusion_small
PROCESSED_DATA=${DATA_DIR}/processed
## work in scratch for easy access
OUTPUT_ROOT_DIR=/home/blackbur/scratch/test_run/
DATA_WORK_DIR=${OUTPUT_ROOT_DIR}/tmp_work_dir/

orion -v hunt --config orion_config.yaml \
    srun python ${ROOT_DIR}/crystal_diffusion/train_diffusion.py \
    --config $CONFIG \
    --data $DATA_DIR \
    --processed_datadir $PROCESSED_DATA \
    --dataset_working_dir $DATA_WORK_DIR \
    --output ${OUTPUT_ROOT_DIR}/'{trial.id}/'
