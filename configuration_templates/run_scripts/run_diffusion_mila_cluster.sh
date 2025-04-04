!/bin/bash

## optional key to track gpu usage
#SBATCH --wckey=courtois2024_phase1
#SBATCH --partition=staff-amlrt
#SBATCH --job-name=crystal_diffusion
#SBATCH --output=logs/%x__%j.out
#SBATCH --error=logs/%x__%j.err
## cpu / gpu / memory selection
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00

module purge
module load python/3.10
module load cuda/12.3.2

## Activate your venv
source $HOME/env_courtois/bin/activate


# Make sure Huggingface's Datasets library doesn't try to connect to the internet.
# see : https://huggingface.co/docs/datasets/loading#offline
export HF_DATASETS_OFFLINE=1

EXPERIMENT_DIR=$HOME/courtois2024/diffusion_for_multi_scale_molecular_dynamics/experiments/mace_diffusion_june24/run1/

DATA_DIR=$HOME/courtois2024/diffusion_for_multi_scale_molecular_dynamics/data/si_diffusion_1x1x1
PROCESSED_DATA=$DATA_DIR/processed

REPO_DIR=$HOME/courtois2024/diffusion_for_multi_scale_molecular_dynamics/crystal_diffusion

DATA_WORK_DIR=$EXPERIMENT_DIR/tmp_work_dir

CONFIG=$EXPERIMENT_DIR/config_mace_equivariant_head.yaml

## work in scratch for easy access
OUTPUT=$EXPERIMENT_DIR/output

## train_diffusion path assumes we are launching from the examples/narval/diffusion folder
srun python $REPO_DIR/train_diffusion.py --config $CONFIG \
                                         --data $DATA_DIR \
                                         --processed_datadir $PROCESSED_DATA \
                                         --dataset_working_dir $DATA_WORK_DIR \
                                         --output $OUTPUT > script_output.txt 2>&1

