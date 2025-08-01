#!/bin/bash

## optional key to track gpu usage
#SBATCH --wckey=courtois2024_extension
#SBATCH --partition=staff-amlrt
#SBATCH --job-name=constrained_sample_si_egnn_3x3x3
#SBATCH --output=logs/%x__%j.out
#SBATCH --error=logs/%x__%j.err
## cpu / gpu / memory selection
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00

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

TOP_EXPERIMENT_DIR=$SCRATCH/experiments/july26_si_egnn_3x3x3/
CHECKPOINT=$TOP_EXPERIMENT_DIR/run1/output/best_model/best_model-epoch\=004-step\=015630.ckpt

SAMPLING_DIR=$TOP_EXPERIMENT_DIR/constrained_samples_from_run1

CONFIG_LINEAR=$SAMPLING_DIR/config_samples_linear_schedule.yaml
CONFIG_EXP=$SAMPLING_DIR/config_samples_exponential_schedule.yaml

list_T_LINEAR="10 20 30 40 50 60 70 80 90"
list_T_EXP="610 620 630 640 650 660 670 680 690"


for STARTING_T in $list_T_LINEAR; do
    SAMPLING_CONSTRAINT_PATH=${SAMPLING_DIR}/sampling_constraints_Si_diffusion_3x3x3/linear/noise_composition_linear_schedule_start_T_${STARTING_T}.pickle
    OUTPUT=$SAMPLING_DIR/output/linear/output_T=${STARTING_T}

    srun python $REPO_DIR/sample_diffusion.py --config $CONFIG_LINEAR \
                                              --checkpoint $CHECKPOINT \
                                              --path_to_starting_configuration_data_pickle $SAMPLING_CONSTRAINT_PATH \
                                              --output $OUTPUT > script_linear_output_T=${STARTING_T}.txt 2>&1

done

for STARTING_T in $list_T_EXP; do
    SAMPLING_CONSTRAINT_PATH=$SAMPLING_DIR/sampling_constraints_Si_diffusion_3x3x3/exponential/noise_composition_exponential_schedule_start_T_${STARTING_T}.pickle
    OUTPUT=$SAMPLING_DIR/output/exponential/output_T=${STARTING_T}
    

    srun python $REPO_DIR/sample_diffusion.py --config $CONFIG_EXP \
                                              --checkpoint $CHECKPOINT \
                                              --path_to_starting_configuration_data_pickle $SAMPLING_CONSTRAINT_PATH \
                                              --output $OUTPUT > script_exp_output_T=${STARTING_T}.txt 2>&1

done
