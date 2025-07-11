#!/bin/bash
## optional key to track gpu usage
#SBATCH --wckey=courtois2025
#SBATCH --partition=staff-amlrt
#SBATCH --job-name=active_learning_amorphous_si
#SBATCH --output=logs/%x__%j.out
#SBATCH --error=logs/%x__%j.err
## cpu / gpu / memory selection
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=8:00:00

#================================================================================
#
# This script drives multiple runs of active learning, each being constituted
# of multiple campaigns. The goal of doing "the same thing" so many times is to
# extract statistics on this stochastic process.
#
# This script should be adapted to use the correct machine-dependent paths.
#
#================================================================================
module load openmpi
module load cuda/12.6.0  # not useful for now
module load python/3.10
module load gcc  

SRC_DIR=/home/mila/r/rousseab/sources
LAMMPS_EXEC=${SRC_DIR}/lammps/build/lmp
ARTN_PLUGIN=${SRC_DIR}/artn-plugin/build/libartn.so
INITIAL_CHECKPOINT=./initial_model/flare_model_sigma_2.json

PYTHON=/home/mila/r/rousseab/repositories/diffusion_for_multi_scale_molecular_dynamics/.venv/bin/python


CONFIG=excise_and_repaint_config.yaml 
TOP_DIR=./excise_and_repaint/
DIFFUSION_CHECKPOINT=/home/mila/r/rousseab/courtois2024/models/march30_reference_egnn_si_2x2x2/run1/output/best_model/best_model-epoch=053-step=337554.ckpt


REPO_ROOT=$HOME/repositories/diffusion_for_multi_scale_molecular_dynamics/

## Activate your venv
source $REPO_ROOT/.venv/bin/activate

mkdir  $TOP_DIR

list_runs="1 2 3 4 5"
for run in $list_runs; do
        echo "Creating run $run of active learning."
	OUTPUT_DIR=${TOP_DIR}/run${run}
	train_active_learning --config $CONFIG \
                              --path_to_score_network_checkpoint $DIFFUSION_CHECKPOINT \
			      --path_to_reference_directory ./reference \
			      --path_to_lammps_executable $LAMMPS_EXEC \
			      --path_to_artn_library_plugin $ARTN_PLUGIN \
			      --path_to_initial_flare_checkpoint $INITIAL_CHECKPOINT \
			      --output_directory $OUTPUT_DIR
done
