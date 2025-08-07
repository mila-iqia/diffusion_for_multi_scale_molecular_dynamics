#!/bin/bash
## optional key to track gpu usage
#SBATCH --wckey=courtois2025
#SBATCH --partition=staff-amlrt
#SBATCH --job-name=active_learning_si_vacancy_excise_and_noop
#SBATCH --output=logs/%x__%j.out
#SBATCH --error=logs/%x__%j.err
## cpu / gpu / memory selection
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

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


REPO_ROOT=$HOME/repositories/diffusion_for_multi_scale_molecular_dynamics/

SRC_DIR=$HOME/sources
LAMMPS_EXEC=${SRC_DIR}/lammps/build/lmp
ARTN_PLUGIN=${SRC_DIR}/artn-plugin/build/libartn.so

INITIAL_FLARE_CHECKPOINT=$SCRATCH/experiments/active_learning/amorphous_silicon/pretrained_flare/flare_model_pretrained.json

CONFIG=excise_and_noop_config.yaml  

TOP_DIR=./output/

REFERENCE_DIRECTORY=$SCRATCH/experiments/active_learning/amorphous_silicon/reference

## Activate your venv
source $REPO_ROOT/.venv/bin/activate

mkdir $TOP_DIR

list_runs="1 2 3 4 5"
for run in $list_runs; do
        echo "Creating run $run of active learning."
	OUTPUT_DIR=${TOP_DIR}/run${run}
	train_active_learning --config $CONFIG \
			                  --path_to_reference_directory $REFERENCE_DIRECTORY \
			                  --path_to_lammps_executable $LAMMPS_EXEC \
			                  --path_to_artn_library_plugin $ARTN_PLUGIN \
			                  --path_to_initial_flare_checkpoint $INITIAL_FLARE_CHECKPOINT \
			                  --output_directory $OUTPUT_DIR
done
