#!/bin/bash
#================================================================================
#
# This script drives multiple runs of active learning, each being constituted
# of multiple campaigns. The goal of doing "the same thing" so many times is to
# extract statistics on this stochastic process.
#
# This script should be adapted to use the correct machine-dependent paths.
#
#================================================================================

SRC_DIR=/Users/brunorousseau/sources/
LAMMPS_EXEC=${SRC_DIR}/lammps/build/lmp
ARTN_PLUGIN=${SRC_DIR}/artn-plugin/build/libartn.dylib
INITIAL_CHECKPOINT=./output/initial_trained_flare.json

list_runs="1 2 3 4 5"

# NoOp
CONFIG=noop_config.yaml 
TOP_DIR=./noop/


# Excise and NoOp
#CONFIG=excise_and_noop_config.yaml 
#TOP_DIR=./excise_and_noop/

# Excise and Random
#CONFIG=excise_and_random_config.yaml 
#TOP_DIR=./excise_and_random/

mkdir  $TOP_DIR
# This creates the initial checkpoint
python create_initial_checkpoint.py --path_to_lammps_executable $LAMMPS_EXEC \
                                    --path_to_initial_flare_checkpoint $INITIAL_CHECKPOINT


for run in $list_runs; do
	OUTPUT_DIR=${TOP_DIR}/run${run}
	train_active_learning --config $CONFIG \
			      --path_to_reference_directory ./reference \
			      --path_to_lammps_executable $LAMMPS_EXEC \
			      --path_to_artn_library_plugin $ARTN_PLUGIN \
			      --path_to_initial_flare_checkpoint $INITIAL_CHECKPOINT \
			      --output_directory $OUTPUT_DIR
done
