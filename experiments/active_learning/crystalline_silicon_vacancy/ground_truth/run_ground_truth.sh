#!/bin/bash
## optional key to track gpu usage
#SBATCH --wckey=courtois2025
#SBATCH --partition=staff-amlrt
#SBATCH --job-name=active_learning_si_vacancy
#SBATCH --output=logs/%x__%j.out
#SBATCH --error=logs/%x__%j.err
## cpu / gpu / memory selection
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00

module load openmpi
module load cuda/12.6.0  # not useful for now
module load python/3.10
module load gcc  


REPO_ROOT=$HOME/repositories/diffusion_for_multi_scale_molecular_dynamics/
POTENTIAL_DIRECTORY=${REPO_ROOT}/data/stillinger_weber_coefficients/


SRC_DIR=$HOME/sources
LAMMPS_EXEC=${SRC_DIR}/lammps/build/lmp


GROUND_TRUTH_DIRECTORY=$SCRATCH/experiments/active_learning/crystalline_silicon_vacancy/ground_truth
TOP_OUTPUT_DIR=${GROUND_TRUTH_DIRECTORY}/output/

REFERENCE_DIRECTORY=$SCRATCH/experiments/active_learning/crystalline_silicon_vacancy/reference

## Activate your venv
source $REPO_ROOT/.venv/bin/activate


mkdir ${TOP_OUTPUT_DIR}
cd ${TOP_OUTPUT_DIR}

for run in {1..10}; do
    RUN_DIR=${TOP_OUTPUT_DIR}/run"$run"

    mkdir ${RUN_DIR}

    cp ${GROUND_TRUTH_DIRECTORY}/lammps.in $RUN_DIR

    cd $RUN_DIR

    ln -s ${POTENTIAL_DIRECTORY}/Si.sw Si.sw
    ln -s ${REFERENCE_DIRECTORY}/artn.in artn.in
    ln -s ${REFERENCE_DIRECTORY}/initial_configuration.dat initial_configuration.dat

    mpirun -np 4 $LAMMPS_EXEC < lammps.in
    cd ${TOP_OUTPUT_DIR}
done
