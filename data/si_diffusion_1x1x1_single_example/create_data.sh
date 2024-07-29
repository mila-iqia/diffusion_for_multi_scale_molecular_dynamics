#!/bin/bash
#================================================================================
# This script creates a 'fake' dataset composed of a single example repeated
# multiple times. 
#================================================================================

TEMPERATURE=0
BOX_SIZE=1
MAX_ATOM=8
STEP=4048
CROP=1 # Crop 1 to make sure there is exactly 4048 examples in the final dataset.
NTRAIN_RUN=1
NVALID_RUN=1

NRUN=$(($NTRAIN_RUN + $NVALID_RUN))

# Generate the data
for SEED in $(seq 1 $NRUN); do
  if [ "$SEED" -le $NTRAIN_RUN ]; then
    MODE="train"
  else
    MODE="valid"
  fi
  echo "Creating LAMMPS data for ${MODE}_run_${SEED}..."
  mkdir -p "${MODE}_run_${SEED}"
  cd "${MODE}_run_${SEED}"
  lmp  -echo none -screen none < ../in.si.lammps -v STEP $STEP -v S $BOX_SIZE -v T $TEMPERATURE

  # extract the thermodynamic outputs in a yaml file
  egrep  '^(keywords:|data:$|---$|\.\.\.$|  - \[)' log.lammps > thermo_log.yaml

  mkdir -p "uncropped_outputs"
  mv "dump.si-${TEMPERATURE}-${BOX_SIZE}.yaml" uncropped_outputs/
  mv thermo_log.yaml uncropped_outputs/

  python ../../crop_lammps_outputs.py \
      --lammps_yaml "uncropped_outputs/dump.si-${TEMPERATURE}-${BOX_SIZE}.yaml" \
      --lammps_thermo "uncropped_outputs/thermo_log.yaml" \
      --crop $CROP \
      --output_dir ./

  cd ..
done

# process the data
python ../process_lammps_data.py --data "./" --processed_datadir "./processed/" --max_atom ${MAX_ATOM}
