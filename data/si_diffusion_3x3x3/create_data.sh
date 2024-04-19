#!/bin/bash

TEMPERATURE=300
BOX_SIZE=3
MAX_ATOM=216
STEP=10000
CROP=10000
NTRAIN_RUN=10
NVALID_RUN=5

NRUN=$(($NTRAIN_RUN + $NVALID_RUN))

# Generate the data
for SEED in $(seq 1 $NRUN); do
  if [ "$SEED" -le $NTRAIN_RUN ]; then
    MODE="train"
  else
    MODE="valid"
  fi
  echo "Creating LAMMPS data for $MODE_run_$SEED..."
  mkdir -p "${MODE}_run_${SEED}"
  cd "${MODE}_run_${SEED}"
  lmp  -echo none -screen none < ../in.si.lammps -v STEP $(($STEP + $CROP)) -v T $TEMPERATURE -v S $BOX_SIZE -v SEED $SEED

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
