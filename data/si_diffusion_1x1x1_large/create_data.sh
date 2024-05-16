#!/bin/bash

TEMPERATURE=300
BOX_SIZE=1
MAX_ATOM=8
STEP=10000
CROP=10000
NTRAIN_RUN=10
NVALID_RUN=5
NTRAIN_RUN_EXTRA=40

NRUN=$(($NTRAIN_RUN + $NVALID_RUN + $NTRAIN_RUN_EXTRA))

# Generate the data
for SEED in $(seq 1 $NRUN); do
  if [ "$SEED" -le $NTRAIN_RUN ]; then
    MODE="train"
  elif  [ "$SEED" -le $(($NTRAIN_RUN + $NVALID_RUN)) ]; then
    MODE="valid"
  else
    MODE="train"
  fi
  echo "Creating LAMMPS data for ${MODE}_run_${SEED}..."
  mkdir -p "${MODE}_run_${SEED}"
  cd "${MODE}_run_${SEED}"
  lmp  -echo none -screen none < ../in.si.lammps -v STEP $(($STEP + $CROP)) -v T $TEMPERATURE -v S $BOX_SIZE -v SEED $SEED

  # extract the thermodynamic outputs in a yaml file
  egrep  '^(keywords:|data:$|---$|\.\.\.$|  - \[)' log.lammps > thermo_log.yaml

  mkdir -p "uncropped_outputs"
  mv "dump.si-${TEMPERATURE}-${BOX_SIZE}.yaml" uncropped_outputs/
  mv thermo_log.yaml uncropped_outputs/

  FILE_LENGTH=$((20 * $STEP))
  tail -n $FILE_LENGTH "uncropped_outputs/dump.si-${TEMPERATURE}-${BOX_SIZE}.yaml" > "lammps_dump.yaml"
  { sed -n '2,3p' uncropped_outputs/thermo_log.yaml; tail -n $(($STEP + 1)) uncropped_outputs/thermo_log.yaml |
  head -n $STEP;  } > lammps_thermo.yaml
  cd ..
done

# process the data
python ../process_lammps_data.py --data "./" --processed_datadir "./processed/" --max_atom ${MAX_ATOM}
