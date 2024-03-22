#!/bin/bash

TEMPERATURE=300
BOX_SIZE=4
STEP=1000
CROP=100
NTRAIN_RUN=10
NVALID_RUN=5

NRUN=$(($NTRAIN_RUN + $NVALID_RUN))

for SEED in {1..$NRUN}
do
  if [ "$SEED" -le $NTRAIN_RUN ]; then
    MODE="train"
  else
    MODE="valid"
  fi
  mkdir -p "${MODE}_run_${SEED}"
  cd "${MODE}_run_${SEED}"
  lmp < ../in.si.lammps -v STEP $(($STEP + $CROP)) -v T $TEMPERATURE -v S $BOX_SIZE

  # extract the thermodynamic outputs in a yaml file
  egrep  '^(keywords:|data:$|---$|\.\.\.$|  - \[)' log.lammps > thermo_log.yaml

  mkdir -p "uncropped_outputs"
  mv dump.si-${T}-${S}.yaml uncropped_outputs/
  mv thermo_log.yaml uncropped_outputs/

  python ../../data/crop_lammps_outputs.py \
      --lammps_yaml uncropped_outputs/dump.si-${T}-${S}.yaml \
      --lammps_thermo uncropped_outputs/thermo_log.yaml \
      --crop $CROP \
      --output_dir ./

  cd ..
done