#!/bin/bash

function create_data_function() {
    # this function drives the creation training and validation data with LAMMPS.
    # It assumes :
    #   - the function is sourced in a bash script (the "calling script") within the folder where the data is to be created.
    #   - the calling script is invoked in a shell with the correct python environment.
    #   - the LAMMPS input file follows a template and has all the passed variables defined.
    #   - the paths are defined with respect to the folder where the generation script is called.

    TEMPERATURE="$1"
    BOX_SIZE="$2"
    STEP="$3"
    CROP="$4"
    NTRAIN_RUN="$5"
    NVALID_RUN="$6"
    SW_PATH="$7"
    IN_PATH="$8"
    CONFIG_PATH="$9"

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

      # Calling LAMMPS with various arguments to keep it quiet. Also, the current location is "${MODE}_run_${SEED}", which is one
      # folder away from the location of the calling script.
      lmp  -echo none -screen none < ../$IN_PATH -v STEP $(($STEP + $CROP)) -v T $TEMPERATURE -v S $BOX_SIZE -v SEED $SEED -v SW_PATH ../$SW_PATH

      # extract the thermodynamic outputs in a yaml file
      egrep  '^(keywords:|data:$|---$|\.\.\.$|  - \[)' log.lammps > thermo_log.yaml

      mkdir -p "uncropped_outputs"
      mv "dump.${TEMPERATURE}-${BOX_SIZE}.yaml" uncropped_outputs/
      mv thermo_log.yaml uncropped_outputs/

      python ../../crop_lammps_outputs.py \
          --lammps_yaml "uncropped_outputs/dump.${TEMPERATURE}-${BOX_SIZE}.yaml" \
          --lammps_thermo "uncropped_outputs/thermo_log.yaml" \
          --crop $CROP \
          --output_dir ./

      cd ..
    done

    # process the data
    python ../process_lammps_data.py --data "./" --processed_datadir "./processed/" --config ${CONFIG_PATH}
}
