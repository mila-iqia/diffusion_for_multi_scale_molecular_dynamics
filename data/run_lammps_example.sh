#!/bin/bash

TEMPERATURE=300
BOX_SIZE=1

lmp < data/lammps_input_example -v STEP 10 -v T $TEMPERATURE -v S $BOX_SIZE