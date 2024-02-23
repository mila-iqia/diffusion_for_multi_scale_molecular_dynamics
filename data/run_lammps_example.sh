#!/bin/bash

TEMPERATURE=300
BOX_SIZE=1

lmp < lammps_input_example.lammps -v STEP 10 -v T $TEMPERATURE -v S $BOX_SIZE