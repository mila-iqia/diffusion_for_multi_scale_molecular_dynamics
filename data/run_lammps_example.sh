#!/bin/bash

TEMPERATURE=300
BOX_SIZE=1

<<<<<<< HEAD
lmp < lammps_input_example.lammps -v STEP 10 -v T $TEMPERATURE -v S $BOX_SIZE

# extract the thermodynamic outputs in a yaml file
egrep  '^(keywords:|data:$|---$|\.\.\.$|  - \[)' log.lammps > log.yaml
