#!/bin/bash

TEMPERATURE=300
BOX_SIZE=1

<<<<<<< HEAD
lmp < lammps_input_example.lammps -v STEP 10 -v T $TEMPERATURE -v S $BOX_SIZE

# extract the thermodynamic outputs in a yaml file
egrep  '^(keywords:|data:$|---$|\.\.\.$|  - \[)' log.lammps > log.yaml
=======
lmp < lammps_input_example.lammps -v STEP 10 -v T $TEMPERATURE -v S $BOX_SIZE
>>>>>>> 5927511b163780d0be25d5db4a0eb8868b12f8b2
