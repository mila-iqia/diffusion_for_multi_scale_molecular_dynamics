# Experiments

This folder contains various experiments and analysis scripts.

## Active Learning Benchmark

TODO

## component analysis
This folder contains scripts to plot different components used in the diffusion model. This is a good
place to start to visualize the various ingredients that go into the construction of a model.

The various scripts will show plots on the matplotlib console and save figures in ./analysis/images/.

## tutorial
This folder contains a jupyter-notebook that presents a simplified experiment. This is an ideal
starting point to understand the various components in the code base and how they all work together.


## generators_sanity_check
The main entry point is the script
> sde_generator_sanity_check.py

which solves an SDE using for a simple situation and creates plots that show that the
SDE produces results that compare well to what is expected. This is a "sanity check" to
validate that the code works.


## atom_types_only_experiments

This folder contains an "atom-type only" diffusion experiment. This is a nice sanity check
that atom type diffusion works while everything else is held fixed. See the README in that folder
for more details.