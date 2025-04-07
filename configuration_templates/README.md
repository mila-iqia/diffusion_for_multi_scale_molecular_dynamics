# Configuration templates

This directory contains example configuration files and scripts.
These are meant to be useful examples for different use cases. They 
should be modified to fit your needs.

Beware that some examples might not be quite right: this repository 
is a work in progress and things change quickly. 

### diffusion_config_files
Various yaml configuration files to train diffusion models. File names indicates
which score network is used, and whether it is intended for Orion.

### Orion_config_files

The preferred way of conducting hyperparameter tuning is with [Orion](https://orion.readthedocs.io/en/stable/).
This folder contains configuration files for this tool.

### run_scripts

These are convience bash script to launch jobs on different computing environment,
such as "local" (a laptop), "mila cluster" (the mila cluster) and "narval" (the Compute Canada cluster, Narval).

### active_learning_config_files
WIP. 


### mtp
Examples if you are using MTP. This is deprecated, and not well maintained...