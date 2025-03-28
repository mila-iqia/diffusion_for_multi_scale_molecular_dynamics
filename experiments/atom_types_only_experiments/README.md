# Atom types-only experiments

This folder contains code to execute a simple experiment where  the lattice parameters and the 
relative coordinates are held fixed and only the atom types diffuse.

The system considered is Si-Ge 1x1x1 (8 atoms in total) where the atoms are
on the crystalline lattice. 

This is a "sanity check" experiment. It is meant to show that the discrete atom type
diffusion works properly in the simplest case where everything else is held fixed. 
The training process will record sample trajectories at the last epoch of training: 
after postprocessing, these trajectories can be visualized to confirm that the diffusion
is successful.

Since this experiment is in some sense "artificial", we choose to implement some 
of the needed classes and methods in this folder (and not in the src/ folder) in order to 
avoid complicating the main code base for what is a marginal use case. Correspondingly,
we use "mock.patch" to inject the needed code where appropriate to turn off 
diffusion on relative coordinates and create a dataset of pure crystalline SiGe 1x1x1.


## Directory Content

Here we document the content of this directory.

### `pseudo_train_diffusion.py`

This is the training entry point. Is is a "patched" version of the `train_diffusion.py` script,
where we substitute production code with what is required to build this artificial diffusion problem.

### patches/
This is where various classes and methods are implemented to patch the main code and "fudge in" 
the artificial use case. We deliberately choose to avoid having this complexity in the main code base
(the `src/` folder).

### experiments/
Here we define a configuration file and a driving bash script to execute the atom type-only diffusion.
Start here to run this experiment.

>  bash run_diffusion.sh

should launch a training run. You can keep track of training with tensorboard. Execution should complete
in a few minutes at most.

### analysis/

This folder contains scripts to analyse the results of the experiment. After the experiment has completed,
come here and execute the scripts to visualize results.

* `plot_atom_type_probabilities.py` will generate matplotlib plots of how the atom type probabilities evolve
  along a sampling trajectory.
* `create_trajectory_cif_files.py` will create cif files for each sampled trajectories. These can then be visualized
with a cif viewer (like Ovito for example) to see what a diffusion trajectory looks like.
