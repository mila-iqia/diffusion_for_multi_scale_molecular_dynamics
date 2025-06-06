# Active learning with FLARE for for a Si vacancy

The work in this folder shows how to perform active learning with the following components:
- LAMMPS
- FLARE
- ARTn

It is assumed that these various components are available and compatible.

## Step 1

Compute the reference ARTn saddle energy by going to folder `Si-vac_sw_potential`.

Adapt the `run.sh` script to point to the correct LAMMPS executable, and run. 

This should produce an ARTn trajectory which identifies the "ground truth" saddle energy.

# Step 2

Adapt the script `launch_active_learning.py` to specify the path to the needed source directories.

Execute the script: this should create a sequence of active learning "campaigns", each
conducted in a number of "rounds".  


# Step 3

Execute the script `plot_barrier`. This creates a figure which compares the "ground truth"
saddle energy to what is obtained with active learning.