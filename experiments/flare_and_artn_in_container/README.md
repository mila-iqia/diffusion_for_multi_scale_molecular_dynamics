# FLARE and ARTn in a Podman Container

FLARE, ARTn and LAMMPS can be compiled together in a Podman container using 

    resources/flare/Containerfile_FLARE_and_ARTn

The various scripts in this folder were located in a mounted directory in a running instance
in order to explore the use of FLARE in active learning. 

The mounted folder was `/home/user/experiments/`.

At this time, FLARE no longer seems like a promising avenue. It appears that the
sparse Gaussian approximations used in that code base lead to uncontrollable 
uncertainty / error relationships. 

These various files are kept for completeness, but are not meant to 
really be used at this stage. You can look at them for inspiration, but this isn't 
a runnable experiment.


# Content

 - `artn/` contains inputs to run a ARTn experiment within LAMMPS using a mapped FLARE model.
    This is derived from the artn tutorial and is for inspiration only. 
-  `flare_experiment/` contains ad hoc code that was created within the Podman container to drive
    some exploration. 