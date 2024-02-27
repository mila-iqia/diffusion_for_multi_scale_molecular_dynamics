# MTP README

## How to setup:

We use the maml (MAterials Machine Learning) library for the MTP implementation.

pip install maml

To make MTP works, we also need the mlp library
https://mlip.skoltech.ru

## Instructions to install on a Mac

First, install open-mpi with brew.
Beware that open-mpi uses Mac gcc compiler, not the brew gcc, which is not linked to a gfortran compiler.
Set the path properly.

brew install gcc  
export HOMEBREW_CC=gcc-13  # gcc-13 is brew gcc: /usr/local/bin/gcc-13
export HOMEBREW_CXX=g++-13  # brew g++
brew install openmpi --build-from-source


If openmpi is already installed, but doesn't work because of the gcc / gfortran unlinkable, you can reinstall

export HOMEBREW_CC=gcc-13  # gcc-13 is brew gcc: /usr/local/bin/gcc-13
export HOMEBREW_CXX=g++-13  # brew g++
brew reinstall openmpi --build-from-source

Then, you can install MLIP

git clone https://gitlab.com/ashapeev/mlip-2.git
cd mlip-2
./configure
make mlip

If the installation is successful, the command is mlip-2/bin/mlp
This should be recognized as mlp, but it was not in my case.
This creates issues with maml that was calling mlp with python subprocess.Popen

The fix was to modify the maml code (change the root path to fix your environment)
In
/Users/simonb/miniconda/envs/crystal_diffusion/lib/python3.11/site-packages/maml/apps/pes/_mtp.py

You want to change the "mlp" variable passed in subprocess.Popen by the full path of the MLIP-2 mlp command.
e.g. at line 597
with open("min_dist", "w") as f, subprocess.Popen(["mlp", "mindist", atoms_filename], stdout=f) as p:

was replaced with:

mlp_command = '/Users/simonb/ic-collab/courtois_collab/crystal_diffusion/mlip-2/bin/mlp'
with open("min_dist", "w") as f, subprocess.Popen([mlp_command, "mindist", atoms_filename], stdout=f) as p:

same at line 615, 722, 724

Also, comment out the block starting with 
if not which("mlp"):
at lines 574 and 696.