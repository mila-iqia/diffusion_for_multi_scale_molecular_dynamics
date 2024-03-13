# MTP README

## How to setup:

We use the maml (MAterials Machine Learning) library for the MTP implementation.

```
pip install maml
```

To make MTP works, we also need the mlp library
https://gitlab.com/ashapeev/mlip-3

## Instructions to install on a Mac

First, install open-mpi with brew.
Beware that open-mpi uses Mac gcc compiler, not the brew gcc, which is not linked to a gfortran compiler.
Set the path properly.

```
brew install gcc  
export HOMEBREW_CC=gcc-13  # gcc-13 is brew gcc: /usr/local/bin/gcc-13
export HOMEBREW_CXX=g++-13  # brew g++
brew install openmpi --build-from-source
```

If openmpi is already installed, but doesn't work because of the gcc / gfortran unlinkable, you can reinstall

```
export HOMEBREW_CC=gcc-13  # gcc-13 is brew gcc: /usr/local/bin/gcc-13
export HOMEBREW_CXX=g++-13  # brew g++
brew reinstall openmpi --build-from-source
```

## Build instructions from the package

Then, you can install MLIP

```
git clone https://gitlab.com/ashapeev/mlip-3.git
```

The build instructions are supposed to be:

```
cd mlip-3
./configure
make mlip
```

## Corrected build instructions

But this doesn't work. Instead, we have to use cmake:

```
cd mlip-3
./configure
mkdir build              # create a build directory
cd build
cmake ..                 # configuration
```

But this doesn't work either, because the cmake instructions are missing from the repo.

```
cp mlip3_missing_files/CMakeLists.txt mlip-3/CMakeLists.txt
cp mlip3_missing_files/src_CMakeLists.txt mlip-3/src/CMakeLists.txt
cp mlip3_missing_files/test_CMakeLists.txt mlip-3/test/CMakeLists.txt
cp mlip3_missing_files/src_mlp_CMakeLists.txt mlip-3/src/mlp/CMakeLists.txt
cp mlip3_missing_files/src_common_CMakeLists.txt mlip-3/src/common/CMakeLists.txt
cp mlip3_missing_files/src_drivers_CMakeLists.txt mlip-3/src/drivers/CMakeLists.txt
cp mlip3_missing_files/src_external_CMakeLists.txt mlip-3/src/external/CMakeLists.txt
cp mlip3_missing_files/test_selftest_CMakeLists.txt mlip-3/test/self-test/CMakeLists.txt
cp mlip3_missing_files/test_examples_CMakeLists.txt mlip-3/test/examples/CMakeLists.txt
cp mlip3_missing_files/test_examples_00_CMakeLists.txt mlip-3/test/examples/00.convert_vasp_outcar/CMakeLists.txt
cp mlip3_missing_files/test_examples_01_CMakeLists.txt mlip-3/test/examples/01.train/CMakeLists.txt
cp mlip3_missing_files/test_examples_02_CMakeLists.txt mlip-3/test/examples/02.check_errors/CMakeLists.txt
cp mlip3_missing_files/test_examples_03_CMakeLists.txt mlip-3/test/examples/03.calculate_efs/CMakeLists.txt
cp mlip3_missing_files/test_examples_04_CMakeLists.txt mlip-3/test/examples/04.calculate_grade/CMakeLists.txt
cp mlip3_missing_files/test_examples_05_CMakeLists.txt mlip-3/test/examples/05.cut_extrapolative_neighborhood/CMakeLists.txt
cp mlip3_missing_files/test_examples_06_CMakeLists.txt mlip-3/test/examples/06.select_add/CMakeLists.txt
cp mlip3_missing_files/test_examples_07_CMakeLists.txt mlip-3/test/examples/07.relax/CMakeLists.txt
cp mlip3_missing_files/test_examples_08_CMakeLists.txt mlip-3/test/examples/08.relax_preselect/CMakeLists.txt
cp -r mlip3_missing_files/cmake mlip-3/
cp mlip3_missing_files/mlp_commands.cpp mlip-3/src/mlp/mlp_commands.cpp
```
A .sh file is provided in the missing_data folder for this.
For the last file, we commented the tests as they were not working.

The CMakeLists files are based on those present in MLIP-2, but modified for the content of MLIP-3.

We can now compile.

On mac, we use openblas:

```
cmake .. -DBLAS_ROOT=/usr/local/opt/openblas/
make
```
This takes a few minutes and raises a lot of warnings. If no error message appears, then MLIP has compiled!

You can try running the tests in *mlip-3/test/examples/* and compare to the results in the respective *sample_out* folder.
This can be done automatically with the following command:
```
make tests
```

## Running MLIP

If the installation is successful, the command is 
```
mlip-3/build/mlp
```
This is called by the *train_mtp.py* script.
