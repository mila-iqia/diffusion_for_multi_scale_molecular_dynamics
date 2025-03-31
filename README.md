# Diffusion for Multiscale Molecular Dynamics

This project implements diffusion-based generative models for periodic atomistic systems (i.e., crystals).
The aim of this project is to be able to train such a model and use it as part of an active learning
framework, where a Machine Learning Interatomic Potential (MLIP) is continually fine-tuned on labels obtained
from a costly oracle such as Density Functional Theory (DFT). The generative model is used to create
few-atom configurations that are computationally tractable for the costly oracle by inpainting 
around problematic atomic configurations. 


# Instructions to set up the project 

## Creating a Virtual Environment
The project dependencies are stated in the `pyproject.toml` file. They must be installed in a virtual environment.

### uv
The recommended way of creating a virtual environment is to use the tool [`uv`](https://docs.astral.sh/uv/). 
Once `uv` is installed locally, the virtual environment can be created with the command
    
    uv sync [--extra pyace]

which will install the exact environment described in file `uv.lock`. The environment can then be activated with
the command

    source .venv/bin/activate

Note that the optional dependency "pyace" is needed for the active learning loop. It is recommended to install it 
for development.

### pip

Alternatively, `pip` can be used to create the virtual environment. Assuming `python` and `pip` are already
available on the system, create a virtual env folder in the root directory with the command

    python -m venv ./.venv/

The environment must then be activated with the command

    source .venv/bin/activate

and the environment should be created in `editable` mode so that the source code can be modified directly,

    pip install -e .

### Testing the Installation
The test suite should be executed to make sure that the environment is properly installed. After activating the 
environment, the tests can be executed with the command

    pytest [--quick] [-n auto]

The argument `--quick` is optional; a few tests are a bit slow and will be skipped if this flag is present.
The argument `-n auto` is optional; if toggled, the tests will run in parallel and go a little faster. 

# Getting Started

Once the environment is set up and the tests pass, the `tutorials` folder is a good place to start. There, 
Jupyter-notebooks give a quick tour of what the code base can do. These are designed to be lightweight and to 
run locally on a laptop: no need for extra datasets or GPUs. 

Next, the `experiments`[TODO: rename this] folder provides many mini-experiments that give insight into the inner workings of the
code. Again, these are self-contained and lightweight enough to run on a laptop. Try them! A README in that folder
provides more information.

Finally, run a fully-fledged experiment. To use [Comet](https://www.comet.com/) as an experiment logger, an account 
must be available and a global configuration file must be created at `$HOME/.comet.config` with content of the form

    [comet]
    api_key=YOUR_API_KEY

A simple experiment is described in the configuration file

    examples/config_files/diffusion/config_diffusion_mlp.yaml

To run the experiment described in this file, a dataset must first be created by executing the script

    data/Si_diffusion_1x1x1/create_data.sh

Then, the experiment itself can be executed by running the script

    examples/local/diffusion/run_diffusion.sh


# For Developers

## Setting up the Development Tools
Various automated tools are used in order to maintain a high quality code base. These must be set up
to start developing. We use

* [flake8](https://flake8.pycqa.org/en/latest/) to insure the coding style is enforced.
* [isort](https://pycqa.github.io/isort/) to insure that the imports are properly ordered.
* [black](https://pypi.org/project/black/) to format the code.

### Setup pre-commit hooks
The folder `./hooks/` contain "pre-commit" scripts that automate various checks at every git commit.
These hooks will 
* validate flake8 before any commit;
* check that jupyter notebook outputs have been stripped.

There are two pre-commit scripts, `pre-commit` and `pre-commit_staged`. Both scripts perform the same
checks; `pre-commit` is used within the continuous integration (CI), while `pre-commit_staged` only 
validates files that are staged in git, making it more developer-friendly.

To activate the pre-commit hook,

    cd .git/hooks/ && ln -s ../../hooks/pre-commit .

Alternatively, to only lint files that have been staged in git, use

    cd .git/hooks/ && ln -s ../../hooks/pre-commit_staged pre-commit

### Setup Continuous Integration

GitHub Actions is used for running continuous integration (CI) checks. 
The cI workflow is described in `.github/workflows/ci.yml`.

CI will run the following:
- check the code syntax with `flake8` 
- execute the unit tests in `./tests/`.
- Checks on documentation presence and format (using `sphinx`).

Since the various tests are relatively costly, the CI actions will only be executed for 
pull requests to the `main` branch.

