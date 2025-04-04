# Toy Problems

This directory considers a simplified "toy problem" composed of two "pseudo atoms" in one dimension. 
The data distribution is an isotropic Gaussian distribution about the mean relative coordinates, 
which are `[[0.25], [0.75]]`, i.e., the first atom at 0.25 and the second atom  at 0.75. The width of the 
Gaussian is controlled by the parameter "sigma_d".

The two great advantages of such a simplified system are :
 * we can solve analytically for what the score vector field should be
 * it is straightforward to visualize the score field.

This helps us gain insights into the diffusion process.

Various regularization strategies are tested on this simple toy problem.

# Directory content

## training/
this is where the various training experiments are defined. They are simple enough to run 
on a laptop and do not require external datasets. Try them!

All experiments are organized in the same way. The file `base_config.yaml` defines all the common 
hyper-parameters that will be used by the experiments. No claim is made that these are "the best" 
hyper-parameters. 

The sub-folder name indicates what kind of experiment is done in the subfolder, and the 
file `specific_config.yaml` contains the regularization configuration. Each subfolder contains a 
`run.sh` bash script that combines `base_config.yaml` and `specific_config.yaml` into  the final 
`config.yaml` and then  launches the training job locally.

To launch the training job,

> bash run.sh

To follow the experiment's progress, use tensorboard,

> tensorboard --logdir output --samples_per_plugin images=99999

The model used is a simple MLP that is made permutation equivariant. This lets us investigate the impact of 
equivariance on the model. Note that this model is not translation invariant, or equivariant to spatial symmetries.

### **no_regularizer** 
no regularization is applied. Only score matching!

### **analytical_regression_regularizer**
This is an *oracle* regularization scheme, where we use regression to the known 
solution. This would not be available in general; this is a sanity check that the score network can learn the right answer.

### **fokker_planck_regularizer**
the Fokker-Planck differential equation is used as a loss term to regularize the score. 
This approach is adapted from the work from the Ermon group [1].


### **consistency_regularizer** 
There is a consistency relationship between the score at different diffusion times [2]. 
We leverage this to regularize the score network. This requires drawing trajectories 
during training: no backprop is done on the trajectories, but the samples obtained
are used to create a Monte Carlo estimate of the consistency condition that should be satisfied.

### **consistency_with_analytical_guide_regularizer** 
This is an *oracle* regularization scheme, where we use the known analytical score to draw the 
needed consistency trajectories needed by the consistency regularizer. This would not be available in general; 
this is a sanity check that the consistency regularizer can work if given a good source of truth.

### References 
[1] Lai, C.H., Takida, Y., Murata, N., Uesaka, T., Mitsufuji, Y. and Ermon, S., 2022. Regularizing score-based models 
with score fokker-planck equations. In NeurIPS 2022 Workshop on Score-Based Methods.

[2] Daras, G., Dagan, Y., Dimakis, A. and Daskalakis, C., 2023. Consistent diffusion models: Mitigating sampling drift 
by learning to be consistent. Advances in Neural Information Processing Systems, 36, pp.42038-42063.


## `analyse_experiments.py`
This is the main analysis script. It will generate diffusion samples and
a video of the score vector field, which can be visualized because the toy problem only has 2 dimensions!

The script goes through a list of all the "experiment_name", assuming all experiments have been executed.
The script should be modified if only a subset of experiments are ready for analysis.

Note that
- For experiment_name = "analytical", no experiment actually need to have been executed: the analytical solution
  is used.
- for any other experiment_name, the script assumes that an experiment has already been executed: 
  the analysis requires a trained checkpoint.

This script will create a folder named "generated_artifacts/" where the
samples, images and videos will be written. These will be organized following the same
folder structure as experiments/.

## utils/
This folder contains library code that is used to generate samples and analyse results.
