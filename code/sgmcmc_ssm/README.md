
## Installation
Add the `sgmcmc_ssm` folder to the PYTHONPATH.

Requirements:
Python 3+, numpy, pandas, scipy, matplotlib, seaborn, joblib, scikit-learn,

## Overview
The main classes are:
* `Parameters` - responsible for keeping track of parameters and common re-parametrizations  (see `base_parameter.py`)
* `Prior` - responsible for the prior distribution of `Parameters`  (see `base_parameter.py`)
* `Preconditioner` - responsible for preconditioning gradients of `Parameters`  (see `base_parameter.py`)
* `Helper` - responsible for loglikelihood, sampling latent variables, and gradients (see `sgmcmc_sampler.py`)
* `Sampler` - responsible for running MCMC (see `sgmcmc_sampler.py`)
* `Evaluator` - responsible for collecting MCMC samples, and evaluating MCMC samples (see `metric_functions.py` and `evaluator.py`)

Model-specific child classes can be found in the `models/` folder.

The `Parameters`, `Prior`, and `Preconditioner` are composed of mixins of individual parameters that can be found in the `variable_mixins/` folder.

Other utility functions + plotting are in `_utils.py` and `plotting_utils.py`


