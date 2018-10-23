# Stochastic Gradient MCMC for Time Series SSMs Code

The `sgmcmc_ssm` folder contains the python module code for ["Stochastic Gradient MCMC for State Space Models"](https://arxiv.org/abs/1810.09098).

The `demo` folder contains python scripts that demonstrate the `sgmcmc_ssm` API
* `hmm_long_demo.py`, `arhmm_long_demo.py`, `lgssm_long_demo.py`, `slds_long_demo.py` cover the various classes and methods provided
* `hmm_demo`, `lgssm_demo`, `slds_demo` are example scripts of how to compare various MCMC samplers on synthetic data.

These scripts must be run with `sgmcmc_ssm` on PYTHONPATH.
For example, running `ipython demo/<script name>.py` from this `code` folder.


