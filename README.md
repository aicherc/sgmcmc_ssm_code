# Stochastic Gradient MCMC for State Space Models

This repo contains the python code for stochastic gradient MCMC in state space models for the following papers: [Stochastic Gradient MCMC for SSMs](https://arxiv.org/abs/1810.09098) and ["Stochastic Gradient MCMC for Nonlinear SSMs"](https://arxiv.org/abs/1901.10568).


## Overview
The `sgmcmc_ssm` folder contains the python module code.

The `demo` folder contains python scripts that demonstrate the `sgmcmc_ssm` module API.
These scripts must be run with `sgmcmc_ssm` on the PYTHONPATH.
For example, running `ipython demo/<script name>.py` from this project root folder.


## Installation
Add the `sgmcmc_ssm` folder to the PYTHONPATH by (i) running code from the project root folder, (ii) adding the project root folder to the python path using `sys.path.append(<path_to_sgmcmc_ssm_code>)`, or (iii) running `setup.py`.

Requirements:
Python 3+, numpy, pandas, scipy, seaborn, joblib, scikit-learn,


## Usage Examples
See `demo/lgssm_quick_start.py` for a quick overview of how to use the code.
See `demo/api_demos/` for additional details for each model.

Synthetic LGSSM Script Example
```
cd <path_to_sgmcmc_ssm_code>
python demo/synthetic/lgssm_demo.py
```

Ion-Channel Gaussian HMM Example
```
cd <path_to_sgmcmc_ssm_code>
python demo/ion_channel/ion_channel_subset_demo.py
```

Exchange Rate SVM Example
```
cd <path_to_sgmcmc_ssm_code>
python demo/exchange_rate/process_exchange_data.py
python demo/exchange_rate/exchange_rate_subset_demo.py
```

## Meta

Christopher Aicher – aicherc [at] uw [dot] edu

Distributed under the MIT license. See ``LICENSE`` for more information.

[https://github.com/aicherc/sgmcmc_ssm_code](https://github.com/aicherc/sgmcmc_ssm_code)

## Contributing

1. Fork it (<https://github.com/aicherc/sgmcmc_ssm_code/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

