"""
Create Setup Script for Demo Experiment

Usage:
  0. Change `project_root` in this file to match the directory of `sgmcmc_nonlinear_ssm`
  1. Run this script, which will create a `setup.sh` script at `<experiment_folder>`
        (default is `./scratch/<experiment_name>/scripts/setup.sh`)
        This will generate the train + test data, initializations + other scripts
  2. Run the `fit.sh` script to fit the models specified in this file
        generates output to `<experiment_folder>/out/fit`
  3. Run `eval_train.sh` or `eval_test.sh` to evaluate the fits on the train or test data
  4. Run `process_out.sh` to aggregate the results to `<experiment_folder>/processed/`
        generates csv files that can be used to make figures
        the main two csv files of interest are:
        "aggregated.csv" and "options.csv" which can be joined together on experiment_id
"""

# Standard Imports
import numpy as np
import pandas as pd
import os
import sys
import joblib
from sklearn.model_selection import ParameterGrid


import logging
LOGGING_FORMAT = '%(levelname)s: %(asctime)s - %(name)s: %(message)s ...'
logging.basicConfig(
        level = logging.INFO,
        format = LOGGING_FORMAT,
        )

## Set Experiment Name
experiment_name = "lgssm_demo"

## Filesystem Paths
conda_env_name = None
project_root = "./" # Must be specified (path to "/sgmcmc_ssm_code/")
os.chdir(project_root)
sys.path.append(os.getcwd()) # Fix Python Path

# Paths relative to project root
current_folder = os.path.join("nonlinear_ssm_pf_experiment_scripts", "lgssm")
python_script_path = os.path.join(current_folder,"driver.py")
experiment_folder = os.path.join("scratch", experiment_name) # Path to output

# Synthetic Data Args
data_reps = 1 # Number of training sets
T = 1000 # Training set size
T_test = 1000 # Test set size
init_methods = ['prior', 'truth'] * 1 # number of intializations + how they are initialized

# LGSSM parameters
from sgmcmc_ssm.models.lgssm import (
        LGSSMParameters,
        )

param_name = 'A=0.9,Q=0.1,R=1'
A = np.eye(1)*0.9
Q = np.eye(1)*0.1
C = np.eye(1)
R = np.eye(1)

LQinv = np.linalg.cholesky(np.linalg.inv(Q))
LRinv = np.linalg.cholesky(np.linalg.inv(R))
parameters = LGSSMParameters(A=A, LQinv=LQinv, C=C, LRinv=LRinv)
parameters.project_parameters()
parameter_list = {param_name: parameters}

# Sampler Args
common_sampler_args = {
        'inference_method': ['SGRLD'],
        'subsequence_length': [40],
        'buffer_length': [10],
        'minibatch_size': [1],
        'steps_per_iteration': [10],
        'max_num_iters': [10000],
        'max_time': [300],
        'epsilon': [0.1],
        }

sampler_args = [
    {
        'method_name': ['Gibbs'],
        'inference_method': ['Gibbs'],
        'epsilon': [-1],
        'minibatch_size': [-1],
        'subsequence_length': [-1],
        'buffer_length': [-1],
        'steps_per_iteration': [1],
        'max_time': [300],
    },
    {
        'method_name': ['KF'],
        'kind': ['marginal'],
        **common_sampler_args
    },
    {
        'method_name': ['MC_100'],
        'kind': ['complete'],
        'num_samples': [100],
        **common_sampler_args
    },
#    {
#        'method_name': ['MC_1000'],
#        'kind': ['complete'],
#        'num_samples': [1000],
#        **common_sampler_args
#    },
    {
        'method_name': ['NEMETH_100'],
        'kind': ['pf'],
        'pf_kwargs': [dict(pf='nemeth', N=100, lambduh=0.95)],
        **common_sampler_args
    },
#    {
#        'method_name': ['PARIS_100'],
#        'kind': ['pf'],
#        'pf_kwargs': [dict(pf='paris', N=100, Ntilde=2)],
#        **common_sampler_args
#    },
#    {
#        'method_name': ['POYIADJIS_N2_100'],
#        'kind': ['pf'],
#        'pf_kwargs': [dict(pf='poyiadjis_N2', N=100)],
#        **common_sampler_args
#    },

]

# Script Kwargs (only really matters when using cluster)
setup_script_kwargs=dict(deploy_target='desktop')
fit_script_kwargs=dict(deploy_target='desktop')
eval_script_kwargs=dict(deploy_target='desktop')
process_out_script_kwargs=dict(deploy_target='desktop')
make_plots_script_kwargs=dict(deploy_target='desktop')


############################################################################
## MAIN SCRIPT
############################################################################
from sgmcmc_ssm.driver_utils import (
        script_builder
        )
if __name__ == "__main__":
    # Setup Folders
    logging.info("Creating Folder for {0}".format(experiment_name))
    path_to_shell_script = os.path.join(experiment_folder, "scripts")
    if not os.path.isdir(experiment_folder):
        os.makedirs(experiment_folder)
    if not os.path.isdir(path_to_shell_script):
        os.makedirs(path_to_shell_script)

    # Create Additional Args
    path_to_additional_args = os.path.join(experiment_folder,
            "setup_additional_args.p")
    sampler_args = [arg
            for args in sampler_args
            for arg in list(ParameterGrid(args))
            ]
    additional_args = dict(
            sampler_args=sampler_args,
            python_script_path=python_script_path,
            path_to_shell_script=path_to_shell_script,
            project_root=project_root,
            experiment_name=experiment_name,
            T=T,
            T_test=T_test,
            parameter_list=parameter_list,
            data_reps=data_reps,
            init_methods=init_methods,
            conda_env_name=conda_env_name,
            fit_script_kwargs=fit_script_kwargs,
            eval_script_kwargs=eval_script_kwargs,
            process_out_script_kwargs=process_out_script_kwargs,
            make_plots_script_kwargs=make_plots_script_kwargs,
            )
    joblib.dump(additional_args, path_to_additional_args)

    # Create Setup Script
    bash_file_masters = script_builder(
            script_name="setup",
            python_script_path=python_script_path,
            python_script_args=[{
                "--experiment_folder": experiment_folder,
                "--path_to_additional_args": path_to_additional_args,
                "--setup": None,
                }],
            path_to_shell_script=path_to_shell_script,
            project_root=project_root,
            conda_env_name=conda_env_name,
            **setup_script_kwargs,
            )

    logging.info("Run {0} to complete settting up {1}".format(
            bash_file_masters[0], experiment_name))

# EOF
