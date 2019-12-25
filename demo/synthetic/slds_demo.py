import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import logging
import os

from tqdm import tqdm
from sgmcmc_ssm.models.slds import (
       generate_slds_data,
       SLDSParameters,
       SLDSPrior,
       SLDSPreconditioner,
       SLDSHelper,
       SLDSSampler,
       )
from sgmcmc_ssm.plotting_utils import (
        compare_metrics,
        plot_trace_plot,
        )
sns.set()
LOGGING_FORMAT = '%(levelname)s: %(asctime)s - %(name)s: %(message)s ...'
logging.basicConfig(
        level = logging.INFO,
        format = LOGGING_FORMAT,
        )

## =========================================================
path_to_save ="scratch/SLDS_demo/"
num_inits = 1
np.random.seed(12345)
## =========================================================

# Parameters
## Define SLDS Parameters
alpha = 0.1
delta=np.pi/2

pi = np.array([
        [1-alpha, alpha],
        [alpha, 1-alpha]])
rot_mat = lambda rho: np.array([[np.cos(rho), -np.sin(rho)], [np.sin(rho), np.cos(rho)]])
A = np.array([rot_mat(-delta/2), rot_mat(delta/2)]) * 0.9
Q = np.array([np.eye(2), np.eye(2)]) * 0.1
C = np.eye(2)
R = np.eye(2)*0.1

logit_pi = np.log(pi+0.000001)
LQinv = np.array([np.linalg.cholesky(np.linalg.inv(Q_k)) for Q_k in Q])
LRinv = np.linalg.cholesky(np.linalg.inv(R))

parameters = SLDSParameters(logit_pi=logit_pi, A=A, C=C, LQinv=LQinv, LRinv=LRinv)

# Generate Data
my_data = generate_slds_data(T=500, parameters=parameters)

# Generate Samplers
preconditioner = SLDSPreconditioner()
sampler_steps = {
        "SGRLD XZ (buffer)": [
            ['sample_sgrld', 'project_parameters']*10,
            [{'epsilon': 0.1, 'subsequence_length': 5, 'buffer_length': 5,
                'kind': 'complete', 'minibatch_size': 10,
                'preconditioner': preconditioner}, {}]*10,
        ],
        "SGRLD X (buffer)": [
            ['sample_sgrld', 'project_parameters']*10,
            [{'epsilon': 0.1, 'subsequence_length': 5, 'buffer_length': 5,
                'kind': 'x_marginal', 'minibatch_size': 10,
                'preconditioner': preconditioner}, {}]*10,
        ],
        "SGRLD Z (buffer)": [
            ['sample_sgrld', 'project_parameters']*10,
            [{'epsilon': 0.1, 'subsequence_length': 5, 'buffer_length': 5,
                'kind': 'z_marginal', 'minibatch_size': 10,
                'preconditioner': preconditioner}, {}]*10,
        ],
        "Gibbs": [
            ['sample_gibbs', 'project_parameters'],
            [{}, {}],
        ],
    }

my_samplers = {
        (key, init): SLDSSampler(name=key, **parameters.dim)
        for key in sampler_steps.keys()
        for init in range(num_inits)
        }

my_prior = SLDSPrior.generate_default_prior(var=1, **parameters.dim)
for key, sampler in my_samplers.items():
    sampler.setup(my_data['observations'], my_prior,
            )
    sampler.init_sample_latent()
    sampler.project_parameters()

# Init Samplers from Prior
init_parameters = {}
for _, sampler in my_samplers.items():
    for init in tqdm(range(num_inits)):
        sampler.sample_gibbs()
        init_parameter = sampler.parameters.project_parameters()
        init_parameters[init] = init_parameter.copy()
    break

for (key, init), sampler in my_samplers.items():
    init_param = init_parameters[init].copy()
    sampler.parameters =  init_param

# Setup my_evaluators
from sgmcmc_ssm.evaluator import SamplerEvaluator
from sgmcmc_ssm.metric_functions import (
        metric_function_from_sampler,
        metric_function_parameters,
        metric_compare_x,
        metric_compare_z,
        noisy_logjoint_loglike_metric,
        sample_function_parameters,
        )
parameter_names = ['pi']
parameter_names2 = ['A', 'Q']
parameter_names3 = ['C', 'R']
my_metric_functions = [
        metric_function_parameters(parameter_names,
            target_values=[getattr(my_data['parameters'], parameter_name)
                for parameter_name in parameter_names],
            metric_names=['logmse' for parameter_name in parameter_names],
            criteria=[min for parameter_name in parameter_names],
            double_permutation_flag=True,
            ),
        metric_function_parameters(parameter_names,
            target_values=[getattr(my_data['parameters'], parameter_name)
                for parameter_name in parameter_names],
            metric_names=['mse' for parameter_name in parameter_names],
            criteria=[min for parameter_name in parameter_names],
            double_permutation_flag=True,
            ),
        metric_function_parameters(parameter_names2,
            target_values=[getattr(my_data['parameters'], parameter_name)
                for parameter_name in parameter_names2],
            metric_names=['mse' for parameter_name in parameter_names2],
            criteria=[min for parameter_name in parameter_names2],
            ),
        metric_function_parameters(parameter_names3,
            target_values=[getattr(my_data['parameters'], parameter_name)
                for parameter_name in parameter_names3],
            metric_names=['mse' for parameter_name in parameter_names2],
            ),
#        metric_compare_x(my_data['latent_vars']['x']),
#        metric_compare_z(my_data['latent_vars']['z']),
        noisy_logjoint_loglike_metric(subsequence_length=50, buffer_length=10),
    ]
my_sample_functions = [
        sample_function_parameters(
            parameter_names2 + ['LRinv', 'LQinv']
            ),
    ]
my_evaluators = {
        "{0}_{1}".format(*key): SamplerEvaluator(sampler,
            my_metric_functions, my_sample_functions,
            sampler_name="{0}_{1}".format(*key))
        for key, sampler in my_samplers.items()
        }

keys = my_evaluators.keys()
for step in tqdm(range(1000)):
    for ii, key in enumerate(keys):
        my_evaluators[key].evaluate_sampler_step(*sampler_steps[key.split("_")[0]])

    if (step % 25) == 0:
        logging.info("============= CHECKPOINT ================")
        if not os.path.isdir(path_to_save):
            os.makedirs(path_to_save)
        joblib.dump({
            key: evaluator.get_state()
            for key, evaluator in my_evaluators.items()},
            os.path.join(path_to_save, "slds_demo.p"))
        g = compare_metrics(my_evaluators)
        g.savefig(os.path.join(path_to_save, "metrics_compare.png"))
        if step > 50:
            g = compare_metrics(my_evaluators, full_trace=False)
            g.savefig(os.path.join(path_to_save, "metrics_compare_zoom.png"))

        for key in my_evaluators.keys():
            sampler = my_evaluators[key].sampler
            fig, axes = plot_trace_plot(my_evaluators[key], single_variables=['C', 'LRinv', 'R', 'Rinv'])
            fig.suptitle(key)
            fig.savefig(os.path.join(path_to_save, "{0}_trace.png".format(key)))
        plt.close('all')








