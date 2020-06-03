#!/usr/bin/python3
#This script is being used to test the particle gradient approximations (full data)

#######IMPORT RELEVANT MODULES######################
import numpy as np
import pandas as pd
import time
import joblib
import os

from sgmcmc_ssm.models.garch import (
        GARCHParameters,
        GARCHSampler,
        GARCHPrior,
        generate_garch_data,
        GARCHHelper,
        )
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns



### Main Function

def make_plots(T, L, N_reps, N_trials, pars, buffer_sizes, path_to_out, seed=12345, save_dat=True):
    print("\n===========================================================")
    print("T = {0}, L = {4}, N_reps = {1}, N_trial={5}, pars = {2}, buffer_sizes = {3}".format(
        T, N_reps, pars, buffer_sizes, L, N_trials))
    print("===========================================================\n")
    np.random.seed(seed)


    # Generate Data
    alpha, beta, gamma, tau = pars[0], pars[1], pars[2], pars[3]
    R = np.eye(1)*tau**2

    log_mu, logit_phi, logit_lambduh = \
            GARCHParameters.convert_alpha_beta_gamma(alpha, beta, gamma)
    LRinv = np.linalg.cholesky(np.linalg.inv(R))
    parameters = GARCHParameters(
            log_mu=log_mu,
            logit_phi=logit_phi,
            logit_lambduh=logit_lambduh,
            LRinv=LRinv,
            )
    parameters_dict = parameters.as_dict().copy()

    def convert_gradient(grad_dict):
        return [
            grad_dict['log_mu'],
            grad_dict['logit_phi'],
            grad_dict['logit_lambduh'],
            grad_dict['LRinv_vec'],
            ]

    results_dfs = []
    for trial in tqdm(range(N_trials), desc="Trial"):
        data = generate_garch_data(T=T, parameters=parameters, tqdm=tqdm)
        t0 = (T+L)//2
        observations = data['observations']
        helper = GARCHHelper(forward_message=data['initial_message'],
                **parameters.dim)


        # Compute Exact (Full Buffered Gradient)
        start_time = time.time()
        full_buffer_gradients = [None]*10
        pbar = tqdm(range(10))
        pbar.set_description('Number of Reps')
        buffer_size = 12
        pf_kwargs = dict(
            observations=observations[t0-buffer_size:t0+L+buffer_size],
            parameters=parameters,
            kernel=None,
            subsequence_start = buffer_size,
            subsequence_end = L+buffer_size,
            pf='poyiadjis_N',
            N=1000000,
            tqdm=tqdm,
        )
        for rep in pbar:
            full_buffer_gradients[rep] = convert_gradient(
                    helper.pf_gradient_estimate(
                        **pf_kwargs,
                    ))
        full_buffer_gradient = np.mean(full_buffer_gradients, axis=0)
        full_buffer_gradient_sd = np.std(full_buffer_gradients, axis=0)
        print(full_buffer_gradient)
        print(full_buffer_gradient_sd)
        full_buffer_time = time.time() - start_time

        estimates_bs = [dict(
                poyiadjis_100=[], poyiadjis_1000=[], poyiadjis_10000=[])
                for _ in range(len(buffer_sizes))]
        runtimes_bs = [{key:[] for key in estimates_bs[0].keys()}
                for _ in range(len(buffer_sizes))]
        pbar_bs = tqdm(zip(buffer_sizes, estimates_bs, runtimes_bs),
                desc="buffer size",
                total=len(buffer_sizes))
        for buffer_size, estimates, runtimes in pbar_bs:
            pf_kwargs = dict(
                observations=observations[t0-buffer_size:t0+L+buffer_size],
                parameters=parameters,
                kernel=None,
                subsequence_start = buffer_size,
                subsequence_end = L+buffer_size,
                tqdm=tqdm,
            )

            pbar = tqdm(range(N_reps))
            pbar.set_description('Number of Reps')
            for rep in pbar:
                # Poyiadjis N Smoother
                start_time = time.time()
                pf_kwargs.update(N=100, pf="poyiadjis_N")
                poy_estimate = convert_gradient(helper.pf_score_estimate(**pf_kwargs))
                estimates['poyiadjis_100'].append(poy_estimate)
                runtimes['poyiadjis_100'].append(time.time() - start_time)

                # Poyiadjis N Smoother
                start_time = time.time()
                pf_kwargs.update(N=1000, pf="poyiadjis_N")
                poy_estimate = convert_gradient(helper.pf_score_estimate(**pf_kwargs))
                estimates['poyiadjis_1000'].append(poy_estimate)
                runtimes['poyiadjis_1000'].append(time.time() - start_time)

                # Poyiadjis N Smoother
                start_time = time.time()
                pf_kwargs.update(N=10000, pf="poyiadjis_N")
                poy_estimate = convert_gradient(helper.pf_score_estimate(**pf_kwargs))
                estimates['poyiadjis_10000'].append(poy_estimate)
                runtimes['poyiadjis_10000'].append(time.time() - start_time)


        dfs = []
        variables = ['log_mu', 'logit_phi', 'logit_lambduh', 'LRinv_vec']
        for buffer_size, estimates, runtimes in zip(buffer_sizes, estimates_bs, runtimes_bs):
            for key, value in estimates.items():
                df = pd.DataFrame(np.array(value), columns=variables)
                df.index.name = 'rep'
                df = df.reset_index()
                df['runtime'] = runtimes[key]
                df = df.melt(id_vars='rep')
                df['buffer_size'] = buffer_size
                df['sampler'] = key
                dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)

        # Checkpoint
        if not os.path.isdir(os.path.join(path_to_out, 'trial')):
            os.makedirs(os.path.join(path_to_out, 'trial'))
        joblib.dump(df, os.path.join(path_to_out, 'trial',
            'dat{0}_joblib.gz'.format(trial)))

        # Append Results
        for ii, variable in enumerate(variables):
            true_grad = full_buffer_gradient[ii]
            var_df = df[df['variable'] == variable]
            runtime_df = df[df['variable'] == 'runtime']
            for (sampler, buffer_size), sub_df in var_df.groupby(
                    ['sampler', 'buffer_size']):
                result_df = pd.DataFrame([dict(
                        sampler=sampler,
                        buffer_size=buffer_size,
                        trial=trial,
                        variable=variable,
                        mse=np.mean((sub_df['value'] - true_grad)**2),
                        bias_sq=(np.mean(sub_df['value']) - true_grad)**2,
                        var=np.var(sub_df['value']),
                        mean_runtime=np.mean(
                            runtime_df.query('sampler == @sampler & buffer_size == @buffer_size')['value'])
                        )])
                results_dfs.append(result_df)

        # Checkpoint Results
        total_result_df = pd.concat(results_dfs, ignore_index=True)
        joblib.dump(total_result_df,
                os.path.join(path_to_out, 'summary_dat_joblib.gz'))

        if trial % 10 == 0:
            for variable in variables:
                plt.close('all')
                fig, ax = plt.subplots(1,1)
                sns.boxplot(x='sampler', y='mse', hue='buffer_size',
                        data=total_result_df.query('variable == @variable'),
                        ax=ax)
                ax.set_title("Boxplot of Gradient MSE")
                fig.set_size_inches(8,6)
                fig.savefig(os.path.join(path_to_out, "{0}_mse.png".format(variable)))
                ax.set_yscale('log')
                fig.savefig(os.path.join(path_to_out, "{0}_logmse.png".format(variable)))

                fig, ax = plt.subplots(1,1)
                sns.boxplot(x='sampler', y='bias_sq', hue='buffer_size',
                        data=total_result_df.query('variable == @variable'),
                        ax=ax)
                ax.set_title("Boxplot of Gradient Bias Squared")
                fig.set_size_inches(8,6)
                fig.savefig(os.path.join(path_to_out, "{0}_bias.png".format(variable)))
                ax.set_yscale('log')
                fig.savefig(os.path.join(path_to_out, "{0}_logbias.png".format(variable)))

                fig, ax = plt.subplots(1,1)
                sns.boxplot(x='sampler', y='var', hue='buffer_size',
                        data=total_result_df.query('variable == @variable'),
                        ax=ax)
                ax.set_title("Boxplot of Gradient Variance")
                fig.set_size_inches(8,6)
                fig.savefig(os.path.join(path_to_out, "{0}_var.png".format(variable)))
                ax.set_yscale('log')
                fig.savefig(os.path.join(path_to_out, "{0}_logvar.png".format(variable)))
                plt.close('all')


### Script
if __name__ == "__main__":
    N_reps = 10 #number of repetitions
    N_trials = 10
    buffer_sizes = np.array([8, 6, 4, 3, 2, 1, 0])
    alpha = 0.1
    beta = 0.8
    gamma = 0.05
    tau = 0.3

    # Set 1
    T = 40 #length of series
    L = 16
    pars = np.array((alpha, beta, gamma, tau))
    path_to_out = os.path.join(
            "./scratch/garch_grad_compare/",
            "{0}".format(tuple(pars)))
    make_plots(T, L, N_reps, N_trials, pars, buffer_sizes, path_to_out)




