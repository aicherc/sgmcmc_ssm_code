import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import logging
logger = logging.getLogger(name=__name__)

def plot_metrics(evaluator, full_trace=True, burnin=None,
        x='iteration'):
    df = evaluator.get_metrics()
    df['variable:metric'] = df['variable']+':'+df['metric']
    if burnin is None:
        if not full_trace:
            df = df.query('iteration >= {0}'.format(evaluator.iteration/2))
    else:
        df = df.query('iteration >= {0}'.format(burnin))
    df = df.sort_values(['variable:metric', 'iteration'])
    g = sns.FacetGrid(df, col='variable:metric',
            col_wrap=4, sharey=False).map(
                plt.plot, x, 'value'
                ).add_legend().set_titles("{col_name}")
    return g

def compare_metrics(evaluators, full_trace=True, burnin=None, errorband=False,
        x='iteration'):

    # Concat Evaluator Metrics
    if isinstance(next(iter(evaluators.keys())), tuple):
        df = pd.concat(
            [evaluator.get_metrics().assign(method=name, init=init)
                for ((name, init), evaluator) in evaluators.items()],
            ignore_index=True,
            )
    else:
        df = pd.concat(
            [evaluator.get_metrics().assign(method=name, init=0)
                for (name, evaluator) in evaluators.items()],
            ignore_index=True,
            )
    df['variable:metric'] = df['variable']+':'+df['metric']

    # Subset Data
    if burnin is None:
        if not full_trace:
            min_iteration = min([evaluator.iteration
                for evaluator in evaluators.values()])
            df = df.query('iteration >= {0}'.format(min_iteration/2))
    else:
        df = df.query('iteration >= {0}'.format(burnin))


    df = df.sort_values(['method', 'variable:metric', 'iteration'])
    if errorband:
        # TODO set x to mean(x) when groupby iteration if x != iteration
        g = sns.FacetGrid(df, col='variable:metric', hue="method",
            col_wrap=4, sharey=False).map_dataframe(
                sns.lineplot, x=x, y='value',
                estimator='mean', ci='sd',
                )
    else:
        g = sns.FacetGrid(df, col='variable:metric', hue="method",
            col_wrap=4, sharey=False).map_dataframe(
                sns.lineplot, x=x, y='value',
                units='init', estimator=None, ci=None,
                )
    g = g.add_legend().set_titles("{col_name}").set_xlabels(x)
    return g

def plot_trace_plot(evaluator, full_trace=True, query_string=None,
        single_variables=[], burnin=None, x='iteration'):
    samples_df = evaluator.get_samples()
    if burnin is None:
        if not full_trace:
            samples_df = samples_df.query('iteration >= {0}'.format(evaluator.iteration/2))
    else:
        samples_df = samples_df.query('iteration >= {0}'.format(burnin))
    if query_string is not None:
        samples_df = samples_df.query(query_string)
    variables = samples_df['variable'].sort_values().unique()
    xs = samples_df[x].sort_values().unique()
    variable_values = {
            key: np.array(df.sort_values(x)['value'].tolist())
            for key, df in samples_df.groupby('variable')
            }

    # Construct Plots
    num_states = getattr(evaluator.sampler, 'num_states', 1)
    fig, axes = plt.subplots(len(variables), num_states,
            sharex='col', sharey='row',
            figsize=(4, len(variables)*3),
            )
    if num_states == 1:
        axes = np.array([[ax] for ax in axes])
    for ii_var, variable in enumerate(variables):
        values = variable_values[variable]
        for k in range(num_states):
            ax = axes[ii_var, k]
            ax.set_title('{0}_{1}'.format(variable, k))
            if variable in single_variables:
                values = np.reshape(values, (values.shape[0], -1))
                for dim in range(values.shape[1]):
                    ax.plot(xs, values[:, dim],
                        label='{0}[{1}]'.format(variable, dim))
            else:
                values = np.reshape(values, (values.shape[0], num_states, -1))
                for dim in range(values.shape[2]):
                    ax.plot(xs, values[:, k, dim],
                            label='{0}[{1}]'.format(variable, dim))
            if k == num_states-1:
                ax.legend()

    return fig, axes


def plot_svm_data_fit(observations, true_latent_vars=None,
        sampler=None, tqdm=None, N=10000,
        ignore_warning=False):
    """ Plot fit of SVM to the Data """
    if observations.shape[0] > 1000 and not ignore_warning:
        raise ValueError("PF inference for observations > 1000 is slow")
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(observations, 'oC0', label='data')
    axes[0].set_ylabel('raw observations')
    axes[1].plot(np.log(observations**2)-np.mean(np.log(observations**2)), 'oC0',
            label='log(data^2) - mean(log(data^2))')
    axes[1].set_ylabel('log(observations^2)')

    if true_latent_vars is not None:
        axes[1].plot(true_latent_vars, '-C1', label='latent_var')

    if sampler is not None:
        from sgmcmc_ssm.models.svm import SVMSampler
        if not isinstance(sampler, SVMSampler):
            raise ValueError("sampler must be an SVMSampler")
        smoothed_mean, smoothed_var = sampler.message_helper.pf_latent_var_marginal(
                observations, sampler.parameters, N=N, tqdm=tqdm)
        axes[1].plot(smoothed_mean[:,0], '-C2', label='PF E[X|Y] +/- SD(X|Y)')
        axes[1].plot(smoothed_mean[:,0]+np.sqrt(smoothed_var[:,0,0]),'--C2')
        axes[1].plot(smoothed_mean[:,0]-np.sqrt(smoothed_var[:,0,0]),'--C2')

    axes[0].legend()
    axes[1].legend()
    return fig, axes

def plot_garch_data_fit(observations, true_latent_vars=None,
        sampler=None, tqdm=None, N=10000,
        ignore_warning=False):
    """ Plot fit of GARCH to the Data """
    if observations.shape[0] > 1000 and not ignore_warning:
        raise ValueError("PF inference for observations > 1000 is slow")
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(observations, 'oC0', label='y_t')
    axes[0].set_ylabel('observations')
    axes[1].plot(observations**2, 'oC0',
            label='y_t^2')
    axes[1].set_ylabel('observations^2')

    if true_latent_vars is not None:
        axes[0].plot(true_latent_vars, '-C1', label='x_t')
        axes[1].plot(true_latent_vars**2, '-C1', label='x_t^2')

    if sampler is not None:
        from sgmcmc_ssm.models.garch import GARCHSampler
        if not isinstance(sampler, GARCHSampler):
            raise ValueError("sampler must be an GARCHSampler")
        smoothed_mean, smoothed_var = sampler.message_helper.pf_latent_var_marginal(
                observations, sampler.parameters, N=N, tqdm=tqdm)
        axes[0].plot(smoothed_mean[:,0], '-C2', label='PF E[X|Y] +/- SD(X|Y)')
        axes[0].plot(smoothed_mean[:,0]+np.sqrt(smoothed_var[:,0,0]),'--C2')
        axes[0].plot(smoothed_mean[:,0]-np.sqrt(smoothed_var[:,0,0]),'--C2')
        axes[1].plot(smoothed_mean[:,0]**2, '-C2', label='PF E[X|Y]**2')

    axes[0].legend()
    axes[1].legend()
    return fig, axes







