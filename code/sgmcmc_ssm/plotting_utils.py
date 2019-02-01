import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import logging
logger = logging.getLogger(name=__name__)

def plot_metrics(evaluator, full_trace=True, burnin=None):
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
                plt.plot, 'iteration', 'value'
                ).add_legend().set_titles("{col_name}")
    return g

def compare_metrics(evaluators, full_trace=True, burnin=None, errorband=False):
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
    if burnin is None:
        if not full_trace:
            min_iteration = min([evaluator.iteration
                for evaluator in evaluators.values()])
            df = df.query('iteration >= {0}'.format(min_iteration/2))
    else:
        df = df.query('iteration >= {0}'.format(burnin))


    df = df.sort_values(['method', 'variable:metric', 'iteration'])
    if errorband:
        g = sns.FacetGrid(df, col='variable:metric', hue="method",
            col_wrap=4, sharey=False).map_dataframe(
                sns.lineplot, x='iteration', y='value',
                estimator='mean', ci='sd',
                )
    else:
        g = sns.FacetGrid(df, col='variable:metric', hue="method",
            col_wrap=4, sharey=False).map_dataframe(
                sns.lineplot, x='iteration', y='value',
                units='init', estimator=None, ci=None,
                )
    g = g.add_legend().set_titles("{col_name}").set_xlabels("iteration")
    return g

def plot_trace_plot(evaluator, full_trace=True, query_string=None,
        single_variables=[], burnin=None):
    samples_df = evaluator.get_samples()
    if burnin is None:
        if not full_trace:
            samples_df = samples_df.query('iteration >= {0}'.format(evaluator.iteration/2))
    else:
        samples_df = samples_df.query('iteration >= {0}'.format(burnin))
    if query_string is not None:
        samples_df = samples_df.query(query_string)
    variables = samples_df['variable'].sort_values().unique()
    iterations = samples_df['iteration'].sort_values().unique()
    variable_values = {
            key: np.array(df.sort_values('iteration')['value'].tolist())
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
                    ax.plot(iterations, values[:, dim],
                        label='{0}[{1}]'.format(variable, dim))
            else:
                values = np.reshape(values, (values.shape[0], num_states, -1))
                for dim in range(values.shape[2]):
                    ax.plot(iterations, values[:, k, dim],
                            label='{0}[{1}]'.format(variable, dim))
            if k == num_states-1:
                ax.legend()

    return fig, axes

