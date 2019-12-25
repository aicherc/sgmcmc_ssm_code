import numpy as np
import pandas as pd
import logging
import time
logger = logging.getLogger(name=__name__)

class BaseEvaluator(object):
    """ Evaluator Base Class """
    def __init__(self, sampler, metric_functions=None, sample_functions=None):
        # Set Sampler
        self.sampler = sampler

        # Check metric functions
        metric_functions = self._process_metric_functions(metric_functions)
        self.metric_functions = metric_functions

        # Check sample functions
        sample_functions = self._process_sample_functions(sample_functions)
        self.sample_functions = sample_functions

        # Base Init
        self.metrics = pd.DataFrame()
        self.samples = pd.DataFrame()

        return

    @staticmethod
    def _process_metric_functions(metric_functions):
        if callable(metric_functions):
            metric_functions = [metric_functions]

        elif isinstance(metric_functions, list):
            for metric_function in metric_functions:
                if not callable(metric_function):
                    raise ValueError("metric_functions must be list of funcs")
        elif metric_functions is None:
            metric_functions = []
        else:
            ValueError("metric_functions should be list of funcs")
        return metric_functions

    @staticmethod
    def _process_sample_functions(sample_functions):
        if callable(sample_functions):
            sample_functions = [sample_functions]

        elif isinstance(sample_functions, list):
            for sample_function in sample_functions:
                if not callable(sample_function):
                    raise ValueError("sample_functions must be list of funcs")
        elif sample_functions is None:
            sample_functions = []
        else:
            ValueError("sample_functions should be list of funcs")
        return sample_functions

    def eval_sample_functions(self, sample_functions = None, iteration = None):
        """ Extract samples from current state of sampler

            Args:
                sample_functions (list of funcs): sample functions
                    Defaults to sample functions defined in __init__

        """
        if sample_functions is None:
            sample_functions = self.sample_functions
        else:
            sample_functions = self._process_sample_functions(sample_functions)
        if iteration is None:
            iteration = self.iteration

        if len(sample_functions) == 0:
            # Skip if no sample_functions
            return

        iter_samples = []
        for sample_function in sample_functions:
            sample = sample_function(self.sampler)
            if isinstance(sample, dict):
                logger.debug("Sample: %s", str(sample))
                iter_samples.append(sample)
            elif isinstance(sample, list):
                for sam in sample:
                    if not isinstance(sam, dict):
                        raise TypeError("Sample function output must be " + \
                        "dict or list of dict")
                    logger.debug("Sample: %s", str(sam))
                    iter_samples.append(sam)
            else:
                raise TypeError("sample_functions output must be " + \
                        "dict or list of dict")

        iter_sample = pd.DataFrame(iter_samples)
        iter_sample["iteration"] = iteration

        self.samples = self.samples.append(iter_sample,
                ignore_index=True, sort=True)
        return

    def eval_metric_functions(self, metric_functions = None, iteration = None):
        """ Evaluate the state of the sampler

        Args:
           metric_functions (list of funcs): evaluation functions
            Defaults to metric functions defined in __init__

        """
        if metric_functions is None:
            metric_functions = self.metric_functions
        else:
            metric_functions = self._process_metric_functions(metric_functions)

        if iteration is None:
            iteration = self.iteration

        if len(metric_functions) == 0:
            # Skip if no metric_functions
            return

        iter_metrics = []
        for metric_function in metric_functions:
            metric = metric_function(self.sampler)
            if isinstance(metric, dict):
                logger.info("Metric: %s", str(metric))
                iter_metrics.append(metric)
            elif isinstance(metric, list):
                for met in metric:
                    if not isinstance(met, dict):
                        raise TypeError("Metric must be dict or list of dict")
                    logger.info("Metric: %s", str(met))
                    iter_metrics.append(met)
            else:
                raise TypeError("Metric must be dict or list of dict")

        iter_metrics = pd.DataFrame(iter_metrics)
        iter_metrics["iteration"] = iteration

        self.metrics = self.metrics.append(iter_metrics,
                ignore_index=True, sort=True)
        return

    def get_metrics(self, extra_columns={}):
        """ Return a pd.DataFrame copy of metrics

        Args:
            extra_columns (dict): extra metadata to add as columns
        Returns:
            pd.DataFrame with columns
                metric, variable, value, iteration, extra_columns

        """
        metrics = self.metrics.copy()
        for k,v in extra_columns.items():
            metrics[k] = v
        return metrics

    def save_metrics(self, filename, extra_columns = {}):
        """ Save a pd.DataFrame to filename + '.csv' """
        metrics = self.get_metrics(extra_columns)

        logger.info("Saving metrics to file %s", filename)
        metrics.to_csv(filename + ".csv", index = False)
        return

    def get_samples(self, extra_columns = {}):
        """ Return a pd.DataFrame of samples """
        if self.sample_functions is None:
            logger.warning("No sample functions were provided to track!!!")
        samples = self.samples.copy()
        for k,v in extra_columns.items():
            samples[k] = v
        return samples

    def save_samples(self, filename):
        """ Save a pd.DataFrame to filename + '.csv' """
        samples = self.get_samples()

        logger.info("Saving samples to file %s", filename)
        samples.to_csv(filename + ".csv", index = False)
        return

# Old Evaluator
class SamplerEvaluator(BaseEvaluator):
    """ Wrapper to handle measuring a Sampler's Performance Online

    Args:
        sampler (Sampler): the sampler
        metric_functions (func or list of funcs): evaluation functions
            Each function takes a sampler and returns a dict (or list of dict)
                {metric, variable, value} for each
            (See metric_functions)
        sample_functions (func or list of funcs, optional): samples to save
            Each function takes a sampler and returns a dict (or list of dict)
                "variable", "value"
        init_state (dict, optional): state of evaluater
            must contain metrics_df, samples_df, and iteration

    Attributes:
        metrics (pd.DataFrame): output data frame with columns
            * metric (string)
            * variable (string)
            * value (double)
            * iteration (int)

        samples (pd.DataFrame): sampled params with columns
            * variable (string)
            * value (object)
            * iteration (int)
    """
    def __init__(self, sampler,
            metric_functions = None, sample_functions = None,
            init_state=None, **kwargs):
        self.sampler = sampler

        # Check metric functions
        metric_functions = self._process_metric_functions(metric_functions)
        self.metric_functions = metric_functions

        # Check sample functions
        sample_functions = self._process_sample_functions(sample_functions)
        self.sample_functions = sample_functions

        if init_state is None:
            self.iteration = 0
            self.elapsed_time = 0.0
            self._init_metrics()
            self._init_samples()
        else:
            self.load_state(**init_state)

        return

    def load_state(self, metrics_df, samples_df, iteration, elapsed_time):
        """ Overwrite metrics, samples, and iteration """
        if metrics_df.shape[0] > 0:
            self.metrics = metrics_df[
                    ['iteration', 'metric', 'variable', 'value']]
        else:
            self.metrics = metrics_df
        if samples_df.shape[0] > 0:
            self.samples = samples_df[['iteration', 'variable', 'value']]
        else:
            self.samples = samples_df
        self.iteration = iteration
        self.elapsed_time = elapsed_time
        return

    def get_state(self):
        """ Return dict with metrics_df, samples_df, iteration """
        state = dict(
                metrics_df = self.metrics,
                samples_df = self.samples,
                iteration = self.iteration,
                elapsed_time = self.elapsed_time,
                )
        return state

    def _init_metrics(self):
        self.metrics = pd.DataFrame()
        self.eval_metric_functions()
        init_metric = {
            "variable": "runtime",
            "metric": "runtime",
            "value": 0.0,
            "iteration": self.iteration,
            }
        self.metrics = self.metrics.append(init_metric,
                ignore_index=True, sort=True)
        return

    def reset_metrics(self):
        """ Reset self.metrics """
        logger.info("Resetting metrics")
        self.iteration = 0
        self.elapsed_time = 0.0
        self._init_metrics()
        return

    def _init_samples(self):
        self.samples = pd.DataFrame()
        self.eval_sample_functions()
        return

    def reset_samples(self):
        """ Reset self.samples """
        logger.info("Resetting samples")
        self._init_samples()
        return

    def evaluate_sampler_step(self, iter_func_name,
            iter_func_kwargs = None, evaluate = True):
        """ Evaluate the performance of the sampler steps

        Args:
            iter_func_name (string or list of strings):
                name(s) of sampler member functions
                (e.g. `'sample_sgld'` or `['sample_sgld']*10`)
            iter_func_kwargs (kwargs or list of kwargs):
                options to pass to iter_func_name
            evaluate (bool): whether to perform evaluation (default = True)

        Returns:
            out (ouptput of iter_func_name)

        """
        logger.info("Sampler %s, Iteration %d",
                self.sampler.name, self.iteration+1)

        # Single Function
        if isinstance(iter_func_name, str):
            iter_func = getattr(self.sampler, iter_func_name, None)
            if iter_func is None:
                raise ValueError(
                    "iter_func_name `{}` is not in sampler".format(
                            iter_func_name)
                        )
            if iter_func_kwargs is None:
                iter_func_kwargs = {}

            sampler_start_time = time.time()
            out = iter_func(**iter_func_kwargs)
            sampler_step_time = time.time() - sampler_start_time

        # Multiple Steps
        elif isinstance(iter_func_name, list):
            iter_funcs = [getattr(self.sampler, func_name, None)
                    for func_name in iter_func_name]
            if None in iter_funcs:
                raise ValueError("Invalid iter_func_name")

            if iter_func_kwargs is None:
                iter_func_kwargs = [{} for _ in iter_funcs]
            if not isinstance(iter_func_kwargs, list):
                raise TypeError("iter_func_kwargs must be a list of dicts")
            if len(iter_func_kwargs) != len(iter_func_name):
                raise ValueError("iter_func_kwargs must be same length " +
                    "as iter_func_name")
            sampler_start_time = time.time()
            out = []
            for iter_func, kwargs in zip(iter_funcs, iter_func_kwargs):
                out.append(iter_func(**kwargs))
            sampler_step_time = time.time() - sampler_start_time

        else:
            raise TypeError("Invalid iter_func_name")

        self.iteration += 1
        self.elapsed_time += sampler_step_time
        time_metric = [{
            "variable": "time",
            "metric": "time",
            "value": sampler_step_time,
            "iteration": self.iteration,
            },
            {
            "variable": "runtime",
            "metric": "runtime",
            "value": self.elapsed_time,
            "iteration": self.iteration,
            }]

        if evaluate:
            # Save Metrics
            self.metrics = self.metrics.append(time_metric,
                    ignore_index=True, sort=True)
            self.eval_metric_functions()

            # Save Samples
            if self.sample_functions is not None:
                self.eval_sample_functions()

        return out

# Offline Evaluator
class OfflineEvaluator(BaseEvaluator):
    """ Wrapper to handle measuring a Sampler's Performance Offline

    Args:
        sampler (Sampler): the sampler
        parameters_list (list or DataFrame): list of parameters to evaluate offline
        parameters_times (list or ndarray, optional): fit time for parameters
            Should be same size as parameters_list
            Time is in seconds
        metric_functions (func or list of funcs, optional): evaluation functions
            Each function takes a sampler and returns a dict (or list of dict)
                {metric, variable, value} for each
            (See metric_functions)
        sample_functions (func or list of funcs, optional): samples to save
            Each function takes a sampler and returns a dict (or list of dict)
                "variable", "value"
        init_state (dict, optional): state of evaluater
            must contain metrics_df, samples_df, and eval_flag

    Attributes:
        metrics (pd.DataFrame): output data frame with columns
            * metric (string)
            * variable (string)
            * value (double)
            * iteration (int)

        samples (pd.DataFrame): sampled params with columns
            * variable (string)
            * value (object)
            * iteration (int)
    """
    def __init__(self, sampler, parameters_list, parameters_times=None,
            metric_functions = None, sample_functions = None,
            init_state=None):
        # Set Sampler
        self.sampler = sampler

        # Set parameter list
        self.parameters_list = self._process_parameters_list(parameters_list)
        self.iteration = np.max(self.parameters_list['iteration'])
        self.parameters_times = self._process_parameters_times(parameters_times)
        if parameters_times is not None:
            if self.parameters_times.shape[0] != self.parameters_list.shape[0]:
                raise ValueError(
            "parameters_times and parameters_list are not the same length"
            )

        # Check metric functions
        metric_functions = self._process_metric_functions(metric_functions)
        self.metric_functions = metric_functions

        # Check sample functions
        sample_functions = self._process_sample_functions(sample_functions)
        self.sample_functions = sample_functions

        if init_state is None:
            self.eval_flag = pd.DataFrame(dict(
                iteration = self.parameters_list['iteration'],
                eval_flag = [False
                    for _ in range(self.parameters_list.shape[0])],
                ))
            self.metrics = pd.DataFrame()
            self.samples = pd.DataFrame()
        else:
            self.load_state(**init_state)
        return

    @staticmethod
    def _process_parameters_list(parameters_list):
        if isinstance(parameters_list, pd.DataFrame):
            if 'iteration' not in parameters_list.columns:
                raise ValueError("`iteration` not found in parameters_list.columns")
            if 'parameters' not in parameters_list.columns:
                raise ValueError("`parameters` not found in parameters_list.columns")
            return parameters_list.sort_values(by='iteration')
        elif isinstance(parameters_list, list):
            iteration = np.arange(len(parameters_list))
            parameters_list = pd.DataFrame(dict(
                iteration = iteration,
                parameters = parameters_list,
                ))
            return parameters_list
        else:
            raise ValueError("parameters_list is not a list or pd.DataFrame")

    @staticmethod
    def _process_parameters_times(parameters_times):
        if parameters_times is None:
            return None
        elif isinstance(parameters_times, pd.DataFrame):
            if 'iteration' not in parameters_times.columns:
                raise ValueError("`iteration` not found in parameters_times")
            if 'time' not in parameters_times.columns:
                raise ValueError("`time` not found in parameters_times")
            return parameters_times.sort_values(by='iteration')
        elif isinstance(parameters_times, list) or \
                isinstance(parameters_times, np.ndarray):
            iteration = np.arange(len(parameters_times))
            parameters_times = pd.DataFrame(dict(
                iteration = iteration,
                time = parameters_times,
                ))
            return parameters_times
        else:
            raise ValueError("parameters_times is not a list or pd.DataFrame")

    def load_state(self, metrics_df, samples_df, eval_flag):
        """ Overwrite metrics, samples, and iteration """
        if metrics_df.shape[0] > 0:
            self.metrics = metrics_df[
                    ['iteration', 'metric', 'variable', 'value']]
        else:
            self.metrics = metrics_df
        if samples_df.shape[0] > 0:
            self.samples = samples_df[['iteration', 'variable', 'value']]
        else:
            self.samples = samples_df

        if eval_flag.shape[0] == self.parameters_list.shape[0]:
            self.eval_flag = eval_flag
        elif eval_flag.shape[0] < self.parameters_list.shape[0]:
            self.eval_flag = pd.DataFrame(dict(
                iteration = self.parameters_list['iteration'],
                eval_flag = [False
                    for _ in range(self.parameters_list.shape[0])],
                ))
            for row_index, row in eval_flag.iterrows():
                if row['iteration'] == True:
                    self.eval_flag.iloc[row_index, 1] = row['eval_flag']
                else:
                    self.eval_flag.loc[
                            self.eval_flag['iteration'] == row['iteration'],
                            'eval_flag'] = row['eval_flag']
        else:
            raise ValueError("eval_flag + parameters_list do not match lengths")

        return

    def get_state(self):
        """ Return dict with metrics_df, samples_df, eval_flag """
        state = dict(
                metrics_df = self.metrics,
                samples_df = self.samples,
                eval_flag = self.eval_flag,
                )
        return state

    def num_to_eval(self):
        return self.eval_flag.shape[0] - self.eval_flag.eval_flag.sum()

    def evaluate(self, num_to_eval=None, max_time=None, eval_order="recursive",
            iter_func_name=None, iter_func_kwargs=None,
            tqdm=None):
        """ Evaluate the parameters in parameters_list

        Evaluate both metric_funcs + iter_funcs

        Args:
            num_to_eval (int): number of parameters to evaluate
                (default is to evaluate all)
            max_time (double): max time to evaluate in seconds
                (default is inf)
            eval_order (string): order to evaluate parameters
                (that haven't been evaluated)
                "recursive": evaluate first, last, and then recursively bisect
                "sequential": evaluate first, second, third
            iter_func_name (string or list of strings):
                functions to call before evaluation (after setting parameters)
            iter_func_kwargs (kwargs or list of kwargs):
                options to pass to iter_func_name

        Return:


        """
        if num_to_eval is None:
            num_to_eval = self.num_to_eval()
        if max_time is None:
            max_time = np.inf

        pbar = range(min(num_to_eval, self.num_to_eval()))
        if tqdm is not None:
            pbar = tqdm(pbar, desc="offline eval")

        eval_start_time = time.time()
        for _ in pbar:
            if self.eval_flag.eval_flag.sum() == self.eval_flag.shape[0]:
                logging.warning("No more parameters to evaluate on")
                return

            iteration = self._get_eval_iteration(eval_order)
            self.sampler.parameters = self.parameters_list[
                    self.parameters_list['iteration'] == iteration
                    ]['parameters'].iloc[0].copy()

            if tqdm is not None:
                pbar.set_description("offline eval on iteration {}".format(
                    iteration))
            logger.info("Sampler %s, Iteration %d",
                self.sampler.name, iteration)

            # Call Sampler Func
            if iter_func_name is None:
                pass

            elif isinstance(iter_func_name, str):
                # Single Functions
                iter_func = getattr(self.sampler, iter_func_name, None)
                if iter_func is None:
                    raise ValueError(
                        "iter_func_name `{}` is not in sampler".format(
                                iter_func_name)
                            )
                if iter_func_kwargs is None:
                    iter_func_kwargs = {}
                iter_func(**iter_func_kwargs)

            elif isinstance(iter_func_name, list):
                # Multiple Functions
                iter_funcs = [getattr(self.sampler, func_name, None)
                        for func_name in iter_func_name]
                if None in iter_funcs:
                    raise ValueError("Invalid iter_func_name")
                if iter_func_kwargs is None:
                    iter_func_kwargs = [{} for _ in iter_funcs]
                if not isinstance(iter_func_kwargs, list):
                    raise TypeError("iter_func_kwargs must be a list of dicts")
                if len(iter_func_kwargs) != len(iter_func_name):
                    raise ValueError("iter_func_kwargs must be same length " +
                        "as iter_func_name")
                for iter_func, kwargs in zip(iter_funcs, iter_func_kwargs):
                    iter_func(**kwargs)
            else:
                raise TypeError("Invalid iter_func_name")

            # Evaluate metrics + samples
            self.eval_metric_functions(iteration=iteration)
            self.eval_sample_functions(iteration=iteration)

            # Mark Iteration as sampled
            self.eval_flag.loc[
                    self.eval_flag.iteration == iteration, 'eval_flag'] = True

            # Early Termination on Max Time
            if time.time() - eval_start_time > max_time:
                break
        return

    def _get_eval_iteration(self, eval_order="recursive"):
        # Get next iteration in parameters_list to evaluate
        if not self.eval_flag.eval_flag.iloc[0]:
            # Always evaluate first iteration
            return self.eval_flag.iteration.iloc[0]

        if eval_order == "sequential":
            index = np.argmax(self.eval_flag.eval_flag.ne(True).values)
            iteration = self.eval_flag.iteration.iloc[index]
        elif eval_order == "recursive":
            num_indices = self.eval_flag.shape[0]
            evaled_indices = np.arange(num_indices, dtype=int)[
                    self.eval_flag.eval_flag]
            eval_gaps = np.pad(evaled_indices, ((0, 1)),
                    mode='constant',
                    constant_values=num_indices-1,
                    )[1:] - evaled_indices
            eval_gaps[-1] *= 2   # Double last diff
            max_gap_index = np.argmax(eval_gaps)
            index = (
                evaled_indices[max_gap_index] -
                (-eval_gaps[max_gap_index]//2) #Ceil
                )
            iteration = self.eval_flag.iteration.iloc[index]
        else:
            raise ValueError("Unrecoginized eval_order {0}".format(eval_order))
        return iteration

    @staticmethod
    def _add_time(df, times):
        """ Append times to df by matching on iteration """
        df_times = pd.merge(df, times, on='iteration')
        return df_times

    def get_metrics(self, extra_columns={}):
        metrics = super().get_metrics(extra_columns=extra_columns)
        if self.parameters_times is not None:
            metrics = self._add_time(metrics, self.parameters_times)
        return metrics

    def get_samples(self, extra_columns={}):
        samples = super().get_samples(extra_columns=extra_columns)
        if self.parameters_times is not None:
            samples = self._add_time(samples, self.parameters_times)
        return samples


# Helper functions
def average_parameters_list(parameters_list, burnin=None):
    """ Return a running average of parameters after burnin

    theta_bar_t = (sum_{s <= t, s > burnin} theta_s)/(t+1) for t >= burnin
    theta_bar_t = theta_t for t < burnin

    Args:
        parameters_list (list or pd.Series): list of parameters
        burnin (int, optional): number of burnin step
            (default is 0.33 of parameters_list)
    Returns:
        averaged_parameters_list (list): list of averaged parameters
    """
    if isinstance(parameters_list, pd.Series):
        parameters_list = parameters_list.tolist()
    if len(parameters_list) == 0:
        return
    Parameters = type(parameters_list[0])
    parameters_dim = parameters_list[0].dim

    if burnin is None:
        burnin = int(len(parameters_list)*0.33)

    parameters_vectors = [None] * len(parameters_list)
    for ii, parameters in enumerate(parameters_list):
        parameters_vectors[ii] = parameters.as_vector()

    parameters_vectors = np.array(parameters_vectors)
    parameters_vectors[burnin:] = (
            np.cumsum(parameters_vectors[burnin:], axis=0) /
            np.arange(1,parameters_vectors.shape[0]-burnin+1)[:,None]
            )

    averaged_parameters_list = [None] * len(parameters_list)
    for ii, vector in enumerate(parameters_vectors):
        averaged_parameters_list[ii] = Parameters(
                **Parameters.from_vector_to_dict(vector, **parameters_dim)
                )
    return averaged_parameters_list

def half_average_parameters_list(parameters_list):
    """ Return a running average of last half parameters

    theta_bar_t = mean_{t/2 <= s <= t} theta_s

    Args:
        parameters_list (list or pd.Series): list of parameters
        burnin (int, optional): number of burnin step
            (default is 0.33 of parameters_list)
    Returns:
        half_averaged_parameters_list (list): list of averaged parameters
    """
    if isinstance(parameters_list, pd.Series):
        parameters_list = parameters_list.tolist()
    if len(parameters_list) == 0:
        return
    Parameters = type(parameters_list[-1])
    parameters_dim = parameters_list[-1].dim

    parameters_vectors = [None] * len(parameters_list)
    for ii, parameters in enumerate(parameters_list):
        parameters_vectors[ii] = parameters.as_vector()

    parameters_vectors = np.array(parameters_vectors)

    parameters_vectors = np.array([
        np.mean(parameters_vectors[ii//2:ii+1,:], axis=0)
            for ii in range(parameters_vectors.shape[0])
        ])

    half_averaged_parameters_list = [None] * len(parameters_list)
    for ii, vector in enumerate(parameters_vectors):
        half_averaged_parameters_list[ii] = Parameters(
                **Parameters.from_vector_to_dict(vector, **parameters_dim)
                )
    return half_averaged_parameters_list


