import numpy as np
import pandas as pd
import logging
import time
logger = logging.getLogger(name=__name__)

class SamplerEvaluator(object):
    """ Wrapper to handle measuring a Sampler's performance

    Args:
        sampler (Sampler): the sampler
        metric_functions (func or list of funcs): evaluation functions
            Each function takes a sampler and returns a dict (or list of dict)
                {metric, variable, value} for each
            (See metric_functions)
        sample_functions (func or list of funcs, optional): samples to save
            Each function takes a sampler and returns a dict (or list of dict)
                "variable", "value"
        experiment_name (string, optional): name for experiment
        sampler_name (string, optional): name for sampler
        data_name (string, optional): name for data
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

    Methods:
        load_state(self, metrics_df, samples_df, iteration)
        get_state(self)
        get_metrics(self, **extra_columns)
        get_samples(self, **extra_columns)
        reset_metrics(self)
        reset_samples(self)
        save_metrics(self, filename)
        save_samples(self, filename)
        evaluate_sampler_step(self, sampler_func_name, sampler_func_kwargs)
        eval_metric_functions(self, metric_functions)
        eval_sample_functions(self, sample_functions)

    Example:
        metric_functions = [
            metric_func_from_parameter("A", true_A, 'mse'),
            metric_func_from_sampler(sampler_func_name = "exact_loglikelihood"),
                          ]
        sample_functions = {
            'A': lambda sampler: sampler.parameter.A,
            'LQinv': lambda sampler: sampler.parameter.LQinv,
            }
        TopicModelEvaluater(data, sampler, metric_functions, sample_functions)


    """
    def __init__(self, sampler,
            metric_functions = None, sample_functions = None,
            experiment_name = None, sampler_name = None, data_name = None,
            init_state=None):
        self.sampler = sampler

        # Check metric functions
        metric_function = self._process_metric_functions(metric_functions)
        self.metric_functions = metric_functions

        # Check sample functions
        sample_functions = self._process_sample_functions(sample_functions)
        self.sample_functions = sample_functions

        self.experiment_name = experiment_name
        self.sampler_name = sampler_name
        self.data_name = data_name

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

    def _init_metrics(self):
        self.metrics = pd.DataFrame()
        self.eval_metric_functions()
        init_metric = {
            "variable": "runtime",
            "metric": "runtime",
            "value": 0.0,
            "iteration": self.iteration,
            }
        self.metrics = self.metrics.append(init_metric, ignore_index = True)
        return

    def get_metrics(self, extra_columns={}):
        """ Return a pd.DataFrame copy of metrics

        Args:
            extra_columns (dict): extra metadata to add as columns
        Returns:
            pd.DataFrame with columns
                metric, variable, value, iteration, sampler, data, extra_columns

        """
        metrics = self.metrics.copy()
        if self.experiment_name is not None:
            metrics["experiment"] = self.experiment_name
        if self.sampler_name is not None:
            metrics["sampler"] = self.sampler_name
        if self.data_name is not None:
            metrics["data"] = self.data_name
        for k,v in extra_columns:
            metrics[k] = v
        return metrics

    def reset_metrics(self):
        """ Reset self.metrics """
        logger.info("Resetting metrics")
        self.iteration = 0
        self.elapsed_time = 0.0
        self._init_metrics()
        return

    def save_metrics(self, filename, extra_columns = {}):
        """ Save a pd.DataFrame to filename + '.csv' """
        metrics = self.get_metrics(extra_columns)

        logger.info("Saving metrics to file %s", filename)
        metrics.to_csv(filename + ".csv", index = False)
        return

    def evaluate_sampler_step(self, sampler_func_name,
            sampler_func_kwargs = None, evaluate = True):
        """ Evaluate the performance of the sampler steps

        Args:
            sampler_func_name (string or list of strings):
                name(s) of sampler member functions
                (e.g. `'sample_sgld'` or `['sample_sgld']*10`)
            sampler_func_kwargs (kwargs or list of kwargs):
                options to pass to sampler_func_name
            evaluate (bool): whether to perform evaluation (default = True)

        Returns:
            out (ouptput of sampler_func_name)

        """
        logger.info("Sampler %s, Iteration %d",
                self.sampler_name, self.iteration+1)

        # Single Function
        if isinstance(sampler_func_name, str):
            sampler_func = getattr(self.sampler, sampler_func_name, None)
            if sampler_func is None:
                raise ValueError(
                    "sampler_func_name `{}` is not in sampler".format(
                            sampler_func_name)
                        )
            if sampler_func_kwargs is None:
                sampler_func_kwargs = {}

            sampler_start_time = time.time()
            out = sampler_func(**sampler_func_kwargs)
            sampler_step_time = time.time() - sampler_start_time

        # Multiple Steps
        elif isinstance(sampler_func_name, list):
            sampler_funcs = [getattr(self.sampler, func_name, None)
                    for func_name in sampler_func_name]
            if None in sampler_funcs:
                raise ValueError("Invalid sampler_func_name")

            if sampler_func_kwargs is None:
                sampler_func_kwargs = [{} for _ in sampler_funcs]
            if not isinstance(sampler_func_kwargs, list):
                raise TypeError("sampler_func_kwargs must be a list of dicts")
            if len(sampler_func_kwargs) != len(sampler_func_name):
                raise ValueError("sampler_func_kwargs must be same length " +
                    "as sampler_func_name")
            sampler_start_time = time.time()
            out = []
            for sampler_func, kwargs in zip(sampler_funcs, sampler_func_kwargs):
                out.append(sampler_func(**kwargs))
            sampler_step_time = time.time() - sampler_start_time

        else:
            raise TypeError("Invalid sampler_func_name")

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
            self.metrics = self.metrics.append(time_metric, ignore_index = True)
            self.eval_metric_functions()

            # Save Samples
            if self.sample_functions is not None:
                self.eval_sample_functions()

        return out

    def eval_metric_functions(self, metric_functions = None):
        """ Evaluate the state of the sampler

        Args:
           metric_functions (list of funcs): evaluation functions
            Defaults to metric functions defined in __init__

        """
        if metric_functions is None:
            metric_functions = self.metric_functions
        else:
            metric_function = self._process_metric_functions(metric_functions)

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
        iter_metrics["iteration"] = self.iteration

        self.metrics = self.metrics.append(iter_metrics, ignore_index = True)
        return

    def _init_samples(self):
        self.samples = pd.DataFrame()
        self.eval_sample_functions()
        return

    def get_samples(self, extra_columns = {}):
        """ Return a pd.DataFrame of samples """
        if self.sample_functions is None:
            logger.warning("No sample functions were provided to track!!!")
        samples = self.samples.copy()

        if self.experiment_name is not None:
            samples["experiment"] = self.experiment_name
        if self.sampler_name is not None:
            samples["sampler"] = self.sampler_name
        if self.data_name is not None:
            samples["data"] = self.data_name
        for k,v in extra_columns:
            samples[k] = v

        return samples

    def reset_samples(self):
        """ Reset self.samples """
        logger.info("Resetting samples")
        self._init_samples()
        return

    def save_samples(self, filename):
        """ Save a pd.DataFrame to filename + '.csv' """
        samples = self.get_samples()

        logger.info("Saving samples to file %s", filename)
        samples.to_csv(filename + ".csv", index = False)
        return

    def eval_sample_functions(self, sample_functions = None):
        """ Extract samples from current state of sampler

            Args:
                sample_functions (list of funcs): sample functions
                    Defaults to sample functions defined in __init__

        """
        if sample_functions is None:
            sample_functions = self.sample_functions
        else:
            sample_functions = self._process_sample_functions(sample_functions)
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
        iter_sample["iteration"] = self.iteration

        self.samples = self.samples.append(iter_sample, ignore_index = True)
        return


