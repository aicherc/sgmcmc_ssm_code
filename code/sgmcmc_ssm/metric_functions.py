"""

Sampler Metric Function Generators

"""
import numpy as np

def sample_function_parameter(parameter_name, return_variable_name=None):
    """ Returns sample function that extracts a parameter from current state

    Args:
        parameter_name (string): atrribute in sampler.parameters
            (e.g. A, C, LRinv, R)
        return_variable_name (string, optional): name of return name
            default is `parameter_name`

    Returns:
        A function of sampler that returns dictionary of variable, value

    """
    if return_variable_name is None:
        return_variable_name = parameter_name

    def custom_sample_function(sampler):
        cur_parameter = np.copy(getattr(sampler.parameters, parameter_name))
        sample = {'variable': return_variable_name,
                  'value': cur_parameter,
                  }
        return sample
    return custom_sample_function

def sample_function_parameters(parameter_names, return_variable_names=None,
        decorator=None):
    """ Returns sample function that extracts a parameter from current state

    Args:
        parameter_names (list of string): atrributes in sampler.parameters
            (e.g. A, C, LRinv, R)
        return_variable_name (list of string, optional): names of return name
            default is `parameter_name`
        decorator (func, optional): function decorator

    Returns:
        A function of sampler that returns list of dictionary of variable, value

    """
    if return_variable_names is None:
        return_variable_names = [None]*len(parameter_names)
    elif len(return_variable_names) != len(parameter_names):
        raise ValueError("parameter and return names must be equal length")

    sample_functions = [
            sample_function_parameter(parameter_name, return_variable_name)
            for parameter_name, return_variable_name in zip(
                parameter_names, return_variable_names)
            ]

    def custom_sample_function(sampler):
        samples = [sample_function(sampler)
                for sample_function in sample_functions]
        return samples

    if decorator is not None:
        custom_sample_function = decorator(custom_sample_function)

    return custom_sample_function

def metric_function_parameter(parameter_name, target_value, metric_name,
        return_variable_name=None):
    """ Returns metric function that compares samplers' state to a target

    Args:
        parameter_name (string): atrribute in sampler.parameters
            (e.g. A, C, LRinv, R)
        target_value (np.ndarray): target value
        metric_name (string): name of a metric function
            * 'logmse': log10 mean squared error
            * 'mse': mean squared error
            * 'mae': mean absolute error
            (see construct_metric_function)
        return_variable_name (string, optional): name of metric return name
            default is `parameter_name`

    Returns:
        A function of sampler that returns dictionary of variable, metric, value

    """
    metric_func = construct_metric_function(metric_name)
    if return_variable_name is None:
        return_variable_name = parameter_name

    def custom_metric_function(sampler):
        cur_parameter = getattr(sampler.parameters, parameter_name)
        metric_value = metric_func(cur_parameter, target_value)
        metric = {'variable': return_variable_name,
                  'metric': metric_name,
                  'value': metric_value
                  }
        return metric
    return custom_metric_function

def metric_function_parameters(parameter_names, target_values, metric_names,
        return_variable_names=None, decorator=None, criteria=None,
        double_permutation_flag=False):
    """ Returns sample function that extracts a parameter from current state

    Args:
        parameter_names (list of string): atrributes in sampler.parameters
            (e.g. A, C, LRinv, R)
        target_values (list of np.ndarray): target values
        metric_names (list of string): names of metric functions
            * 'logmse': log10 mean squared error
            * 'mse': mean squared error
            * 'mae': mean absolute error
            (see construct_metric_function)
        return_variable_names (list of string, optional): names of return name
            default is `parameter_name`
        decorator (func, optional): function decorator
        criteria (list of func or tuple, optional): e.g. [min] or None
        double_permutation_flag (bool, optional): default False
            Only for use with criteria for pi, expanded_pi, logit_pi vars

    Returns:
        A function of sampler that returns list of dictionaries of
            variable, metric, value

    """
    if (len(target_values) != len(parameter_names) or \
        len(metric_names) != len(parameter_names)):
        raise ValueError("input args not equal length")
    if return_variable_names is None:
        return_variable_names = [None]*len(parameter_names)
    elif len(return_variable_names) != len(parameter_names):
        raise ValueError("parameter and return names must be equal length")

    if criteria is None:
        metric_functions = [
            metric_function_parameter(parameter_name, target_value,
                metric_name, return_variable_name)
            for parameter_name, target_value, metric_name, return_variable_name\
                    in zip(
                parameter_names, target_values, metric_names,
                return_variable_names)
            ]
    else:
        if double_permutation_flag:
            metric_functions = [
                best_double_permutation_metric_function_parameter(
                    parameter_name, target_value, metric_name,
                    return_variable_name, best_function)
                for parameter_name, target_value, metric_name, \
                        return_variable_name, best_function \
                        in zip(
                    parameter_names, target_values, metric_names,
                    return_variable_names, criteria)
                ]
        else:
            metric_functions =[
                best_permutation_metric_function_parameter(parameter_name,
                    target_value, metric_name, return_variable_name,
                    best_function)
                for parameter_name, target_value, metric_name, \
                        return_variable_name, best_function \
                        in zip(
                    parameter_names, target_values, metric_names,
                    return_variable_names, criteria)
                ]

    def custom_metric_function(sampler):
        metrics = [metric_function(sampler)
                for metric_function in metric_functions]
        return metrics

    if decorator is not None:
        custom_metric_function = decorator(custom_metric_function)

    return custom_metric_function

def metric_function_from_sampler(sampler_func_name, metric_name=None,
        return_variable_name="sampler",
        **sampler_func_kwargs):
    """ Returns metric function that evaluates sampler_func_name

    Example:
        metric_function_of_sampler(sampler_func_name = "exact_loglikelihood")
    """
    if metric_name is None:
        metric_name = sampler_func_name

    def custom_metric_function(sampler):
        sampler_func = getattr(sampler, sampler_func_name, None)
        if sampler_func is None:
            raise ValueError(
                "sampler_func_name `{}` is not in sampler".format(
                        sampler_func_name)
                    )
        else:
            metric_value = sampler_func(**sampler_func_kwargs)
            metric = {'variable': return_variable_name,
                      'metric': metric_name,
                      'value': metric_value}
        return metric
    return custom_metric_function

def construct_metric_function(metric_name):
    """ Return a metric function

    Args:
        metric_name (string): name of metric. Must be one of
            * 'logmse': log10 mean squared error
            * 'mse': mean squared error
            * 'rmse': root mean squared error (L2 norm)
            * 'mae': mean absolute error

    Returns:
        metric_function (function):
            function of two inputs (result, expected)
    """
    if(metric_name == "mse"):
        def metric_function(result, expected):
            return np.mean((result - expected)**2)
    elif(metric_name == "logmse"):
        def metric_function(result, expected):
            return np.log10(np.mean((result - expected)**2))

    elif(metric_name == "rmse"):
        def metric_function(result, expected):
            return np.sqrt(np.mean((result - expected)**2))

    elif(metric_name == "mae"):
        def metric_function(result, expected):
            return np.mean(np.abs(result - expected))

    else:
        raise ValueError("Unrecognized metric name = %s" % metric_name)

    return metric_function

def average_input_decorator(sampler_function):
    """ Average over all calls to sampler_function """
    def average_function(sampler):
        average_function.num_calls += 1
        average_function.sum_vector += sampler.parameters.vector
        sampler_params = sampler.parameters.vector
        sampler.parameters.vector = (average_function.sum_vector / \
                average_function.num_calls)

        output = sampler_function(sampler)

        if isinstance(output, dict):
            output['variable'] = "avg_"+output['variable']
        else:
            for out in output:
                out['variable'] = "avg_"+out['variable']

        sampler.parameters.vector = sampler_params
        return output

    average_function.num_calls = 0
    average_function.sum_vector = 0.0
    return average_function

def best_permutation_metric_function_parameter(parameter_name, target_value,
        metric_name, return_variable_name=None, best_function=max):
    """ Select `best' metric across all permutations of first index
    Args:
        parameter_name (string): atrribute in sampler.parameters
            (e.g. A, Q) with sampler.num_states # different versions
        target_value (np.ndarray): target value
        metric_name (string): name of a metric function
            * 'mse': mean squared error
            * 'mae': mean absolute error
            (see construct_metric_function)
        return_variable_name (string, optional): name of metric return name
            default is `parameter_name`
        best_function (function, optional): takes list of double, return `best'
            default is max, (e.g. max, min)

    Returns:
        A function of sampler that returns dictionary of variable, metric, value

    """
    if best_function is None:
        return metric_function_parameter(parameter_name, target_value,
                metric_name, return_variable_name)
    if return_variable_name is None:
        return_variable_name = parameter_name

    metric_func = construct_metric_function(metric_name)
    if return_variable_name is None:
        return_variable_name = parameter_name

    import itertools
    def custom_metric_function(sampler):
        cur_parameter = getattr(sampler.parameters, parameter_name)
        metric_values = [
            metric_func(
                    cur_parameter,
                    target_value[np.array(permuted_indices)],
                )
            for permuted_indices in itertools.permutations(
                np.arange(target_value.shape[0]))
            ]
        metric_value = best_function(metric_values)
        metric = {'variable': return_variable_name,
                  'metric': metric_name,
                  'value': metric_value
                  }
        return metric
    return custom_metric_function

def best_double_permutation_metric_function_parameter(parameter_name,
        target_value, metric_name, return_variable_name=None, best_function=max):
    """ Select `best' metric across all permutations of first and second index
    Args:
        parameter_name (string): atrribute in sampler.parameters
            (e.g. pi) with sampler.num_states # different versions
        target_value (np.ndarray): target value
        metric_name (string): name of a metric function
            * 'mse': mean squared error
            * 'mae': mean absolute error
            (see construct_metric_function)
        return_variable_name (string, optional): name of metric return name
            default is `parameter_name`
        best_function (function, optional): takes list of double, return `best'
            default is max, (e.g. max, min)

    Returns:
        A function of sampler that returns dictionary of variable, metric, value

    """
    if best_function is None:
        return metric_function_parameter(parameter_name, target_value,
                metric_name, return_variable_name)
    if return_variable_name is None:
        return_variable_name = parameter_name

    metric_func = construct_metric_function(metric_name)
    if return_variable_name is None:
        return_variable_name = parameter_name

    import itertools
    def custom_metric_function(sampler):
        cur_parameter = getattr(sampler.parameters, parameter_name)
        metric_values = [
            metric_func(
                    cur_parameter,
                    target_value[np.array(permuted_indices)][:,
                        np.array(permuted_indices)],
                )
            for permuted_indices in itertools.permutations(
                np.arange(target_value.shape[0]))
            ]
        metric_value = best_function(metric_values)
        metric = {'variable': return_variable_name,
                  'metric': metric_name,
                  'value': metric_value
                  }
        return metric
    return custom_metric_function

def noisy_logjoint_loglike_metric(**kwargs):
    def custom_metric_func(sampler):
        res = sampler.noisy_logjoint(return_loglike=True, **kwargs)
        return [
            dict(
                variable='sampler',
                metric='noisy_logjoint',
                value=res['logjoint'],
                ),
            dict(
                variable='sampler',
                metric='noisy_loglikelihood',
                value=res['loglikelihood'],
                ),
            ]
    return custom_metric_func

def metric_compare_z(true_z):
    """ Return NMI, Precision, Recall between inferred and true discrete labels
    Args:
        true_z (ndarray) length must match formatted observations
            Most likely for AR(p) (T) -> (T-p+1)
    """
    from sklearn.metrics import confusion_matrix, normalized_mutual_info_score
    def metric_z_function(sampler):
        pred_z = sampler.sample_z(track_samples=False)
        nmi = normalized_mutual_info_score(true_z, pred_z)
        cm = confusion_matrix(true_z, pred_z)
        precision = np.sum(np.max(cm, axis=0))/(np.sum(cm)*1.0)
        recall = np.sum(np.max(cm, axis=1))/(np.sum(cm)*1.0)
        metric_list = [
            dict(metric='nmi', variable='z', value=nmi),
            dict(metric='precision', variable='z', value=precision),
            dict(metric='recall', variable='z', value=recall),
        ]
        return metric_list
    return metric_z_function

def metric_compare_x(true_x):
    """ Return RMSE, MAE between inferred and true latent variables
    Args:
        true_x (ndarray)
    """
    def metric_x_function(sampler):
        pred_x = sampler.sample_x(track_samples=False)
        rmse = np.sqrt(np.mean((true_x - pred_x)**2))
        logmse = np.log10(np.mean((true_x - pred_x)**2))
        mae = np.mean(np.abs(true_x - pred_x))
        metric_list = [
            dict(metric='rmse', variable='x', value=rmse),
            dict(metric='mae', variable='x', value=mae),
            dict(metric='logmse', variable='x', value=logmse),
        ]
        return metric_list
    return metric_x_function




