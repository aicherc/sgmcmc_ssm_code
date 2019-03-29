"""
Wrapper Helper Class around PaRIS
"""
import numpy as np
import functools
from .pf import (
        paris_smoother, nemeth_smoother, poyiadjis_smoother,
        pf_filter,
        log_normalize,
        )

def pf_wrapper(
        observations, parameters, N,
        kernel, smoother,
        additive_statistic_func, statistic_dim,
        t1=0, tL=None,
        prior_mean = 0.0, prior_var = 1.0,
        tqdm = None, tqdm_name = None,
        save_all=False, elementwise_statistic=False,
        **kwargs):
    """ Wrapper around particle smoothers for calculating additive_statistics

    Args:
        observations (ndarray): observed data
        parameters (Parameters): parameters for kernel + additive_statistic_func
        N (int): number of smoothed particles
        kernel (Kernel): kernel with proposal, reweight, prior_log_density funcs
        smoother (func): one of the pf methods in `pf.py`
        additive_statistic_func (func): additive statistic func
        statistic_dim (int): dimension of additive_statistic_func return array
        t1 (int): relative start of left buffer
        tL (int): relative end of right buffer (exclusive, buffer is [t1, tL-1])
        prior_mean (ndarray): prior mean for latent variable
        prior_var (ndarray): prior var for latent variable
        tqdm (optional): progress bar
        tqdm_name (string): message for progress bar
        save_all (bool): whether to save intermediate output
        elementwise_statistic (bool): whether additive stastic return elementwise
        **kwargs (dict): additional args for smoother

    Returns:
        out (dict): with outputs
            x_t (N by n ndarray): final smoothed/filtered latent variables
            log_weights (N ndarray): weights for latent variables
            statistics (N by statistic_dim ndarray): final smoothed statistics
                (or (statistic_dim ndarray) if smoother is pf_filter)
            loglikelihood_estimate (float): loglikelihood estimate
    """
    T, m = np.shape(observations)
    if tL is None:
        tL = T

    n = parameters.n
    kernel.set_parameters(parameters=parameters)

    if elementwise_statistic:
        statistic_dim = (tL - t1) * statistic_dim


    # Initialize PF
    x_t = kernel.sample_x0(prior_mean=prior_mean, prior_var=prior_var,
            N=N, n=n)
    log_weights = np.zeros(N)
    loglikelihood_estimate = 0.0
    if kwargs.get('is_filter', False):
        statistics = np.zeros((statistic_dim))
    else:
        statistics = np.zeros((N, statistic_dim))

    def zero_statistics(x_t, x_next, **kwargs):
        Ntilde = np.shape(x_t)[0]
        return np.zeros((Ntilde, statistic_dim))

    if save_all:
        all_x_t = [x_t]
        all_log_weights = [log_weights]
        all_statistics = [statistics]
        all_loglikelihood_estimate = [loglikelihood_estimate]

    pbar = range(T)
    if tqdm is not None:
        pbar = tqdm(pbar)
        if tqdm_name is not None:
            pbar.set_description('PF: {0}'.format(tqdm_name))
    for t in pbar:
        kernel.set_y_next(y_next=observations[t])
        # Only Sum over terms not in the buffer
        if t < t1 or t >= tL:
            additive_statistic_func_t = zero_statistics
        else:
            additive_statistic_func_t = functools.partial(
                    additive_statistic_func,
                    y_next = observations[t],
                    t = t,
                    parameters=parameters)
            if elementwise_statistic:
                additive_statistic_func_t = elementwise_statistic_wrapper(
                        additive_statistic_func_t,
                        shift=t-t1,
                        length=tL-t1,
                        )

        x_t, log_weights, statistics = smoother(
                x_t, log_weights, statistics,
                additive_statistic_func=additive_statistic_func_t,
                kernel=kernel,
                **kwargs,
                )
        if (t >= t1) and (t < tL):
            loglikelihood_estimate += np.log(np.mean(np.exp(log_weights)))

        if save_all:
            all_x_t.append(x_t)
            all_log_weights.append(log_weights)
            all_statistics.append(statistics)
            all_loglikelihood_estimate.append(loglikelihood_estimate)

    out = {}
    if save_all:
        out['all_x_t'] = np.array(all_x_t)
        out['all_log_weights'] = np.array(all_log_weights)
        out['all_statistics'] = np.array(all_statistics)
        out['all_loglikelihood_estimate'] = np.array(all_loglikelihood_estimate)

    out['x_t'] = x_t
    out['log_weights'] = log_weights
    out['statistics'] = statistics
    out['loglikelihood_estimate'] = loglikelihood_estimate

    return out

def average_statistic(out):
    mean_statistic = np.sum(
            out['statistics'].T * log_normalize(out['log_weights']), axis=1)
    return mean_statistic

def buffered_pf_wrapper(pf, **kwargs):
    """ Wrapper for buffered pf wrappers
    Args:
        pf (string)
            "nemeth" - use Nemeth et al. O(N)
            "poyiadjis_N" - use Poyiadjis et al. O(N)
            "poyiadjis_N2" - use Poyiadjis et al. O(N^2)
            "paris" - use PaRIS Olsson + Westborn O(N log N)
            "filter" - just use PF (no smoothing)
        **kwargs

    Returns:
        out (dict)
    """
    if pf == "nemeth":
        tqdm_name = kwargs.pop('tqdm_name', 'Nemeth O(N)')
        smoother = nemeth_smoother
        out = pf_wrapper(smoother=smoother, tqdm_name=tqdm_name, **kwargs)

    elif pf == "poyiadjis_N":
        tqdm_name = kwargs.pop('tqdm_name', 'Poyiadjis O(N)')
        lambduh = 1.0
        smoother = nemeth_smoother
        out = pf_wrapper(smoother=smoother, tqdm_name=tqdm_name,
                lambduh=lambduh, **kwargs)

    elif pf == "poyiadjis_N2":
        tqdm_name = kwargs.pop('tqdm_name', 'Poyiadjis O(N^2)')
        smoother = poyiadjis_smoother
        out = pf_wrapper(smoother=smoother, tqdm_name=tqdm_name, **kwargs)

    elif pf == "paris":
        tqdm_name = kwargs.pop('tqdm_name', 'PaRIS')
        smoother = paris_smoother
        out = pf_wrapper(smoother=smoother, tqdm_name=tqdm_name, **kwargs)

    elif pf == "pf_filter":
        tqdm_name = kwargs.pop('tqdm_name', 'PF Filter')
        smoother = pf_filter
        kwargs['is_filter'] = True
        out = pf_wrapper(smoother=smoother, tqdm_name=tqdm_name, **kwargs)

    else:
        raise ValueError("Unrecognized pf = {0}".format(pf))
    return out

def elementwise_statistic_wrapper(
        additive_statistic_func, shift, length):
    def shifted_statistic_func(*args, **kwargs):
        statistic = additive_statistic_func(*args, **kwargs)
        N, statistic_dim = np.shape(statistic)
        shifted_statistic = np.zeros((N, statistic_dim*length))
        shifted_statistic[:,shift*statistic_dim:(shift+1)*statistic_dim] = \
                statistic
        return shifted_statistic
    return shifted_statistic_func



