import numpy as np
import logging
logger = logging.getLogger(name=__name__)

from ...sgmcmc_sampler import SGMCMCHelper
from .kernels import SVMPriorKernel
from ...particle_filters.buffered_smoother import (
        buffered_pf_wrapper,
        average_statistic,
        )
from ..lgssm.helper import gaussian_sufficient_statistics

class SVMHelper(SGMCMCHelper):
    """ SVM Helper

        forward_message (dict) with keys
            log_constant (double) log scaling const
            mean_precision (ndarray) mean precision
            precision (ndarray) precision

        backward_message (dict) with keys
            log_constant (double) log scaling const
            mean_precision (ndarray) mean precision
            precision (ndarray) precision
    """
    def __init__(self, n, m, forward_message=None, backward_message=None,
            **kwargs):
        self.n = n
        self.m = m

        if forward_message is None:
             forward_message = {
                    'log_constant': 0.0,
                    'mean_precision': np.zeros(self.n),
                    'precision': np.eye(self.n)/10,
                    }
        self.default_forward_message=forward_message

        if backward_message is None:
            backward_message = {
                'log_constant': 0.0,
                'mean_precision': np.zeros(self.n),
                'precision': np.zeros((self.n, self.n)),
                }
        self.default_backward_message=backward_message
        return

    def _forward_messages(self, observations, parameters, forward_message,
            weights=None, tqdm=None, **kwargs):
        raise NotImplementedError('SVM does not have analytic message passing')

    def _backward_messages(self, observations, parameters, backward_message,
            weights=None, tqdm=None, **kwargs):
        raise NotImplementedError('SVM does not have analytic message passing')

    def _get_kernel(self, kernel):
        if kernel is None:
            kernel = "prior"
        if kernel == "prior":
            Kernel = SVMPriorKernel()
        elif kernel == "optimal":
            raise NotImplementedError("SVM optimal kernel not analytic")
        else:
            raise ValueError("Unrecoginized kernel = {0}".format(kernel))
        return Kernel

    def pf_gradient_estimate(self, observations, parameters,
            subsequence_start=0, subsequence_end=None, weights=None,
            pf="poyiadjis_N", N=1000, kernel=None, forward_message=None,
            **kwargs):
        """ Particle Filter Score Estimate

        Args:
            observations (ndarray): num_obs bufferd observations
            parameters (Parameters): parameters
            subsequence_start (int): relative start of subsequence
                (0:subsequence_start are left buffer)
            subsequence_end (int): relative end of subsequence
                (subsequence_end: is right buffer)
            weights (ndarray): weights for [subsequence_start, subsequence_end)
            pf (string): particle filter name
                "nemeth" - use Nemeth et al. O(N)
                "poyiadjis_N" - use Poyiadjis et al. O(N)
                "poyiadjis_N2" - use Poyiadjis et al. O(N^2)
                "paris" - use PaRIS Olsson + Westborn O(N log N)
            N (int): number of particles used by particle filter
            kernel (string): kernel to use
                "prior" - bootstrap filter P(X_t | X_{t-1})
                "optimal" - bootstrap filter P(X_t | X_{t-1}, Y_t)
            **kwargs - additional keyword args for individual filters

        Return:
            grad (dict): grad of variables in parameters

        """
        # Set kernel
        Kernel = self._get_kernel(kernel)

        # Prior Mean + Variance
        if forward_message is None:
            forward_message = self.default_forward_message
        prior_var = np.linalg.inv(forward_message['precision'])
        prior_mean = np.linalg.solve(prior_var,
                forward_message['mean_precision'])

        # Run buffered pf
        out = buffered_pf_wrapper(pf=pf,
                observations=observations,
                parameters=parameters,
                N=N,
                kernel=Kernel,
                additive_statistic_func=svm_complete_data_loglike_gradient,
                statistic_dim=3,
                t1=subsequence_start,
                tL=subsequence_end,
                weights=weights,
                prior_mean=prior_mean,
                prior_var=prior_var,
                **kwargs
                )
        score_estimate = average_statistic(out)
        grad = dict(
            LRinv_vec = score_estimate[0],
            LQinv_vec = score_estimate[1],
            A = score_estimate[2],
            )

        return grad

    def pf_loglikelihood_estimate(self, observations, parameters,
            subsequence_start=0, subsequence_end=None, weights=None,
            pf="poyiadjis_N", N=1000, kernel=None, forward_message=None,
            **kwargs):
        """ Particle Filter Marginal Log-Likelihood Estimate

        Args:
            observations (ndarray): num_obs bufferd observations
            parameters (Parameters): parameters
            subsequence_start (int): relative start of subsequence
                (0:subsequence_start are left buffer)
            subsequence_end (int): relative end of subsequence
                (subsequence_end: is right buffer)
            weights (ndarray): weights for [subsequence_start, subsequence_end)
            pf (string): particle filter name
                "nemeth" - use Nemeth et al. O(N)
                "poyiadjis_N" - use Poyiadjis et al. O(N)
                "poyiadjis_N2" - use Poyiadjis et al. O(N^2)
                "paris" - use PaRIS Olsson + Westborn O(N log N)
            N (int): number of particles used by particle filter
            kernel (string): kernel to use
                "prior" - bootstrap filter P(X_t | X_{t-1})
                "optimal" - bootstrap filter P(X_t | X_{t-1}, Y_t)
            **kwargs - additional keyword args for individual filters

        Return:
            loglikelihood (double): marignal log likelihood estimate

        """
        # Set kernel
        Kernel = self._get_kernel(kernel)

        # Prior Mean + Variance
        if forward_message is None:
            forward_message = self.default_forward_message
        prior_var = np.linalg.inv(forward_message['precision'])
        prior_mean = np.linalg.solve(prior_var,
                forward_message['mean_precision'])

        # Run buffered pf
        out = buffered_pf_wrapper(pf=pf,
                observations=observations,
                parameters=parameters,
                N=N,
                kernel=Kernel,
                additive_statistic_func=gaussian_sufficient_statistics,
                statistic_dim=3,
                t1=subsequence_start,
                tL=subsequence_end,
                weights=weights,
                prior_mean=prior_mean,
                prior_var=prior_var,
                **kwargs
                )
        loglikelihood = out['loglikelihood_estimate']
        return loglikelihood

    def pf_predictive_loglikelihood_estimate(self, observations, parameters,
            num_steps_ahead=5,
            subsequence_start=0, subsequence_end=None,
            pf="filter", N=1000, kernel=None, forward_message=None,
            **kwargs):
        """ Particle Filter Predictive Log-Likelihood Estimate

        Returns predictive log-likleihood for k = [0,1, ...,num_steps_ahead]

        Args:
            observations (ndarray): num_obs bufferd observations
            parameters (Parameters): parameters
            num_steps_ahead (int): number of steps
            subsequence_start (int): relative start of subsequence
                (0:subsequence_start are left buffer)
            subsequence_end (int): relative end of subsequence
                (subsequence_end: is right buffer)
            N (int): number of particles used by particle filter
            kernel (string): kernel to use
            **kwargs - additional keyword args for individual filters

        Return:
            predictive_loglikelihood (num_steps_ahead + 1 ndarray)

        """
        if pf != "filter":
            raise ValueError("Only can use pf = 'filter' since we are filtering")
        # Set kernel
        Kernel = self._get_kernel(kernel)

        # Prior Mean + Variance
        if forward_message is None:
            forward_message = self.default_forward_message
        prior_var = np.linalg.inv(forward_message['precision'])
        prior_mean = np.linalg.solve(prior_var,
                forward_message['mean_precision'])

        from functools import partial
        additive_statistic_func = partial(svm_predictive_loglikelihood,
                num_steps_ahead=num_steps_ahead,
                observations=observations,
                )

        # Run buffered pf
        out = buffered_pf_wrapper(pf=pf,
                observations=observations,
                parameters=parameters,
                N=N,
                kernel=Kernel,
                additive_statistic_func=additive_statistic_func,
                statistic_dim=num_steps_ahead+1,
                t1=subsequence_start,
                tL=subsequence_end,
                prior_mean=prior_mean,
                prior_var=prior_var,
                **kwargs
                )
        predictive_loglikelihood = out['statistics']
        predictive_loglikelihood[0] = out['loglikelihood_estimate']
        return predictive_loglikelihood

    def pf_latent_var_distr(self, observations, parameters, lag=None,
            subsequence_start=0, subsequence_end=None, weights=None,
            pf="poyiadjis_N", N=1000, kernel=None, forward_message=None,
            **kwargs):
        if lag == 0 and pf != 'filter':
            raise ValueError("pf must be filter for lag = 0")
        elif lag is None and pf == 'filter':
            raise ValueError("pf must not be filter for smoothing")
        elif lag is not None and lag != 0:
            raise NotImplementedError("lag can only be None or 0")

        # Set kernel
        Kernel = self._get_kernel(kernel)

        # Prior Mean + Variance
        if forward_message is None:
            forward_message = self.default_forward_message
        prior_var = np.linalg.inv(forward_message['precision'])
        prior_mean = np.linalg.solve(prior_var,
                forward_message['mean_precision'])

        # Run buffered pf
        out = buffered_pf_wrapper(pf=pf,
                observations=observations,
                parameters=parameters,
                N=N,
                kernel=Kernel,
                additive_statistic_func=gaussian_sufficient_statistics,
                statistic_dim=3,
                t1=subsequence_start,
                tL=subsequence_end,
                weights=weights,
                prior_mean=prior_mean,
                prior_var=prior_var,
                elementwise_statistic=True,
                **kwargs
                )
        avg_statistic = average_statistic(out)
        avg_statistic = np.reshape(avg_statistic, (-1, 3))
        x_mean = avg_statistic[:, 0]
        x_cov = avg_statistic[:, 1] - x_mean**2

        x_mean = np.reshape(x_mean, (x_mean.shape[0], 1))
        x_cov = np.reshape(x_cov, (x_cov.shape[0], 1, 1))

        return x_mean, x_cov

# Additive Statistics
def svm_complete_data_loglike_gradient(x_t, x_next, y_next, parameters, **kwargs):
    """ Gradient of Complete Data Log-Likelihood

    Gradient w/r.t. parameters of log Pr(y_{t+1}, x_{t+1} | x_t, parameters)

    Args:
        x_t (N by n ndarray): particles for x_t
        x_next (N by n ndarray): particles for x_{t+1}
        y_next (m ndarray): y_{t+1}
        parameters (Parameters): parameters
    Returns:
        grad_complete_data_loglike (N by p ndarray):
            gradient of complete data loglikelihood for particles
            [ grad_LRinv, grad_LQinv, grad_A ]
    """
    N, n = np.shape(x_next)
    m = np.shape(y_next)[0]

    A = parameters.A
    LQinv = parameters.LQinv
    Qinv = parameters.Qinv
    LRinv = parameters.LRinv
    Rinv = parameters.Rinv

    grad_complete_data_loglike = [None] * N
    if (n != 1) or (m != 1):
        LQinv_Tinv = np.linalg.inv(LQinv).T
        LRinv_Tinv = np.linalg.inv(LRinv).T
        for i in range(N):
            grad = {}
            diff = x_next[i] - np.dot(A, x_t[i])
            grad['A'] = np.outer(
                np.dot(Qinv, diff), x_t[i])
            grad['LQinv'] = LQinv_Tinv + -1.0*np.dot(np.outer(diff, diff), LQinv)

            diff2 = y_next**2 /np.exp(x_next[i])
            grad['LRinv'] = LRinv_Tinv + -1.0*np.dot(diff2, LRinv)

            grad_complete_data_loglike[i] = np.concatenate([
                grad['LRinv'].flatten(),
                grad['LQinv'].flatten(),
                grad['A'].flatten(),
                ])
        grad_complete_data_loglike = np.array(grad_complete_data_loglike)
    else:
        diff_x = x_next - A * x_t
        grad_A = Qinv * diff_x * x_t
        grad_LQinv = (LQinv**-1) - (diff_x**2) * LQinv
        diff_y2 = y_next**2/np.exp(x_next)
        grad_LRinv = (LRinv**-1) - (diff_y2) * LRinv
        grad_complete_data_loglike = np.hstack([
            grad_LRinv, grad_LQinv, grad_A])

    return grad_complete_data_loglike

def svm_predictive_loglikelihood(x_t, x_next, t, num_steps_ahead,
        parameters, observations, Ntilde=1,
        **kwargs):
    """ Predictive Log-Likelihood

    Calculate [Pr(y_{t+1+k} | x_{t+1} for k in [0,..., num_steps_ahead]]


    Args:
        x_t (N by n ndarray): particles for x_t
        x_next (N by n ndarray): particles for x_{t+1}
        num_steps_ahead
        parameters (Parameters): parameters
        observations (T by m ndarray): y
        Ntilde (int): number of MC samples
    Returns:
        predictive_loglikelihood (N by num_steps_ahead+1 ndarray)

    """
    N, n = np.shape(x_next)
    T, m = np.shape(observations)

    predictive_loglikelihood = np.zeros((N, num_steps_ahead+1))

    x_pred_mean = x_next + 0.0
    x_pred_cov = 0.0
    R, Q = parameters.R, parameters.Q
    for k in range(num_steps_ahead+1):
        if t+k >= T:
            break
        diff = observations[t+k]
        x_mc = (np.outer(x_pred_mean, np.ones(Ntilde)) + \
                np.sqrt(x_pred_cov)*np.random.normal(size=(N,Ntilde)))
        y_pred_cov = R*np.exp(x_mc)
        pred_loglike = np.mean(
                -0.5*diff**2/y_pred_cov + \
                -0.5*np.log(2.0*np.pi) - 0.5*np.log(y_pred_cov),
                axis = 1)
        predictive_loglikelihood[:,k] = pred_loglike

        x_pred_mean = parameters.A * x_pred_mean
        x_pred_cov = Q + parameters.A**2 * x_pred_cov

    return predictive_loglikelihood


