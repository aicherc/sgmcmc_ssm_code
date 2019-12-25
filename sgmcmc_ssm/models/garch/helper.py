import numpy as np
import logging
logger = logging.getLogger(name=__name__)

from ...sgmcmc_sampler import SGMCMCHelper
from .kernels import GARCHPriorKernel, GARCHOptimalKernel
from ...particle_filters.buffered_smoother import (
        buffered_pf_wrapper,
        average_statistic,
        )

class GARCHHelper(SGMCMCHelper):
    """ GARCH Helper

        forward_message (dict) with keys
            log_constant (double) log scaling const
            mean_precision (ndarray) mean precision
            precision (ndarray) precision

        backward_message (dict) with keys
            log_constant (double) log scaling const
            mean_precision (ndarray) mean precision
            precision (ndarray) precision
    """
    def __init__(self, n=1, m=1, forward_message=None, backward_message=None,
            **kwargs):
        self.n = n
        self.m = m
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
            weights=None, tqdm=None):
        raise NotImplementedError('GARCH does not have analytic message passing')

    def _backward_messages(self, observations, parameters, backward_message,
            weights=None, tqdm=None):
        raise NotImplementedError('GARCH does not have analytic message passing')

    def _get_kernel(self, kernel):
        if kernel is None:
            kernel = "optimal"
        if kernel == "prior":
            Kernel = GARCHPriorKernel()
        elif kernel == "optimal":
            Kernel = GARCHOptimalKernel()
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
        prior_mean, prior_var = self._get_prior_x(forward_message, parameters)

        # Run buffered pf
        out = buffered_pf_wrapper(pf=pf,
                observations=observations,
                parameters=parameters,
                N=N,
                kernel=Kernel,
                additive_statistic_func=garch_complete_data_loglike_gradient,
                statistic_dim=4,
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
            log_mu = score_estimate[1],
            logit_phi = score_estimate[2],
            logit_lambduh = score_estimate[3],
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
        prior_mean, prior_var = self._get_prior_x(forward_message, parameters)

        # Run buffered pf
        out = buffered_pf_wrapper(pf=pf,
                observations=observations,
                parameters=parameters,
                N=N,
                kernel=Kernel,
                additive_statistic_func=garch_sufficient_statistics,
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
            loglikelihood (double): marignal log likelihood estimate

        """
        if pf != "filter":
            raise ValueError("Only can use pf = 'filter' since we are filtering")
        # Set kernel
        Kernel = self._get_kernel(kernel)

        # Prior Mean + Variance
        prior_mean, prior_var = self._get_prior_x(forward_message, parameters)

        from functools import partial
        prior_kernel = self._get_kernel("prior")
        prior_kernel.set_parameters(parameters=parameters)
        additive_statistic_func = partial(garch_predictive_loglikelihood,
                num_steps_ahead=num_steps_ahead,
                observations=observations,
                prior_kernel=prior_kernel,
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

    def pf_latent_var_marginal(self, observations, parameters,
            subsequence_start=0, subsequence_end=None, weights=None,
            pf="poyiadjis_N", N=1000, kernel=None, forward_message=None,
            squared=False,
            **kwargs):
        # Set kernel
        Kernel = self._get_kernel(kernel)

        # Prior Mean + Variance
        prior_mean, prior_var = self._get_prior_x(forward_message, parameters)

        # Run buffered pf
        out = buffered_pf_wrapper(pf=pf,
                observations=observations,
                parameters=parameters,
                N=N,
                kernel=Kernel,
                additive_statistic_func=garch_sufficient_statistics,
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
        if not squared:
            x_mean = avg_statistic[:, 0]
            x_cov = avg_statistic[:, 1] - x_mean**2
        else:
            x_mean = avg_statistic[:, 1]
            x_cov = avg_statistic[:, 2] - x_mean**2

        x_mean = np.reshape(x_mean, (x_mean.shape[0], 1))
        x_cov = np.reshape(x_cov, (x_cov.shape[0], 1, 1))

        return x_mean, x_cov

    def pf_latent_var_distr(self, observations, parameters, lag=None,
            subsequence_start=0, subsequence_end=None, weights=None,
            pf="poyiadjis_N", N=1000, kernel=None, forward_message=None,
            squared=False,
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
        prior_mean, prior_var = self._get_prior_x(forward_message, parameters)

        # Run buffered pf
        out = buffered_pf_wrapper(pf=pf,
                observations=observations,
                parameters=parameters,
                N=N,
                kernel=Kernel,
                additive_statistic_func=garch_sufficient_statistics,
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
        if not squared:
            x_mean = avg_statistic[:, 0]
            x_cov = avg_statistic[:, 1] - x_mean**2
        else:
            x_mean = avg_statistic[:, 1]
            x_cov = avg_statistic[:, 2] - x_mean**2

        x_mean = np.reshape(x_mean, (x_mean.shape[0], 1))
        x_cov = np.reshape(x_cov, (x_cov.shape[0], 1, 1))

        return x_mean, x_cov



    def _get_prior_x(self, forward_message, parameters):
        if forward_message is None:
            prior_mean = 0
            prior_var = parameters.alpha/(1-parameters.beta-parameters.gamma)
        else:
            prior_var = np.linalg.inv(forward_message['precision'])
            prior_mean = np.linalg.solve(prior_var,
                    forward_message['mean_precision'])
        return prior_mean, prior_var

# Additive Statistics
def garch_complete_data_loglike_gradient(x_t, x_next, y_next, parameters, **kwargs):
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
    N, _ = np.shape(x_next)
    mu = parameters.mu
    phi = parameters.phi
    lambduh = parameters.lambduh
    LRinv = parameters.LRinv
    Rinv = parameters.Rinv

    v = x_next[:,1]
    grad_v = -0.5*(v-x_next[:,0]**2)/(v**2)
    grad_log_mu = grad_v * (1-phi) * mu
    grad_logit_phi = (grad_v * \
            (-mu + lambduh*x_t[:,0]**2 + (1-lambduh)*x_t[:,1]) * (1-phi)*phi
            )
    grad_logit_lambduh = (grad_v * \
            phi*(x_t[:,0]**2 - x_t[:,1]) * (1-lambduh)*lambduh
            )
    diff_y = y_next - x_next[:,0]
    grad_LRinv = (LRinv**-1) - (diff_y**2) * LRinv

    grad_complete_data_loglike = np.array([
        grad_LRinv[0], grad_log_mu, grad_logit_phi, grad_logit_lambduh]).T

    return grad_complete_data_loglike

def garch_predictive_loglikelihood(x_t, x_next, t, num_steps_ahead,
        parameters, observations, prior_kernel, Ntilde=1,
        **kwargs):
    """ Predictive Log-Likelihood

    Calculate [Pr(y_{t+1+k} | x_{t+1} for k in [0,..., num_steps_ahead]]
    Uses MC so is very noisy for large k


    Args:
        x_t (N by n ndarray): particles for x_t
        x_next (N by n ndarray): particles for x_{t+1}
        num_steps_ahead (int):
        parameters (Parameters): parameters
        observations (T by m ndarray): y
        Ntilde (int): number of MC samples

    Returns:
        predictive_loglikelihood (N by num_steps_ahead+1 ndarray)

    """
    N = np.shape(x_next)[0]
    T = np.shape(observations)[0]

    predictive_loglikelihood = np.zeros((N, num_steps_ahead+1))

    x_pred = x_next + 0
    R = parameters.R
    for k in range(num_steps_ahead+1):
        if t+k >= T:
            break
        diff = np.ones(N)*observations[t+k] - x_pred[:,0]
        y_pred_cov = R
        pred_loglike = -0.5*diff**2/y_pred_cov + \
                -0.5*np.log(2.0*np.pi) - 0.5*np.log(y_pred_cov)
        predictive_loglikelihood[:,k] = pred_loglike
        x_pred = prior_kernel.rv(x_pred)

    return predictive_loglikelihood

def garch_sufficient_statistics(x_t, x_next, y_next, **kwargs):
    """ GARCH Sufficient Statistics

    h[0] = sum(x_{t+1})
    h[1] = sum(x_{t+1}**2)
    h[2] = sum(x_{t+1}**4)

    Args:
        x_t (N by n ndarray): particles for x_t
        x_next (N by n ndarray): particles for x_{t+1}
        y_next (m ndarray): y_{t+1}
    Returns:
        h (N by p ndarray): sufficient statistic
    """
    N = np.shape(x_t)[0]
    h = np.array([x_next[:,0], x_next[:,0]**2, x_next[:,0]**4]).T
    return h




