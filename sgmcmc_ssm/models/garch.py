import numpy as np
import logging
logger = logging.getLogger(name=__name__)

from ..base_parameters import (
        BaseParameters, BasePrior,
        )
from ..variables import (
        CovarianceParamHelper, CovariancePriorHelper,
        )
from ..variables.garch_var import (
        GARCHParamHelper, GARCHPriorHelper,
        )
from ..sgmcmc_sampler import (
        SGMCMCSampler,
        SGMCMCHelper,
        SeqSGMCMCSampler,
        )
from .._utils import (
        var_stationary_precision,
        lower_tri_mat_inv,
        )
from ..particle_filters.kernels import (
        GARCHPriorKernel, GARCHOptimalKernel,
        )
from ..particle_filters.buffered_smoother import (
        buffered_pf_wrapper,
        average_statistic,
        )
from scipy.special import logit

class GARCHParameters(BaseParameters):
    """ GARCH Parameters """
    _param_helper_list = [
            GARCHParamHelper(),
            CovarianceParamHelper(name='R', dim_names=['m']),
            ]
    for param_helper in _param_helper_list:
        properties = param_helper.get_properties()
        for name, prop in properties.items():
            vars()[name] = prop

    def __str__(self):
        my_str = "GARCHParameters:"
        my_str +="\nalpha:{0}, beta:{1}, gamma:{2}, tau:{3}\n".format(
                np.around(np.asscalar(self.alpha), 6),
                np.around(np.asscalar(self.beta), 6),
                np.around(np.asscalar(self.gamma), 6),
                np.around(np.asscalar(self.tau), 6))
        return my_str

    @property
    def tau(self):
        if self.m == 1:
            tau = self.var_dict['LRinv'] ** -1
        else:
            tau = np.linalg.inv(self.var_dict['LRinv'].T)
        return tau

    @staticmethod
    def convert_alpha_beta_gamma(alpha, beta, gamma):
        """ Convert alpha, beta, gamma to log_mu, logit_phi, logit_lambduh

        mu = alpha / (1- beta - gamma)
        phi = beta + gamma
        lambda = beta / (beta + gamma)
        """
        if alpha <= 0 or beta <= 0 or gamma <= 0:
            raise ValueError("Cannot have alpha, beta, or gamma <= 0")
        if beta + gamma >=1:
            raise ValueError("Cannot have beta + gamma >- 1")
        log_mu = np.log(alpha/(1 - beta - gamma))
        logit_phi = logit(beta + gamma)
        logit_lambduh = logit(beta/(beta + gamma))
        return log_mu, logit_phi, logit_lambduh

class GARCHPrior(BasePrior):
    """ GARCH Prior
    See individual Prior Mixins for details
    """
    _Parameters = GARCHParameters
    _prior_helper_list = [
            GARCHPriorHelper(),
            CovariancePriorHelper(name='R', dim_names=['m']),
            ]

def generate_garch_data(T, parameters, initial_message = None,
        tqdm=None):
    """ Helper function for generating GARCH time series

    Args:
        T (int): length of series
        parameters (GARCHParameters): parameters
        initial_message (ndarray): prior for u_{-1}

    Returns:
        data (dict): dictionary containing:
            observations (ndarray): T by m
            latent_vars (ndarray): T by n
            parameters (GARCHParameters)
            init_message (ndarray)
    """
    n = 1
    m = 1
    alpha = parameters.alpha
    beta = parameters.beta
    gamma = parameters.gamma
    R = parameters.R

    if initial_message is None:
        init_precision =  np.atleast_2d((1 - beta - gamma)/alpha)
        initial_message = {
                'log_constant': 0.0,
                'mean_precision': np.zeros(n),
                'precision': init_precision,
                }

    latent_vars = np.zeros((T, n), dtype=float)
    sigma2s = np.zeros((T), dtype=float)
    obs_vars = np.zeros((T, m), dtype=float)
    latent_prev = np.random.multivariate_normal(
            mean=np.linalg.solve(initial_message['precision'],
                initial_message['mean_precision']),
            cov=np.linalg.inv(initial_message['precision']),
            )

    pbar = range(T)
    if tqdm is not None:
        pbar = tqdm(pbar)
        pbar.set_description("generating data")
    sigma2_prev = 0
    for t in pbar:
        sigma2s[t] = alpha + beta*latent_prev**2 + gamma*sigma2_prev
        latent_vars[t] = np.random.multivariate_normal(
                mean=np.zeros(1),
                cov=np.array([[sigma2s[t]]]),
                )
        obs_vars[t] = np.random.multivariate_normal(
                mean=latent_vars[t],
                cov=R,
                )
        latent_prev = latent_vars[t]
        sigma2_prev = sigma2s[t]

    data = dict(
            observations=obs_vars,
            latent_vars=latent_vars,
            sigma2s=sigma2s,
            parameters=parameters,
            initial_message=initial_message,
            )
    return data

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
            weights=None, tqdm=None):
        raise NotImplementedError('GARCH does not have analytic message passing')

    def _backward_messages(self, observations, parameters, backward_message,
            weights=None, tqdm=None):
        raise NotImplementedError('GARCH does not have analytic message passing')

    def latent_var_sample(self, observations, parameters,
            forward_message=None, backward_message=None,
            distribution='smoothed', num_samples=None,
            tqdm=None, include_init=False):
        """ Sample latent vars from observations

        Draw Samples from PF

        Args:
            observations (ndarray): num_obs by n observations
            parameters (GARCHParameters): parameters
            forward_message (dict): alpha message
                (e.g. Pr(x_{-1} | y_{-inf:-1}))
            distr (string): 'smoothed', 'filtered', 'predict'
            num_samples (int, optional) number of samples
            include_init (bool, optional): whether to sample x_{-1} | y

        Returns
            x (ndarray): (num_obs by n) latent values (if num_samples is None)
            or
            xs (ndarray): (num_obs by n by num_samples) latent variables

        """
        raise NotImplementedError()

    def latent_var_marginal(self, observations, parameters,
            forward_message=None, backward_message=None,
            distribution='smoothed', tqdm=None,
            include_init=False):
        raise NotImplementedError()

    def latent_var_pairwise_marginal(self, observations, parameters,
            forward_message=None, backward_message=None,
            distribution='smoothed', tqdm=None):
        raise NotImplementedError()

    def gradient_complete_data_loglikelihood(self, observations, latent_vars,
            parameters, forward_message=None, weights=None, tqdm=None, **kwargs):
        raise NotImplementedError()

    def gradient_loglikelihood(self, kind='marginal', **kwargs):
        if kind == 'marginal':
            return self.gradient_marginal_loglikelihood(**kwargs)
        elif kind == 'complete':
            return self.gradient_complete_data_loglikelihood(**kwargs)
        else:
            raise ValueError("Unrecognized `kind' {0}".format(kind))

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

    def pf_score_estimate(self, observations, parameters,
            subsequence_start=0, subsequence_end=None, weights=None,
            pf="poyiadjis_N", N=100, kernel=None,
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
        prior_mean, prior_var = self._get_prior_x(parameters)

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
            LRinv = score_estimate[0],
            log_mu = score_estimate[1],
            logit_phi = score_estimate[2],
            logit_lambduh = score_estimate[3],
            )

        return grad

    def pf_loglikelihood_estimate(self, observations, parameters,
            subsequence_start=0, subsequence_end=None, weights=None,
            pf="poyiadjis_N", N=1000, kernel=None,
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
        prior_mean, prior_var = self._get_prior_x(parameters)

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
            pf="pf_filter", N=1000, kernel=None,
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
        if pf != "pf_filter":
            raise ValueError("Only can use pf = 'pf_filter' since we are filtering")
        # Set kernel
        Kernel = self._get_kernel(kernel)

        # Prior Mean + Variance
        prior_mean, prior_var = self._get_prior_x(parameters)

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
            pf="poyiadjis_N", N=100, kernel=None,
            squared=False,
            **kwargs):
        # Set kernel
        Kernel = self._get_kernel(kernel)

        # Prior Mean + Variance
        prior_mean, prior_var = self._get_prior_x(parameters)

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

    def _get_prior_x(self, parameters):
        prior_mean = 0
        prior_var = parameters.alpha/(1-parameters.beta-parameters.gamma)
        return prior_mean, prior_var

class GARCHSampler(SGMCMCSampler):
    def __init__(self, n=1, m=1, name="GARCHSampler", **kwargs):
        self.options = kwargs
        self.n = n
        self.m = m
        self.name = name

        Helper = kwargs.get('Helper', GARCHHelper)
        self.message_helper=Helper(
                n=self.n,
                m=self.m,
                )
        return

    def setup(self, observations, prior, parameters=None, forward_message=None):
        """ Initialize the sampler

        Args:
            observations (ndarray): T by m ndarray of time series values
            prior (GARCHPrior): prior
            forward_message (ndarray): prior probability for latent state
            parameters (GARCHParameters): initial parameters
                (optional, will sample from prior by default)

        """
        # Check Shape
        if np.shape(observations)[1] != self.m:
            raise ValueError("observations second dimension does not match m")

        self.observations = observations
        self.T = np.shape(self.observations)[0]

        self.prior = prior

        if parameters is None:
            self.parameters = self.prior.sample_prior()
        else:
            if not isinstance(parameters, GARCHParameters):
                raise ValueError("parameters is not a GARCHParameter")
            self.parameters = parameters


        if forward_message is None:
             forward_message = {
                    'log_constant': 0.0,
                    'mean_precision': np.zeros(self.n),
                    'precision': np.eye(self.n)/10,
                    }
        self.forward_message = forward_message
        self.backward_message = {
                'log_constant': 0.0,
                'mean_precision': np.zeros(self.n),
                'precision': np.zeros((self.n, self.n)),
                }

        return

    def sample_x(self, parameters=None, observations=None, tqdm=None,
            num_samples=None, **kwargs):
        """ Sample X """
        raise NotImplementedError()

    def sample_gibbs(self, tqdm=None):
        """ One Step of Blocked Gibbs Sampler

        Returns:
            parameters (GARCHParameters): sampled parameters after one step
        """
        raise NotImplementedError()


class SeqGARCHSampler(SeqSGMCMCSampler, GARCHSampler):
    pass
