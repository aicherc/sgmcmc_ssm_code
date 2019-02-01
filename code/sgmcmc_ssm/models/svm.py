import numpy as np
import logging
logger = logging.getLogger(name=__name__)

from ..base_parameter import (
        BaseParameters, BasePrior, BasePreconditioner,
        )
from ..variable_mixins import (
        ASingleMixin, ASinglePrior, ASinglePreconditioner,
        QSingleMixin, QSinglePrior, QSinglePreconditioner,
        RSingleMixin, RSinglePrior, RSinglePreconditioner,
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
        SVMPriorKernel,
        )
from ..particle_filters.buffered_smoother import (
        buffered_pf_wrapper,
        average_statistic,
        )
from sgmcmc_ssm.particle_filters.pf import (
        gaussian_sufficient_statistics,
        )


class SVMParameters(RSingleMixin, QSingleMixin, ASingleMixin,
        BaseParameters):
    """ SVM Parameters """
    def __str__(self):
        my_str = "SVMParameters:"
        my_str += "\nA:\n" + str(self.A)
        my_str += "\nQ:\n" + str(self.Q)
        my_str += "\nR:\n" + str(self.R)
        return my_str


    @property
    def phi(self):
        phi = self.var_dict['A']
        return phi

    @property
    def sigma(self):
        if self.n == 1:
            sigma = self.var_dict['LQinv'] ** -1
        else:
            sigma = np.linalg.inv(self.var_dict['LQinv'].T)
        return sigma

    @property
    def tau(self):
        if self.m == 1:
            tau = self.var_dict['LRinv'] ** -1
        else:
            tau = np.linalg.inv(self.var_dict['LRinv'].T)
        return tau


class SVMPrior(RSinglePrior, QSinglePrior, ASinglePrior,
        BasePrior):
    """ SVM Prior
    See individual Prior Mixins for details
    """
    @staticmethod
    def _parameters(**kwargs):
        return SVMParameters(**kwargs)

#class SVMPreconditioner(RSinglePreconditioner, QSinglePreconditioner,
#        ASinglePreconditioner, BasePreconditioner):
#    """ Preconditioner for SVM
#    See individual Preconditioner Mixin for details
#    """
#    pass

def generate_svm_data(T, parameters, initial_message = None,
        tqdm=None):
    """ Helper function for generating SVM time series

    Args:
        T (int): length of series
        parameters (SVMParameters): parameters
        initial_message (ndarray): prior for u_{-1}

    Returns:
        data (dict): dictionary containing:
            observations (ndarray): T by m
            latent_vars (ndarray): T by n
            parameters (SVMParameters)
            init_message (ndarray)
    """
    n, _ = np.shape(parameters.A)
    m, _ = np.shape(parameters.R)
    A = parameters.A
    Q = parameters.Q
    R = parameters.R

    if initial_message is None:
        init_precision = var_stationary_precision(
                parameters.Qinv, parameters.A, 10)
        initial_message = {
                'log_constant': 0.0,
                'mean_precision': np.zeros(n),
                'precision': init_precision,
                }

    latent_vars = np.zeros((T, n), dtype=float)
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
    for t in pbar:
        latent_vars[t] = np.random.multivariate_normal(
                mean=np.dot(A, latent_prev),
                cov=Q,
                )
        obs_vars[t] = np.random.multivariate_normal(
                mean=np.zeros(1),
                cov=np.exp(latent_vars[t])*R,
                )
        latent_prev = latent_vars[t]

    data = dict(
            observations=obs_vars,
            latent_vars=latent_vars,
            parameters=parameters,
            initial_message=initial_message,
            )
    return data

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
            tqdm=None):
        raise NotImplementedError('SVM does not have analytic message passing')

    def _backward_messages(self, observations, parameters, backward_message,
            tqdm=None):
        raise NotImplementedError('SVM does not have analytic message passing')

    def latent_var_sample(self, observations, parameters,
            forward_message=None, backward_message=None,
            distribution='smoothed', num_samples=None,
            tqdm=None, include_init=False):
        """ Sample latent vars from observations

        Draw Samples from PF

        Args:
            observations (ndarray): num_obs by n observations
            parameters (SVMParameters): parameters
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
            parameters, forward_message=None, tqdm=None, **kwargs):
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
            kernel = "prior"
        if kernel == "prior":
            Kernel = SVMPriorKernel()
        elif kernel == "optimal":
            raise NotImplementedError("SVM optimal kernel not analytic")
        else:
            raise ValueError("Unrecoginized kernel = {0}".format(kernel))
        return Kernel

    def pf_score_estimate(self, observations, parameters,
            subsequence_start=0, subsequence_end=None,
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
        prior_var = self.default_forward_message['precision'][0,0]**-1
        prior_mean = \
                self.default_forward_message['mean_precision'][0] * prior_var

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
                prior_mean=prior_mean,
                prior_var=prior_var,
                **kwargs
                )
        score_estimate = average_statistic(out)
        grad = dict(
            LRinv = score_estimate[0],
            LQinv = score_estimate[1],
            A = score_estimate[2],
            )

        return grad

    def pf_loglikelihood_estimate(self, observations, parameters,
            subsequence_start=0, subsequence_end=None,
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
        prior_var = self.default_forward_message['precision'][0,0]**-1
        prior_mean = \
                self.default_forward_message['mean_precision'][0] * prior_var

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
            predictive_loglikelihood (num_steps_ahead + 1 ndarray)

        """
        if pf != "pf_filter":
            raise ValueError("Only can use pf = 'pf_filter' since we are filtering")
        # Set kernel
        Kernel = self._get_kernel(kernel)

        # Prior Mean + Variance
        prior_var = self.default_forward_message['precision'][0,0]**-1
        prior_mean = \
                self.default_forward_message['mean_precision'][0] * prior_var

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

    def pf_latent_var_marginal(self, observations, parameters,
            subsequence_start=0, subsequence_end=None,
            pf="poyiadjis_N", N=100, kernel=None,
            **kwargs):
        # Set kernel
        Kernel = self._get_kernel(kernel)

        # Prior Mean + Variance
        prior_var = self.default_forward_message['precision'][0,0]**-1
        prior_mean = \
                self.default_forward_message['mean_precision'][0] * prior_var

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

class SVMSampler(SGMCMCSampler):
    def __init__(self, n, m, name="SVMSampler", **kwargs):
        self.options = kwargs
        self.n = n
        self.m = m
        self.name = name

        Helper = kwargs.get('Helper', SVMHelper)
        self.message_helper=Helper(
                n=self.n,
                m=self.m,
                )
        return

    def setup(self, observations, prior, parameters=None, forward_message=None):
        """ Initialize the sampler

        Args:
            observations (ndarray): T by m ndarray of time series values
            prior (SVMPrior): prior
            forward_message (ndarray): prior probability for latent state
            parameters (SVMParameters): initial parameters
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
            if not isinstance(parameters, SVMParameters):
                raise ValueError("parameters is not a SVMParameter")
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
            parameters (SVMParameters): sampled parameters after one step
        """
        raise NotImplementedError()


class SeqSVMSampler(SeqSGMCMCSampler, SVMSampler):
    pass
