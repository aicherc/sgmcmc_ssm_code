import numpy as np
import logging
logger = logging.getLogger(name=__name__)

from ..base_parameters import (
        BaseParameters, BasePrior, BasePreconditioner,
        )
from ..variables import (
        TransitionMatrixParamHelper, TransitionMatrixPriorHelper,
        TransitionMatrixPrecondHelper,
        VectorsParamHelper, VectorsPriorHelper,
        VectorsPrecondHelper,
        CovariancesParamHelper, CovariancesPriorHelper,
        CovariancesPrecondHelper,
        )
from ..sgmcmc_sampler import (
        SGMCMCSampler,
        SGMCMCHelper,
        )
from .._utils import (
        random_categorical,
        )

class GaussHMMParameters(BaseParameters):
    """ Gaussian HMM Parameters """
    _param_helper_list = [
            TransitionMatrixParamHelper(name='pi', dim_names=['num_states', 'pi_type']),
            VectorsParamHelper(name='mu', dim_names=['m', 'num_states']),
            CovariancesParamHelper(name='R', dim_names=['m', 'num_states']),
            ]
    for param_helper in _param_helper_list:
        properties = param_helper.get_properties()
        for name, prop in properties.items():
            vars()[name] = prop

    def __str__(self):
        my_str = "GaussHMMParameters:"
        my_str += "\npi:\n" + str(self.pi)
        my_str += "\nmu:\n" + str(self.mu)
        my_str += "\nR:\n" + str(self.R)
        return my_str

class GaussHMMPrior(BasePrior):
    """ Gaussian HMM Prior
    See individual Prior Mixins for details
    """
    _Parameters = GaussHMMParameters
    _prior_helper_list = [
            CovariancesPriorHelper(name='R', dim_names=['m', 'num_states'], matrix_name='mu'),
            TransitionMatrixPriorHelper(name='pi', dim_names=['num_states']),
            VectorsPriorHelper(name='mu', dim_names=['m', 'num_states'],
                var_row_name='R'),
            ]

class GaussHMMPreconditioner(BasePreconditioner):
    """ Gaussian HMM Preconditioner
    See individual Precondition Mixins for details
    """
    _precond_helper_list = [
            TransitionMatrixPrecondHelper(name='pi', dim_names=['num_states']),
            VectorsPrecondHelper(name='mu', dim_names=['m', 'num_states'], var_row_name='R'),
            CovariancesPrecondHelper(name='R', dim_names=['m', 'num_states']),
            ]

def generate_gausshmm_data(T, parameters, initial_message = None,
        tqdm=None):
    """ Helper function for generating Gaussian HMM time series

    Args:
        T (int): length of series
        parameters (GAUSSHMMParameters): parameters
        initial_message (ndarray): prior for u_{-1}

    Returns:
        data (dict): dictionary containing:
            observations (ndarray): T by m
            latent_vars (ndarray): T takes values in {1,...,num_states}
            parameters (GaussHMMParameters)
            init_message (ndarray)
    """
    from .._utils import random_categorical
    k, m = np.shape(parameters.mu)
    mu = parameters.mu
    R = parameters.R
    Pi = parameters.pi

    if initial_message is None:
        initial_message = {
                'prob_vector': np.ones(k)/k,
                'log_constant': 0.0,
                }

    latent_vars = np.zeros((T), dtype=int)
    obs_vars = np.zeros((T, m))
    latent_prev = random_categorical(initial_message['prob_vector'])
    pbar = range(T)
    if tqdm is not None:
        pbar = tqdm(pbar)
        pbar.set_description("generating data")
    for t in pbar:
        latent_vars[t] = random_categorical(Pi[latent_prev])
        mu_k = mu[latent_vars[t]]
        R_k = R[latent_vars[t]]
        obs_vars[t] = np.random.multivariate_normal(mean=mu_k, cov = R_k)
        latent_prev = latent_vars[t]

    data = dict(
            observations=obs_vars,
            latent_vars=latent_vars,
            parameters=parameters,
            initial_message=initial_message,
            )
    return data

class HMMHelper(SGMCMCHelper):
    """ HMM Helper

        forward_message (dict) with keys
            prob_vector (ndarray) dimension num_states
            log_constant (double) log scaling const

        backward_message (dict) with keys
            likelihood_vector (ndarray) dimension num_states
            log_constant (double) log scaling const
    """
    def __init__(self, forward_message=None, backward_message=None, **kwargs):
        if forward_message is None:
            forward_message = {
                'prob_vector': np.ones(self.num_states) / \
                        self.num_states,
                'log_constant': 0.0,
                        }
        self.default_forward_message=forward_message

        if backward_message is None:
            backward_message = {
                'likelihood_vector': np.ones(self.num_states)/self.num_states,
                'log_constant': np.log(self.num_states),
                }
        self.default_backward_message=backward_message
        return

    def _forward_messages(self, observations, parameters, forward_message,
            weights=None, tqdm=None, only_return_last=False):
        # Return list of forward messages
        # y is num_obs x m matrix
        num_obs = np.shape(observations)[0]
        if not only_return_last:
            forward_messages = [None]*(num_obs+1)
            forward_messages[0] = forward_message

        Pi = parameters.pi
        prob_vector = forward_message['prob_vector']
        log_constant = forward_message['log_constant']

        pbar = range(num_obs)
        if tqdm is not None:
            pbar = tqdm(pbar)
            pbar.set_description("forward messages")
        for t in pbar:
            y_cur = observations[t]
            weight_t = 1.0 if weights is None else weights[t]
            P_t, log_t = self._likelihoods(y_cur, parameters=parameters)
            prob_vector = np.dot(prob_vector, Pi)
            prob_vector = prob_vector * P_t
            log_constant += weight_t * (log_t + np.log(np.sum(prob_vector)))
            prob_vector = prob_vector/np.sum(prob_vector)

            if not only_return_last:
                forward_messages[t+1] = {
                    'prob_vector': prob_vector,
                    'log_constant': log_constant,
                }
        if only_return_last:
            last_message = {
                    'prob_vector': prob_vector,
                    'log_constant': log_constant,
                }
            return last_message
        else:
            return forward_messages

    def _backward_messages(self, observations, parameters, backward_message,
            weights=None, tqdm=None, only_return_last=False):
        # Return list of backward messages
        # y is num_obs x m matrix
        num_obs = np.shape(observations)[0]
        if not only_return_last:
            backward_messages = [None]*(num_obs+1)
            backward_messages[-1] = backward_message

        Pi = parameters.pi
        prob_vector = backward_message['likelihood_vector']
        log_constant = backward_message['log_constant']
        y_cur = None

        pbar = reversed(range(num_obs))
        if tqdm is not None:
            pbar = tqdm(pbar, total=num_obs)
            pbar.set_description("backward messages")
        for t in pbar:
            y_cur = observations[t]
            weight_t = 1.0 if weights is None else weights[t]
            P_t, log_t = self._likelihoods(y_cur=y_cur,
                    parameters=parameters)
            prob_vector = P_t * prob_vector
            prob_vector = np.dot(Pi, prob_vector)
            log_constant += weight_t * (log_t + np.log(np.sum(prob_vector)))
            prob_vector = prob_vector/np.sum(prob_vector)
            if not only_return_last:
                backward_messages[t] = {
                    'likelihood_vector': prob_vector,
                    'log_constant': log_constant,
                }
        if only_return_last:
            last_message = {
                'likelihood_vector': prob_vector,
                'log_constant': log_constant,
            }
            return last_message
        else:
            return backward_messages

    def marginal_loglikelihood(self, observations, parameters,
            forward_message=None, backward_message=None, weights=None,
            **kwargs):
        # Run forward pass + combine with backward pass
        # y is num_obs x p x m matrix
        if forward_message is None:
            forward_message = self.default_forward_message
        if backward_message is None:
            backward_message = self.default_backward_message

        # forward_pass is Pr(z_{T-1} | y_{<=T-1})
        forward_pass = self._forward_message(
                observations=observations,
                parameters=parameters,
                forward_message=forward_message,
                weights=weights,
                **kwargs)

        likelihood = np.dot(
                forward_pass['prob_vector'],
                backward_message['likelihood_vector'],
                )
        weight_t = 1.0 if weights is None else weights[-1]
        loglikelihood = forward_pass['log_constant'] + \
            weight_t * (np.log(likelihood) + backward_message['log_constant'])
        return loglikelihood

    def predictive_loglikelihood(self, observations, parameters, lag=10,
            forward_message=None, backward_message=None, **kwargs):
        if forward_message is None:
            forward_message = self.default_forward_message
        if backward_message is None:
            backward_message = self.default_backward_message

        # Calculate Filtered
        forward_messages = self.forward_pass(observations[0:-lag], parameters,
                forward_message, **kwargs)
        loglike = 0.0
        Pi = parameters.pi
        for t in range(lag, np.shape(observations)[0]):

            # Calculate Pr(z_t | y_{<=t-lag}, theta)
            prob_vector = forward_messages[t-lag]['prob_vector']
            for l in range(lag):
                prob_vector = np.dot(prob_vector, Pi)

            P_t, log_constant = self._likelihoods(observations[t], parameters)
            likelihood = np.dot(prob_vector, P_t)
            loglike += np.log(likelihood) + log_constant
        return loglike

    def latent_var_sample(self, observations, parameters,
            forward_message=None, backward_message=None,
            distribution='smoothed', tqdm=None):
        """ Sample latent vars from observations

        Backward pass + forward sampler for GAUSSHMM

        Args:
            observations (ndarray): num_obs by p by m observations
            parameters (GAUSSHMMParameters): parameters of GAUSSHMM
            forward_message (dict): alpha message
                (e.g. Pr(z_{-1} | y_{-inf:-1}))
                'log_constant' (double) log scaling constant
                'prob_vector' (ndarray) dimension num_states
                'y_prev' (ndarray) dimension p by m, optional
            backward_message (dict): beta message
                (e.g. Pr(y_{T:inf} | z_{T-1}))
                'log_constant' (double) log scaling constant
                'likelihood_vector' (ndarray) dimension num_states
                'y_next' (ndarray) dimension p by m, optional
            distribution (string): 'smoothed', 'filtered', 'predict'

        Returns
            z (ndarray): num_obs sampled latent values (in 1,...,K)
        """
        if forward_message is None:
            forward_message = self.default_forward_message
        if backward_message is None:
            backward_message = self.default_backward_message

        Pi = parameters.pi
        z = np.zeros(np.shape(observations)[0], dtype=int)

        if distribution == 'smoothed':
            # Backward Pass
            backward_messages = self.backward_pass(
                    observations=observations,
                    parameters=parameters,
                    backward_message=backward_message,
                    tqdm=tqdm
                    )

            # Forward Sampler
            pbar = enumerate(backward_messages)
            if tqdm is not None:
                pbar = tqdm(pbar, total=len(backward_messages))
                pbar.set_description("forward smoothed sampling z")
            for t, backward_t in pbar:
                y_cur = observations[t]
                if t == 0:
                    post_t = np.dot(forward_message['prob_vector'], Pi)
                else:
                    post_t = Pi[z[t-1]]
                P_t, _ = self._likelihoods(
                        y_cur=y_cur,
                        parameters=parameters,
                    )
                post_t = post_t * P_t * backward_t['likelihood_vector']
                post_t = post_t/np.sum(post_t)
                z[t] = random_categorical(post_t)

        elif distribution == 'filtered':
            # Forward Sampler (not a valid probability density)
            pbar = enumerate(observations)
            if tqdm is not None:
                pbar = tqdm(pbar)
                pbar.set_description("forward filtered sampling z")
            for t, y_cur in pbar:
                if t == 0:
                    post_t = np.dot(forward_message['prob_vector'], Pi)
                else:
                    post_t = Pi[z[t-1]]
                P_t, _ = self._likelihoods(
                        y_cur=y_cur,
                        parameters=parameters,
                    )
                post_t = post_t * P_t
                post_t = post_t/np.sum(post_t)
                z[t] = random_categorical(post_t)

        elif distribution == 'predictive':
            # Forward Sampler (not a valid probability density)
            pbar = range(np.shape(observations)[0])
            if tqdm is not None:
                pbar = tqdm(pbar)
                pbar.set_description("forward filtered sampling z")
            for t in pbar:
                if t == 0:
                    prob_vector = np.dot(forward_message['prob_vector'], Pi)
                else:
                    P_t, _ = self._likelihoods(
                            y_cur=observations[t-1],
                            parameters=parameters,
                        )
                    prob_vector = prob_vector * P_t
                    prob_vector = np.dot(prob_vector, Pi)
                prob_vector = prob_vector/np.sum(prob_vector)
                z[t] = random_categorical(prob_vector)
        else:
            raise ValueError("Unrecognized distr {0}".format(distr))

        return z

    def latent_var_marginal(self, observations, parameters,
           forward_message=None, backward_message=None,
           distribution='smoothed', tqdm=None):
        """ Calculate latent var marginal distribution

        Backward pass + forward sampler for HMM

        Args:
            observations (ndarray): observations
            parameters (parameters): parameters
            forward_message (dict): alpha message
                (e.g. Pr(z_{-1} | y_{-inf:-1}))
                'log_constant' (double) log scaling constant
                'prob_vector' (ndarray) dimension num_states
                'y_prev' (ndarray) observation, optional
            backward_message (dict): beta message
                (e.g. Pr(y_{T:inf} | z_{T-1}))
                'log_constant' (double) log scaling constant
                'likelihood_vector' (ndarray) dimension num_states
                'y_next' (ndarray) observation, optional
            distribution (string): 'smoothed', 'filtered', 'predict'

        Returns
            z_prob (ndarray): num_obs by K marginal posterior for z
        """
        if forward_message is None:
            forward_message = self.default_forward_message
        if backward_message is None:
            backward_message = self.default_backward_message

        L = np.shape(observations)[0]
        Pi = parameters.pi
        z_prob = np.zeros((L, parameters.num_states), dtype=float)

        if distribution == 'smoothed':
            forward_messages = self.forward_pass(
                    observations=observations,
                    parameters=parameters,
                    forward_message=forward_message,
                    tqdm=tqdm
                    )
            backward_messages = self.backward_pass(
                observations=observations,
                    parameters=parameters,
                    backward_message=backward_message,
                    tqdm=tqdm
                    )

            pbar = range(L)
            if tqdm is not None:
                pbar = tqdm(pbar, total=L)
                pbar.set_description("Marginalization")
            for t in pbar:
                log_prob_t = (np.log(forward_messages[t]['prob_vector']) + \
                              np.log(backward_messages[t]['likelihood_vector']))
                log_prob_t -= np.max(log_prob_t)
                z_prob[t] = np.exp(log_prob_t)/np.sum(np.exp(log_prob_t))

        elif distribution == 'filtered':
            raise NotImplementedError()

        elif distribution == 'predictive':
            raise NotImplementedError()
        else:
            raise ValueError("Unrecognized distr {0}".format(distr))

        return z_prob

    def y_marginal(self, observations, parameters,
            forward_message=None, backward_message=None,
            latent_vars=None, distribution="smoothed", tqdm=None):
        """ Calculate mean and variance y_t | y_{1:t-1}, z_t, theta """
        raise NotImplementedError()

    def calc_gibbs_sufficient_statistic(self, observations, latent_vars,
            **kwargs):
        """ Gibbs Sample Sufficient Statistics
        Args:
            observations (ndarray): num_obs observations
            latent_vars (ndarray): latent vars

        Returns:
            sufficient_stat (dict)
        """
        raise NotImplementedError()

    def _likelihoods(self, y_cur, parameters):
        raise NotImplementedError()

    def gradient_marginal_loglikelihood(self, observations, parameters,
            forward_message=None, backward_message=None,
            tqdm=None):
        raise NotImplementedError()

class GaussHMMHelper(HMMHelper):
    """ GaussHMM Helper

        forward_message (dict) with keys
            prob_vector (ndarray) dimension num_states
            log_constant (double) log scaling const

        backward_message (dict) with keys
            likelihood_vector (ndarray) dimension num_states
            log_constant (double) log scaling const
            y_next (ndarray) y_{t+1}
    """
    def __init__(self, num_states, m,
            forward_message=None,
            backward_message=None,
            **kwargs):
        self.num_states = num_states
        self.m = m
        super().__init__(
                forward_message=forward_message,
                backward_message=backward_message,
                **kwargs)
        return

    def y_marginal(self, observations, parameters,
            forward_message=None, backward_message=None,
            latent_vars=None, distribution="smoothed", tqdm=None):
        """ Calculate mean and standard deviation y_t | y_{1:t-1}, z_t, theta"""
        if forward_message is None:
            forward_message = self.default_forward_message
        if backward_message is None:
            backward_message = self.default_backward_message

        if latent_vars is None:
            latent_vars = self.latent_var_sample(
                    observations, parameters,
                    forward_message, backward_message,
                    distribution=distribution, tqdm=tqdm)

        num_obs, m = np.shape(observations)

        predictive_y_mean = np.zeros((num_obs, m))
        predictive_y_sd = np.zeros((num_obs, m))

        R = parameters.R
        for t in range(num_obs):
            z_t = latent_vars[t]
            predictive_y_mean[t] = parameters.mu[z_t]
            predictive_y_sd[t] = np.sqrt(np.diag(R[z_t]))
        return predictive_y_mean, predictive_y_sd

    def calc_gibbs_sufficient_statistic(self, observations, latent_vars,
            **kwargs):
        """ Gibbs Sample Sufficient Statistics
        Args:
            observations (ndarray): num_obs observations
            latent_vars (ndarray): latent vars

        Returns:
            sufficient_stat (dict) containing:
                alpha_Pi (ndarray) num_states by num_states, pairwise z counts
                S_count (ndarray) num_states, z counts
                S_prevprev (ndarray) num_states, z counts
                S_curprev (ndarray) num_states m, y sum
                S_curcur (ndarray) num_states m by m, yy.T sum
        """
        y, z = observations, latent_vars

        # Sufficient Statistics for Pi
        z_pair_count = np.zeros((self.num_states, self.num_states))

        for t in range(1, np.size(z)):
            z_pair_count[z[t-1], z[t]] += 1.0

        # Sufficient Statistics for mu and R
        S_count = np.zeros(self.num_states)
        y_sum = np.zeros((self.num_states, self.m))
        yy_curcur = np.zeros((self.num_states, self.m, self.m))

        for k in range(self.num_states):
            S_count[k] = np.sum(z == k)
            if S_count[k] == 0:
                # No Sufficient Statistics for No Observations
                continue
            yk = y[z==k, :]
            # Sufficient Statistics for group k
            y_sum[k] = np.sum(yk, axis=0)
            yy_curcur[k] = np.dot(yk.T, yk)

        # Return sufficient Statistics
        sufficient_stat = {}
        sufficient_stat['pi'] = dict(alpha = z_pair_count)
        sufficient_stat['mu'] = dict(S_prevprev = S_count, S_curprev = y_sum)
        sufficient_stat['R'] = dict(
                S_count=S_count,
                S_prevprev = S_count,
                S_curprev = y_sum,
                S_curcur=yy_curcur,
                )
        return sufficient_stat

    def _gauss_loglikelihoods(self, y_cur, parameters):
        # y_cur should be m,
        # mu is num_states by m
        # LRinv is num_states by m by m
        loglikelihoods = np.zeros(self.num_states, dtype=float)
        for k, (mu_k, LRinv_k) in enumerate(
                zip(parameters.mu, parameters.LRinv)):
            delta = y_cur - mu_k
            LRinvTdelta = np.dot(delta, LRinv_k)
            loglikelihoods[k] = \
                -0.5 * np.dot(LRinvTdelta, LRinvTdelta) + \
                -0.5 * self.m * np.log(2*np.pi) + \
                np.sum(np.log(np.diag(LRinv_k)))
        return loglikelihoods

    def _likelihoods(self, y_cur, parameters):
        logP_t = self._gauss_loglikelihoods(
                y_cur=y_cur, parameters=parameters,
                )
        log_constant = np.max(logP_t)
        logP_t = logP_t - log_constant
        P_t = np.exp(logP_t)
        return P_t, log_constant

    def gradient_marginal_loglikelihood(self, observations, parameters,
            forward_message=None, backward_message=None,
            weights=None, tqdm=None):
        # Forward Pass
        forward_messages = self.forward_pass(observations, parameters,
                forward_message, include_init_message=True)
        # Backward Pass
        backward_messages = self.backward_pass(observations, parameters,
                backward_message, include_init_message=True)

        # Gradients
        grad = {var: np.zeros_like(value)
                for var, value in parameters.as_dict().items()}

        Pi, expanded_pi = parameters.pi, parameters.expanded_pi
        mu = parameters.mu
        LRinv, Rinv, R = parameters.LRinv, parameters.Rinv, parameters.R

        pbar = enumerate(zip(forward_messages[:-1], backward_messages[1:]))
        if tqdm is not None:
            pbar = tqdm(pbar)
            pbar.set_description("gradient loglike")
        for t, (forward_t, backward_t) in pbar:
            # r_t is Pr(z_{t-1} | y_{< t})
            # s_t is Pr(z_t | y_{< t})
            # q_t is Pr(y_{> t} | z_t)
            r_t = forward_t['prob_vector']
            s_t = np.dot(r_t, Pi)
            q_t = backward_t['likelihood_vector']

            weight_t = 1.0 if weights is None else weights[t]

            # Calculate P_t = Pr(y_t | z_t)
            y_cur = observations[t]
            P_t, _ = self._likelihoods(
                    y_cur=y_cur,
                    parameters=parameters
                )

            # Marginal + Pairwise Marginal
            joint_post = np.diag(r_t).dot(Pi).dot(np.diag(P_t*q_t))
            joint_post = joint_post/np.sum(joint_post)
            marg_post = np.sum(joint_post, axis=0)

            # Grad for pi
            if parameters.pi_type == "logit":
                # Gradient of logit_pi
                grad['logit_pi'] += weight_t * (joint_post - \
                        np.diag(np.sum(joint_post, axis=1)).dot(Pi))
            elif parameters.pi_type == "expanded":
                grad['expanded_pi'] += weight_t * np.array([
                    (expanded_pi[k]**-1)*(
                        joint_post[k] - np.sum(joint_post[k])*Pi[k])
                    for k in range(self.num_states)
                    ])
            else:
                raise RuntimeError()

            # grad for mu and LRinv
            for k, mu_k, LRinv_k, Rinv_k, R_k in zip(
                    range(self.num_states), mu, LRinv, Rinv, R):
                diff_k = y_cur - mu_k
                grad['mu'][k] += weight_t * Rinv_k.dot(diff_k) * marg_post[k]
                grad['LRinv'][k] += weight_t * (
                        (R_k - np.outer(diff_k, diff_k)).dot(LRinv_k)
                        ) * marg_post[k]
        return grad

class GaussHMMSampler(SGMCMCSampler):
    def __init__(self, num_states, m, name="GAUSSHMMSampler", **kwargs):
        self.options = kwargs
        self.num_states = num_states
        self.m = m
        self.name = name
        self.message_helper=GaussHMMHelper(
                num_states=self.num_states,
                m=self.m,
                )
        return

    def setup(self, observations, prior, parameters=None,
            forward_message=None):
        """ Initialize the sampler

        Args:
            observations (ndarray): T by m ndarray of time series values
            prior (GAUSSHMMPrior): prior
            forward_message (ndarray): prior probability for latent state
            parameters (GAUSSHMMParameters): initial parameters
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
            if not isinstance(parameters, GaussHMMParameters):
                raise ValueError("parameters is not a GaussHMMParameter")
            self.parameters = parameters


        if forward_message is None:
            forward_message = {
                    'prob_vector': np.ones(self.num_states) / \
                            self.num_states,
                    'log_constant': 0.0,
                            }
        self.forward_message = forward_message
        self.backward_message = {
                'likelihood_vector': np.ones(self.num_states)/self.num_states,
                'log_constant': np.log(self.num_states),
                }

        return

    def init_parameters_from_z(self, z):
        """ Get initial parameters for the sampler

        Args:
            z (ndarray): latent var assignment

        Return:
            init_parameters (GAUSSHMMParameters): init_parameters
        """
        # Check z is appropriate size
        if np.shape(z)[0] != self.T:
            raise ValueError("z must be length T = {0}".format(self.T))

        if not np.issubdtype(z.dtype, np.integer):
            raise ValueError("z must be integers, not {0}".format(z.dtype))

        if np.max(z) >= self.num_states or np.min(z) < 0:
            raise ValueError("z must be in (0, \ldots, {0}-1)".format(
                self.num_states))

        # Perform on Gibb Step
        init_parameters = self.message_helper.parameters_gibbs_sample(
                observations=self.observations,
                latent_vars=z,
                prior=self.prior,
                )

        return init_parameters

    def init_parameters_from_k_means(self, lags=[0], kmeans=None, **kwargs):
        """ Get initial parameters for the sampler

        Use KMeans on data (treating observations as independent)
        Each point is concat(y[lag] for lag in lags)

        Args:
            lags (list of indices): indices of lags to use for clustering
            kmeans (sklearn model): e.g. sklearn.cluster.KMeans
            **kwargs (dict): keyword args to pass to sklearn's kmean
                "n_init" : int (default = 10)
                "max_iter": int (default = 300)
                "n_jobs" : int (default = 1)
                See sklearn.cluster.KMeans for more


        Returns:
            init_parameters (GAUSSHMMParameters): init_parameters
        """
        from sklearn.cluster import KMeans, MiniBatchKMeans

        # Run KMeans
        if kmeans is None:
            if self.T <= 10**6:
                kmeans = KMeans(n_clusters = self.num_states, **kwargs)
            else:
                kmeans = MiniBatchKMeans(n_clusters = self.num_states, **kwargs)

        X = self.observations.reshape((self.T, -1))
        X_lagged = np.hstack([
            X[max(lags)-lag:X.shape[0]-lag] for lag in lags
        ])

        z = kmeans.fit_predict(X=X_lagged)
        if z.size < self.T:
            z = np.concatenate([np.zeros(self.T-z.size, dtype=int), z])


        # Calculate Initial Param from KMeans init
        init_parameters = self.init_parameters_from_z(z)

        return init_parameters

    def sample_z(self, parameters=None, observations=None, tqdm=None, **kwargs):
        """ Sample Z """
        if parameters is None:
            parameters = self.parameters
        if observations is None:
            observations = self.observations
        z = self.message_helper.latent_var_sample(
                observations=observations,
                parameters=parameters,
                forward_message=self.forward_message,
                backward_message=self.backward_message,
                tqdm=tqdm,
                )
        return z

    def calc_z_prob(self, parameters=None, observations=None, tqdm=None,
            **kwargs):
        """ Calculate Posterior Marginal over Z """
        if parameters is None:
            parameters = self.parameters
        if observations is None:
            observations = self.observations
        z_prob = self.message_helper.latent_var_marginal(
                observations=observations,
                parameters=parameters,
                forward_message=self.forward_message,
                backward_message=self.backward_message,
                tqdm=tqdm,
                )
        return z_prob



    def sample_gibbs(self, tqdm=None):
        """ One Step of Blocked Gibbs Sampler

        Returns:
            parameters (GAUSSHMMParameters): sampled parameters after one step
        """
        z = self.sample_z(tqdm=tqdm)
        new_parameters = self.message_helper.parameters_gibbs_sample(
                observations=self.observations,
                latent_vars=z,
                prior=self.prior,
                )
        self.parameters = new_parameters
        return self.parameters










