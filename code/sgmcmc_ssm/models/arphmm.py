import numpy as np
import logging
logger = logging.getLogger(name=__name__)

from ..base_parameter import (
        BaseParameters, BasePrior, BasePreconditioner,
        )
from ..variable_mixins import (
        PiMixin, PiPrior, PiPreconditioner,
        DMixin, DPrior, DPreconditioner,
        RMixin, RPrior, RPreconditioner,
        )
from ..sgmcmc_sampler import (
        SGMCMCSampler,
        SGMCMCHelper,
        )
from .gauss_hmm import HMMHelper
from .._utils import (
        random_categorical,
        varp_stability_projection,
        )

class ARPHMMParameters(PiMixin, RMixin, DMixin,
        BaseParameters):
    """ ARPHMM Parameters """
    def __str__(self):
        my_str = "ARPHMMParameters:"
        my_str += "\npi:\n" + str(self.pi)
        my_str += "\npi_type: `" + str(self.pi_type) + "`"
        my_str += "\nD:\n" + str(self.D)
        my_str += "\nR:\n" + str(self.R)
        return my_str

    def _project_parameters(self, **kwargs):
        if kwargs.get('thresh_D', True):
           # Threshold B to be stable
           D = self.D
           for k, D_k in enumerate(D):
               D_k = varp_stability_projection(D_k,
                       eigenvalue_cutoff=0.9999, logger=logger)
               D[k] = D_k
           self.D = D

    @property
    def p(self):
        return self.dim['d']//self.dim['m']

class ARPHMMPrior(PiPrior, RPrior, DPrior, BasePrior):
    """ ARPHMM Prior for (Pi, D, Rinv)

    The prior for Pi_k is Dirichlet
    The prior for D_k is Gaussian (mean, cov \propto to Q)
    The prior for Rinv_k is Wishart

    """
    @staticmethod
    def _parameters(**kwargs):
        return ARPHMMParameters(**kwargs)

    @property
    def p(self):
        return self.dim['d']//self.dim['m']

class ARPHMMPreconditioner(PiPreconditioner, RPreconditioner,
        DPreconditioner, BasePreconditioner):
    """ Preconditioner for ARPHMM
    See individual Preconditioner Mixin for details
    """
    pass

def generate_arphmm_data(T, parameters, initial_message = None,
        tqdm=None):
    """ Helper function for generating ARPHMM time series

    Args:
        T (int): length of series
        parameters (ARPHMMParameters): parameters
        initial_message (ndarray): prior for u_{-1}

    Returns:
        data (dict): dictionary containing:
            observations (ndarray): T by p+1 by m
            latent_vars (ndarray): T takes values in {1,...,num_states}
            parameters (ARPHMMParameters)
            init_message (ndarray)
    """
    num_states, m, mp = np.shape(parameters.D)
    p = mp//m
    D = parameters.D
    R = parameters.R
    Pi = parameters.pi

    if initial_message is None:
        initial_message = {
                'prob_vector': np.ones(num_states)/num_states,
                'log_constant': 0.0,
                'y_prev': np.zeros((p,m))
                }

    latent_vars = np.zeros((T+p), dtype=int)
    obs_vars = np.zeros((T+p, m))
    latent_prev = random_categorical(initial_message['prob_vector'])
    y_prev = initial_message.get('y_prev')
    pbar = range(T)
    if tqdm is not None:
        pbar = tqdm(pbar)
        pbar.set_description("generating data")
    for t in pbar:
        latent_vars[t] = random_categorical(Pi[latent_prev])
        D_k = D[latent_vars[t]]
        R_k = R[latent_vars[t]]
        obs_vars[t] = np.random.multivariate_normal(
                mean=np.dot(D_k, y_prev.flatten()),
                cov = R_k,
                )
        latent_prev = latent_vars[t]
        y_prev = np.vstack([obs_vars[t], y_prev[:-1,:]])

    observations = stack_y(obs_vars, p)
    latent_vars = latent_vars[p:]

    data = dict(
            observations=observations,
            latent_vars=latent_vars,
            parameters=parameters,
            initial_message=initial_message,
            )
    return data

def stack_y(y, p):
    """ Stack y
    Args:
        y (ndarray): T+p by m matrix
    Returns:
        y_stacked (ndarray): T by p+1 by m matrix
    """
    T, m = np.shape(y)
    y_lags = [np.pad(y, ((0, lag), (0,0)), mode='constant')[lag:, :]
        for lag in reversed(range(p+1))]
    y_stacked = np.swapaxes(np.dstack(y_lags), 1, 2)[:T-p]
    return y_stacked

class ARPHMMHelper(HMMHelper):
    """ ARPHMM Helper

        forward_message (dict) with keys
            prob_vector (ndarray) dimension num_states
            log_constant (double) log scaling const

        backward_message (dict) with keys
            likelihood_vector (ndarray) dimension num_states
            log_constant (double) log scaling const
            y_next (ndarray) y_{t+1}
    """
    def __init__(self, num_states, m, d,
            forward_message=None,
            backward_message=None,
            **kwargs):
        self.num_states = num_states
        self.m = m
        self.d = d
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

        num_obs, p, m = np.shape(observations)

        predictive_y_mean = np.zeros((num_obs, m))
        predictive_y_sd = np.zeros((num_obs, m))

        R = parameters.R
        for t in range(num_obs):
            z_t = latent_vars[t]
            y_prev = observations[t,1:]
            predictive_y_mean[t] = np.dot(parameters.D[z_t],
                    y_prev.flatten())
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
                S_prevprev (ndarray) num_states mp by mp, z counts
                S_curprev (ndarray) num_states m by mp, y sum
                S_curcur (ndarray) num_states m by m, yy.T sum
        """
        z = latent_vars

        # Sufficient Statistics for Pi
        z_pair_count = np.zeros((self.num_states, self.num_states))

        for t in range(1, np.size(z)):
            z_pair_count[z[t-1], z[t]] += 1.0

        # Sufficient Statistics for mu and R
        S_count = np.zeros(self.num_states)
        y_sum = np.zeros((self.num_states, self.m))
        yy_prevprev = np.zeros((self.num_states, self.d, self.d))
        yy_curprev = np.zeros((self.num_states, self.m, self.d))
        yy_curcur = np.zeros((self.num_states, self.m, self.m))

        for k in range(self.num_states):
            S_count[k] = np.sum(z == k)
            if S_count[k] == 0:
                # No Sufficient Statistics for No Observations
                continue
            yk = observations[z==k, 0, :]
            yprevk = np.reshape(observations[z==k, 1:, :], (yk.shape[0], -1))

            # Sufficient Statistics for group k
            y_sum[k] = np.sum(yk, axis=0)
            yy_prevprev[k] = np.dot(yprevk.T, yprevk)
            yy_curprev[k] = np.dot(yk.T, yprevk)
            yy_curcur[k] = np.dot(yk.T, yk)

        # Return sufficient Statistics
        sufficient_stat = dict(
                alpha_Pi=z_pair_count,
                S_count=S_count,
                S_prevprev=yy_prevprev,
                S_curprev=yy_curprev,
                S_curcur=yy_curcur,
                )
        return sufficient_stat

    def _arphmm_loglikelihoods(self, y_cur, parameters):
        # y_cur should be p+1 by m,
        loglikelihoods = np.zeros(self.num_states, dtype=float)
        y_prev = y_cur[1:].flatten()
        for k, (D_k, LRinv_k) in enumerate(
                zip(parameters.D, parameters.LRinv)):
            delta = y_cur[0] - np.dot(D_k, y_prev)
            LRinvTdelta = np.dot(delta, LRinv_k)
            loglikelihoods[k] = \
                -0.5 * np.dot(LRinvTdelta, LRinvTdelta) + \
                -0.5 * self.m * np.log(2*np.pi) + \
                np.sum(np.log(np.diag(LRinv_k)))
        return loglikelihoods

    def _likelihoods(self, y_cur, parameters):
        logP_t = self._arphmm_loglikelihoods(
                y_cur=y_cur, parameters=parameters,
                )
        log_constant = np.max(logP_t)
        logP_t = logP_t - log_constant
        P_t = np.exp(logP_t)
        return P_t, log_constant

    def gradient_marginal_loglikelihood(self, observations, parameters,
            forward_message=None, backward_message=None,
            tqdm=None):
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
        D = parameters.D
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
                grad['logit_pi'] += joint_post - \
                        np.diag(np.sum(joint_post, axis=1)).dot(Pi)
            elif parameters.pi_type == "expanded":
                grad['expanded_pi'] += np.array([
                    (expanded_pi[k]**-1)*(
                        joint_post[k] - np.sum(joint_post[k])*Pi[k])
                    for k in range(self.num_states)
                    ])
            else:
                raise RuntimeError()

            # grad for mu and LRinv
            y_prev = y_cur[1:].flatten()
            for k, D_k, LRinv_k, Rinv_k, R_k in zip(
                    range(self.num_states), D, LRinv, Rinv, R):
                    diff_k = y_cur[0] - np.dot(D_k, y_prev)
                    grad['D'][k] += \
                            np.outer(Rinv_k.dot(diff_k), y_prev) * marg_post[k]
                    grad['LRinv'][k] += (
                            (R_k - np.outer(diff_k, diff_k)).dot(LRinv_k)
                            ) * marg_post[k]

        return grad

class ARPHMMSampler(SGMCMCSampler):
    # Note d = m*p
    def __init__(self, num_states, m, d, name="GAUSSHMMSampler", **kwargs):
        self.options = kwargs
        self.num_states = num_states
        self.m = m
        self.d = d
        self.name = name
        self.message_helper=ARPHMMHelper(
                num_states=self.num_states,
                m=self.m,
                d=self.d,
                )
        return

    @property
    def p(self):
        return self.d//self.m

    def setup(self, observations, prior, parameters=None,
            forward_message=None):
        """ Initialize the sampler

        Args:
            observations (ndarray): T by p+1 by m ndarray of time series values
            prior (GAUSSHMMPrior): prior
            forward_message (ndarray): prior probability for latent state
            parameters (GAUSSHMMParameters): initial parameters
                (optional, will sample from prior by default)

        """
        # Check Shape
        if np.shape(observations)[1] != self.m:
            raise ValueError("observations second dimension does not match m")
        if np.shape(observations)[2] != self.m*self.p:
            raise ValueError("observations third dimension does not match m*p")

        self.observations = observations
        self.T = np.shape(self.observations)[0]

        self.prior = prior

        if parameters is None:
            self.parameters = self.prior.sample_prior()
        else:
            if not isinstance(parameters, ARPHMMParameters):
                raise ValueError("parameters is not a ARPHMMParameter")
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
                forward_message=self.forward_message,
                backward_message=self.backward_message,
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

        X = self.observations[:, 0, :].reshape((self.T, -1))
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

    def sample_gibbs(self, tqdm=None):
        """ One Step of Blocked Gibbs Sampler

        Returns:
            parameters (GAUSSHMMParameters): sampled parameters after one step
        """
        z = self.sample_z(tqdm=tqdm)
        new_parameters = self.message_helper.parameters_gibbs_sample(
                observations=self.observations,
                latent_vars=z,
                forward_message=self.forward_message,
                backward_message=self.backward_message,
                prior=self.prior,
                )
        self.parameters = new_parameters
        return self.parameters


