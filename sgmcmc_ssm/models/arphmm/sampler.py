import numpy as np
import logging
logger = logging.getLogger(name=__name__)

from ...sgmcmc_sampler import SGMCMCSampler, SeqSGMCMCSampler
from .parameters import ARPHMMParameters, ARPHMMPrior, ARPHMMPreconditioner
from .helper import ARPHMMHelper

class ARPHMMSampler(SGMCMCSampler):
    # Note d = m*p
    def __init__(self, num_states, m, p=None, d=None,
            observations=None, prior=None,
            parameters=None, forward_message=None,
            name="ARPHMMHelper",
            **kwargs):
        self.options = kwargs
        self.num_states = num_states
        self.m = m
        if p is None:
            if d is None:
                raise ValueError("Need to specify p or d dimension parameter")
            else:
                self.p = d//m
        else:
            self.p = p
        self.name = name
        self.setup(
                observations=observations,
                prior=prior,
                parameters=parameters,
                forward_message=forward_message,
                )

        return

    @property
    def d(self):
        return self.p*self.m

    def setup(self, observations, prior, parameters=None,
            forward_message=None):
        """ Initialize the sampler

        Args:
            observations (ndarray): T by p+1 by m ndarray of time series values
            prior (ARPMMHMMPrior): prior
            forward_message (ndarray): prior probability for latent state
            parameters (ARPHMMParameters): initial parameters
                (optional, will sample from prior by default)

        """
        self.observations = observations

        if prior is None:
            prior = ARPHMMPrior.generate_default_prior(
                    num_states=self.num_states, m=self.m, d=self.d,
                    )
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
        self.message_helper=ARPHMMHelper(
                num_states=self.num_states,
                m=self.m,
                d=self.d,
                forward_message=forward_message,
                backward_message=self.backward_message,
                )

        return

    def _check_observation_shape(self, observations):
        if observations is None:
            return
        # Check Shape
        if np.shape(observations)[1] != self.p+1:
            raise ValueError("observations second dimension does not match p+1, did you call stack_y?")
        if np.shape(observations)[2] != self.m:
            raise ValueError("observations third dimension does not match m, did you call stack_y?")

    def _get_preconditioner(self, preconditioner=None):
        if preconditioner is None:
            preconditioner = ARPHMMPreconditioner()
        return preconditioner

    def init_parameters_from_z(self, z, observations=None):
        """ Get initial parameters for the sampler

        Args:
            z (ndarray): latent var assignment

        Return:
            init_parameters (ARPHMMParameters): init_parameters
        """
        observations = self._get_observations(observations=observations)
        T = self._get_T(observations=observations)

        # Check z is appropriate size
        if np.shape(z)[0] != self._get_T():
            raise ValueError("z must be length T = {0}".format(self._get_T()))

        if not np.issubdtype(z.dtype, np.integer):
            raise ValueError("z must be integers, not {0}".format(z.dtype))

        if np.max(z) >= self.num_states or np.min(z) < 0:
            raise ValueError("z must be in (0, \ldots, {0}-1)".format(
                self.num_states))

        # Perform on Gibb Step
        init_parameters = self.message_helper.parameters_gibbs_sample(
                observations=observations,
                latent_vars=z,
                forward_message=self.forward_message,
                backward_message=self.backward_message,
                prior=self.prior,
                )
        self.parameters = init_parameters

        return init_parameters

    def init_parameters_from_k_means(self, observations=None,
            lags=[0], kmeans=None, **kwargs):
        """ Get initial parameters for the sampler

        Use KMeans on data (treating observations as independent)
        Each point is concat(y[lag] for lag in lags)

        Args:
            observations (ndarray): observations
            lags (list of indices): indices of lags to use for clustering
            kmeans (sklearn model): e.g. sklearn.cluster.KMeans
            **kwargs (dict): keyword args to pass to sklearn's kmean
                "n_init" : int (default = 10)
                "max_iter": int (default = 300)
                "n_jobs" : int (default = 1)
                See sklearn.cluster.KMeans for more


        Returns:
            init_parameters (ARPHMMParameters): init_parameters
        """
        from sklearn.cluster import KMeans, MiniBatchKMeans
        observations = self._get_observations(observations=observations)
        T = self._get_T(observations=observations)

        # Run KMeans
        if kmeans is None:
            if T <= 10**6:
                kmeans = KMeans(n_clusters = self.num_states, **kwargs)
            else:
                kmeans = MiniBatchKMeans(n_clusters = self.num_states, **kwargs)

        X = observations[:, 0, :].reshape((T, -1))
        X_lagged = np.hstack([
            X[max(lags)-lag:X.shape[0]-lag] for lag in lags
        ])

        z = kmeans.fit_predict(X=X_lagged)
        if z.size < T:
            z = np.concatenate([np.zeros(T-z.size, dtype=int), z])


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
        z_prob = self.message_helper.latent_var_distr(
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
            parameters (ARPHMMParameters): sampled parameters after one step
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

class SeqARPHMMSampler(SeqSGMCMCSampler, ARPHMMSampler):
    pass

