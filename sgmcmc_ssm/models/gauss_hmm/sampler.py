import numpy as np
import logging
logger = logging.getLogger(name=__name__)

from ...sgmcmc_sampler import SGMCMCSampler, SeqSGMCMCSampler
from ..hmm_helper import CIRSamplerMixin
from .parameters import GaussHMMParameters, GaussHMMPrior, GaussHMMPreconditioner
from .helper import GaussHMMHelper

class GaussHMMSampler(CIRSamplerMixin, SGMCMCSampler):
    def __init__(self, num_states, m, observations=None, prior=None,
            parameters=None, forward_message=None, name="GaussHMMHelper",
            **kwargs):
        self.options = kwargs
        self.num_states = num_states
        self.m = m
        self.name = name
        self.setup(
                observations=observations,
                prior=prior,
                parameters=parameters,
                forward_message=forward_message,
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

        if prior is None:
            prior = GaussHMMPrior.generate_default_prior(
                    num_states=self.num_states, m=self.m,
                    )
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
        self.message_helper=GaussHMMHelper(
                num_states=self.num_states,
                m=self.m,
                forward_message=forward_message,
                backward_message=self.backward_message,
                )
        return

    def _check_observation_shape(self, observations):
        if observations is None:
            return
        if np.shape(observations)[1] != self.m:
            raise ValueError("observations last dimension does not match m")
        return

    def _get_preconditioner(self, preconditioner=None):
        if preconditioner is None:
            preconditioner = GaussHMMPreconditioner()
        return preconditioner

    def init_parameters_from_z(self, z, observations=None):
        """ Get initial parameters for the sampler

        Args:
            z (ndarray): latent var assignment

        Return:
            init_parameters (GAUSSHMMParameters): init_parameters
        """
        observations = self._get_observations(observations)

        # Check z is appropriate size
        if np.shape(z)[0] != observations.shape[0]:
            raise ValueError("z must be length of observations = {0}".format(observations.shape[0]))

        if not np.issubdtype(z.dtype, np.integer):
            raise ValueError("z must be integers, not {0}".format(z.dtype))

        if np.max(z) >= self.num_states or np.min(z) < 0:
            raise ValueError("z must be in (0, \ldots, {0}-1)".format(
                self.num_states))

        # Perform on Gibb Step
        init_parameters = self.message_helper.parameters_gibbs_sample(
                observations=observations,
                latent_vars=z,
                prior=self.prior,
                )
        self.parameters = init_parameters

        return init_parameters

    def init_parameters_from_k_means(self, lags=[0], kmeans=None,
            observations=None, **kwargs):
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
        observations = self._get_observations(observations)

        # Run KMeans
        if kmeans is None:
            if observations.shape[0] <= 10**6:
                kmeans = KMeans(n_clusters = self.num_states, **kwargs)
            else:
                kmeans = MiniBatchKMeans(n_clusters = self.num_states, **kwargs)

        X = observations.reshape((observations.shape[0], -1))
        X_lagged = np.hstack([
            X[max(lags)-lag:X.shape[0]-lag] for lag in lags
        ])

        z = kmeans.fit_predict(X=X_lagged)
        if z.size < observations.shape[0]:
            z = np.concatenate([np.zeros(observations.shape[0]-z.size,
                dtype=int), z])


        # Calculate Initial Param from KMeans init
        init_parameters = self.init_parameters_from_z(z, observations)

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

class SeqGaussHMMSampler(SeqSGMCMCSampler, GaussHMMSampler):
    pass

