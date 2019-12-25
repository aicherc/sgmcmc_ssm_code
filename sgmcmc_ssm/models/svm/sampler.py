import numpy as np
from ...sgmcmc_sampler import SGMCMCSampler, SeqSGMCMCSampler
from .parameters import SVMPrior, SVMParameters
from .helper import SVMHelper

class SVMSampler(SGMCMCSampler):
    def __init__(self, n, m, observations=None, prior=None, parameters=None,
            forward_message=None, name="SVMSampler", **kwargs):
        self.options = kwargs
        self.n = n
        self.m = m
        self.name = name
        self.setup(
                observations=observations,
                prior=prior,
                parameters=parameters,
                forward_message=forward_message,
                )
        return

    def setup(self, observations=None, prior=None,
            parameters=None, forward_message=None):
        """ Initialize the sampler

        Args:
            observations (ndarray): T by m ndarray of time series values
            prior (SVMPrior): prior
            forward_message (ndarray): prior probability for latent state
            parameters (SVMParameters): initial parameters
                (optional, will sample from prior by default)

        """
        self.observations = observations

        if prior is None:
            prior = SVMPrior.generate_default_prior(n=self.n, m=self.m)
        self.prior = prior

        if parameters is None:
            self.parameters = self.prior.sample_prior().project_parameters()
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
        self.message_helper=SVMHelper(
                n=self.n,
                m=self.m,
                forward_message=forward_message,
                backward_message=self.backward_message,
                )
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
