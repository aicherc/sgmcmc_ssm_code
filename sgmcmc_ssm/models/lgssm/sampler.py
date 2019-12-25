import numpy as np
from ...sgmcmc_sampler import SGMCMCSampler, SeqSGMCMCSampler
from .parameters import LGSSMPrior, LGSSMPreconditioner
from .helper import LGSSMHelper

class LGSSMSampler(SGMCMCSampler):
    def __init__(self, n, m, observations=None, prior=None, parameters=None,
            forward_message=None, backward_message=None, name="LGSSMSampler",
            **kwargs):
        self.options = kwargs
        self.n = n
        self.m = m
        self.name = name
        self.setup(
                observations=observations,
                prior=prior,
                parameters=parameters,
                forward_message=forward_message,
                backward_message=backward_message,
                )
        return

    def setup(self, observations=None, prior=None, parameters=None,
            forward_message=None, backward_message=None):
        self.observations = observations

        if prior is None:
            prior = LGSSMPrior.generate_default_prior(n=self.n, m=self.m)
        self.prior = prior

        if parameters is None:
            self.parameters = self.prior.sample_prior()
        else:
            self.parameters = parameters

        if forward_message is None:
             forward_message = {
                    'log_constant': 0.0,
                    'mean_precision': np.zeros(self.n),
                    'precision': np.eye(self.n)/10,
                    }
        self.forward_message = forward_message
        if backward_message is None:
            backward_message = {
                    'log_constant': 0.0,
                    'mean_precision': np.zeros(self.n),
                    'precision': np.zeros((self.n, self.n)),
                    }
        self.backward_message = backward_message
        self.message_helper=LGSSMHelper(
                n=self.n,
                m=self.m,
                forward_message=forward_message,
                backward_message=backward_message,
                )
        return

    def _check_observation_shape(self, observations):
        if observations is None:
            return
        if np.shape(observations)[1] != self.m:
            raise ValueError("observations second dimension does not match m")
        return

    def _get_preconditioner(self, preconditioner=None):
        if preconditioner is None:
            preconditioner = LGSSMPreconditioner()
        return preconditioner

    def sample_x(self, observations=None, parameters=None, tqdm=None,
            num_samples=None, **kwargs):
        """ Sample X """
        return self.predict(target='latent', kind='analytic',
                observations=observations, parameters=parameters,
                num_samples=num_samples, tqdm=tqdm,
                **kwargs,
                )

    def sample_gibbs(self, parameters=None, observations=None, tqdm=None):
        """ One Step of Blocked Gibbs Sampler

        Returns:
            parameters (LGSSMParameters): sampled parameters after one step
        """
        if parameters is None:
            parameters = self.parameters
        observations = self._get_observations(observations)
        x = self.sample_x(parameters=parameters, observations=observations,
                tqdm=tqdm)
        new_parameters = self.message_helper.parameters_gibbs_sample(
                observations=observations,
                latent_vars=x,
                prior=self.prior,
                )
        self.parameters = new_parameters
        return self.parameters

class SeqLGSSMSampler(SeqSGMCMCSampler, LGSSMSampler):
    pass

