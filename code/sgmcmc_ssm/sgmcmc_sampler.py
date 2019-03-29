import numpy as np
import logging
logger = logging.getLogger(name=__name__)
NOISE_NUGGET=1e-9


class SGMCMCSampler(object):
    """ Base Class for SGMCMC with Time Series """
    def __init__(self, **kwargs):
        raise NotImplementedError()

    def setup(self, **kwargs):
        # Defines observations, prior, parameters, forward_message, backward_message
        raise NotImplementedError()

    def exact_loglikelihood(self):
        """ Return the exact loglikelihood given the current parameters """
        loglikelihood = self.message_helper.marginal_loglikelihood(
                observations=self.observations,
                parameters=self.parameters,
                forward_message=self.forward_message,
                backward_message=self.backward_message,
                )
        return loglikelihood

    def exact_logjoint(self, return_loglike=False):
        """ Return the loglikelihood + logprior given the current parameters """
        loglikelihood = self.exact_loglikelihood()
        logprior = self.prior.logprior(self.parameters)
        if return_loglike:
            return dict(
                    logjoint=loglikelihood + logprior,
                    loglikelihood=loglikelihood,
                    )
        else:
            return loglikelihood + logprior

    def predictive_loglikelihood(self, kind='marginal', lag=10,
            subsequence_length=-1, minibatch_size=1, buffer_length=10,
            num_samples=1000, parameters=None, observations=None,
            **kwargs):
        """ Return the predictive loglikelihood given the parameters """
        if parameters is None:
            parameters = self.parameters
        if observations is None:
            observations = self.observations

        T = observations.shape[0]
        if kind == 'marginal':
            pred_loglikelihood = 0.0
            for s in range(0, minibatch_size):
                out = self._random_subsequence_and_buffers(buffer_length,
                        subsequence_length=subsequence_length,
                        T=T)
                forward_message = self.message_helper.forward_message(
                        observations[
                            out['left_buffer_start']:out['subsequence_start']
                            ],
                        self.parameters,
                        forward_message=self.forward_message)
                # Noisy Loglikelihood should use only forward pass
                # E.g. log Pr(y) \approx \sum_s log Pr(y_s | y<min(s))

                pred_loglikelihood_S = (
                        self.message_helper.predictive_loglikelihood(
                        observations=observations,
                        parameters=parameters,
                        forward_message=forward_message,
                        backward_message=self.backward_message,
                        lag=lag
                        ))
                pred_loglikelihood += (
                        pred_loglikelihood_S * (T-lag)/(
                        out['subsequence_end'] - out['subsequence_start'] - lag
                        ))

            pred_loglikelihood *= 1.0/minibatch_size
            return pred_loglikelihood
        elif kind == 'pf':
            if kwargs.get("N", None) is None:
                kwargs['N'] = num_samples
            pred_loglikelihood = np.zeros(lag+1)
            for s in range(0, minibatch_size):
                out = self._random_subsequence_and_buffers(
                        buffer_length=buffer_length,
                        subsequence_length=subsequence_length,
                        T=T)
                relative_start = (out['subsequence_start'] -
                        out['left_buffer_start'])
                relative_end = (out['subsequence_end'] -
                        out['left_buffer_start'])
                buffer_ = observations[
                            out['left_buffer_start']:
                            out['right_buffer_end']
                            ]
                pred_loglike_add = (
                    self.message_helper
                    .pf_predictive_loglikelihood_estimate(
                        observations=buffer_,
                        parameters=self.parameters,
                        num_steps_ahead=lag,
                        subsequence_start=relative_start,
                        subsequence_end=relative_end,
                        **kwargs)
                    )
                for ll in range(lag+1):
                    pred_loglikelihood[ll] += pred_loglike_add[ll] * (T-ll)/(
                        out['subsequence_end'] - out['subsequence_start']-ll
                        )
            pred_loglikelihood *= 1.0/minibatch_size
            return pred_loglikelihood
        else:
            raise ValueError("Unrecognized kind = {0}".format(kind))

    def noisy_loglikelihood(self, kind='marginal',
            subsequence_length=-1, minibatch_size=1, buffer_length=10,
            num_samples=None, observations=None,
            **kwargs):
        """ Subsequence Approximation to loglikelihood

        Args:
            kind (string): how to estimate the loglikelihood
            subsequence_length (int): length of subsequence used in evaluation
            minibatch_size (int): number of subsequences
            buffer_length (int): length of each subsequence buffer

        """
        if observations is None:
            observations = self.observations
        T = observations.shape[0]
        noisy_loglikelihood = 0.0
        if kind == 'marginal':
            for s in range(0, minibatch_size):
                out = self._random_subsequence_and_buffers(buffer_length,
                        subsequence_length=subsequence_length,
                        T=T)
                forward_message = self.message_helper.forward_message(
                        observations[
                            out['left_buffer_start']:out['subsequence_start']
                            ],
                        self.parameters,
                        forward_message=self.forward_message)
                # Noisy Loglikelihood should use only forward pass
                # E.g. log Pr(y) \approx \sum_s log Pr(y_s | y<min(s))
                loglikelihood_S = self.message_helper.marginal_loglikelihood(
                    observations=observations[
                            out['subsequence_start']:out['subsequence_end']
                            ],
                    parameters=self.parameters,
                    forward_message=forward_message,
                    backward_message=self.backward_message,
                    ) - forward_message['log_constant']
                noisy_loglikelihood += (
                        loglikelihood_S * T/(
                            out['subsequence_end'] - out['subsequence_start']
                        ))
            noisy_loglikelihood *= 1.0/minibatch_size
            return noisy_loglikelihood

        elif kind == 'complete':
            for s in range(0, minibatch_size):
                out = self._random_subsequence_and_buffers(
                        buffer_length=buffer_length,
                        subsequence_length=subsequence_length,
                        T=T)


                buffer_ = observations[
                        out['left_buffer_start']:out['right_buffer_end']
                        ]

                # Draw Samples:
                latent_buffer = self.sample_x(
                        parameters=self.parameters,
                        observations=buffer_,
                        num_samples=num_samples,
                        )
                relative_start = out['subsequence_start']-out['left_buffer_start']
                relative_end = out['subsequence_end']-out['left_buffer_start']
                forward_message = {}
                if relative_start > 0:
                    forward_message = dict(
                            x_prev = latent_buffer[relative_start-1]
                            )
                loglikelihood_S = \
                    self.message_helper.complete_data_loglikelihood(
                        observations=observations[
                                out['subsequence_start']:out['subsequence_end']
                                ],
                        latent_vars=latent_buffer[relative_start:relative_end],
                        parameters=self.parameters,
                        forward_message=forward_message,
                        )


        elif kind == 'pf':
            if kwargs.get("N", None) is None:
                kwargs['N'] = num_samples

            noisy_loglikelihood = 0.0
            for s in range(0, minibatch_size):
                out = self._random_subsequence_and_buffers(
                        buffer_length=buffer_length,
                        subsequence_length=subsequence_length,
                        T=T)
                relative_start = (out['subsequence_start'] -
                        out['left_buffer_start'])
                relative_end = (out['subsequence_end'] -
                        out['left_buffer_start'])
                buffer_ = observations[
                            out['left_buffer_start']:
                            out['right_buffer_end']
                            ]
                noisy_loglike_add = (
                    self.message_helper
                    .pf_loglikelihood_estimate(
                        observations=buffer_,
                        parameters=self.parameters,
                        subsequence_start=relative_start,
                        subsequence_end=relative_end,
                        **kwargs)
                    )
                noisy_loglikelihood += noisy_loglike_add * 1.0*T/(
                        out['subsequence_end'] - out['subsequence_start']
                        )

            noisy_loglikelihood *= 1.0/minibatch_size
            if np.isnan(noisy_loglikelihood):
                raise ValueError("NaNs in loglikelihood")
            return noisy_loglikelihood
        else:
            raise ValueError("Unrecognized kind = {0}".format(kind))

    def noisy_logjoint(self, return_loglike=False, **kwargs):
        """ Return the loglikelihood + logprior given the current parameters """
        loglikelihood = self.noisy_loglikelihood(**kwargs)
        logprior = self.prior.logprior(self.parameters)
        if return_loglike:
            return dict(
                    logjoint=loglikelihood + logprior,
                    loglikelihood=loglikelihood,
                    )
        else:
            return loglikelihood + logprior

    def project_parameters(self, **kwargs):
        """ Project parameters to valid values + fix constants

        See **kwargs in __init__ for more details
        """
        self.parameters.project_parameters(**self.options, **kwargs)
        return self.parameters

    def _random_subsequence_and_buffers(self, buffer_length,
            subsequence_length, T=None):
        """ Get a subsequence and the forward and backward message approx"""
        if T is None:
            T = self.T
        if buffer_length == -1:
            buffer_length = T
        if subsequence_length == -1:
            subsequence_start = 0
            subsequence_end = T
        elif T - subsequence_length <= 0:
            subsequence_start = 0
            subsequence_end = T
        else:
            subsequence_start, subsequence_end = \
                    random_subsequence_and_buffers_helper(
                            subsequence_length=subsequence_length,
                            T=T,
                            options=self.options,
                            )

        left_buffer_start = max(0, subsequence_start - buffer_length)
        right_buffer_end = min(T, subsequence_end + buffer_length)


        out = dict(
            subsequence_start = subsequence_start,
            subsequence_end = subsequence_end,
            left_buffer_start = left_buffer_start,
            right_buffer_end = right_buffer_end,
            )
        return out

    def _noisy_grad_loglikelihood(self, kind='marginal',
            subsequence_length=-1, minibatch_size=1, buffer_length=0,
            num_samples=None, observations=None, **kwargs):
        if observations is None:
            observations = self.observations
        T = observations.shape[0]
        if kind == 'marginal':
            noisy_grad = {var: np.zeros_like(value)
                    for var, value in self.parameters.as_dict().items()}

            for s in range(0, minibatch_size):
                out = self._random_subsequence_and_buffers(buffer_length,
                        subsequence_length=subsequence_length,
                        T=T)
                forward_message = self.message_helper.forward_message(
                        observations[
                            out['left_buffer_start']:out['subsequence_start']
                            ],
                        self.parameters,
                        forward_message=self.forward_message)

                backward_message = self.message_helper.backward_message(
                        observations[
                            out['subsequence_end']:out['right_buffer_end']
                            ],
                        self.parameters,
                        backward_message=self.backward_message,
                        )

                gradient_kwargs = dict(
                    observations=observations[
                        out['subsequence_start']:out['subsequence_end']
                        ],
                    parameters=self.parameters,
                    forward_message=forward_message,
                    backward_message=backward_message,
                    )
                gradient_kwargs.update(**kwargs)
                noisy_grad_add = (
                    self.message_helper
                    .gradient_marginal_loglikelihood(
                        **gradient_kwargs
                        )
                    )
                for var in noisy_grad:
                    noisy_grad[var] += noisy_grad_add[var] * 1.0*T/(
                        out['subsequence_end'] -
                        out['subsequence_start']
                        )

            for var in noisy_grad:
                noisy_grad[var] *= 1.0/minibatch_size

                if np.any(np.isnan(noisy_grad[var])):
                    raise ValueError("NaNs in gradient of {0}".format(var))
                if np.linalg.norm(noisy_grad[var]) > 1e16:
                    logger.warning("Norm of noisy_grad_loglike[{1} > 1e16: {0}".format(
                        noisy_grad[var], var))
            return noisy_grad

        elif kind == 'complete':
            noisy_grad = {var: np.zeros_like(value)
                for var, value in self.parameters.as_dict().items()}

            for s in range(0, minibatch_size):
                out = self._random_subsequence_and_buffers(
                        buffer_length=buffer_length,
                        subsequence_length=subsequence_length,
                        T=T)


                buffer_ = observations[
                        out['left_buffer_start']:out['right_buffer_end']
                        ]
                # Draw Samples:
                latent_buffer = self.sample_x(
                        parameters=self.parameters,
                        observations=buffer_,
                        num_samples=num_samples,
                        )
                relative_start = out['subsequence_start']-out['left_buffer_start']
                relative_end = out['subsequence_end']-out['left_buffer_start']
                forward_message = {}
                if relative_start > 0:
                    forward_message = dict(
                            x_prev = latent_buffer[relative_start-1]
                            )
                noisy_grad_add = (
                    self.message_helper
                    .gradient_complete_data_loglikelihood(
                        observations=observations[
                            out['subsequence_start']:out['subsequence_end']
                            ],
                        latent_vars=latent_buffer[relative_start:relative_end],
                        parameters=self.parameters,
                        forward_message=forward_message,
                        **kwargs)
                    )
                for var in noisy_grad:
                    noisy_grad[var] += noisy_grad_add[var] * 1.0*T/(
                        out['subsequence_end'] -
                        out['subsequence_start']
                        )
            for var in noisy_grad:
                noisy_grad[var] *= 1.0/minibatch_size

                if np.any(np.isnan(noisy_grad[var])):
                    raise ValueError("NaNs in gradient of {0}".format(var))
                if np.linalg.norm(noisy_grad[var]) > 1e16:
                    logger.warning("Norm of noisy_grad_loglike[{1} > 1e16: {0}".format(
                        noisy_grad_loglike, var))
            return noisy_grad

        elif kind == 'pf':
            if kwargs.get("N", None) is None:
                kwargs['N'] = num_samples

            noisy_grad = {var: np.zeros_like(value, dtype=float)
                for var, value in self.parameters.as_dict().items()}

            for s in range(0, minibatch_size):
                out = self._random_subsequence_and_buffers(
                        buffer_length=buffer_length,
                        subsequence_length=subsequence_length,
                        T=T)
                relative_start = (out['subsequence_start'] -
                        out['left_buffer_start'])
                relative_end = (out['subsequence_end'] -
                        out['left_buffer_start'])
                buffer_ = observations[
                            out['left_buffer_start']:
                            out['right_buffer_end']
                            ]
                noisy_grad_add = (
                    self.message_helper
                    .pf_score_estimate(
                        observations=buffer_,
                        parameters=self.parameters,
                        subsequence_start=relative_start,
                        subsequence_end=relative_end,
                        **kwargs)
                    )
                for var in noisy_grad:
                    noisy_grad[var] += noisy_grad_add[var] * 1.0*T/(
                        out['subsequence_end'] -
                        out['subsequence_start']
                        )

            for var in noisy_grad:
                noisy_grad[var] *= 1.0/minibatch_size
                if np.any(np.isnan(noisy_grad[var])):
                    raise ValueError("NaNs in gradient of {0}".format(var))
                if np.linalg.norm(noisy_grad[var]) > 1e16:
                    logger.warning("Norm of noisy_grad_[{1}] > 1e16: {0}".format(
                        noisy_grad[var], var))
            return noisy_grad
        else:
            raise ValueError("Unrecognized kind = {0}".format(kind))

    def noisy_gradient(self, preconditioner=None, is_scaled=True, **kwargs):
        """ Noisy Gradient Estimate

        noisy_gradient = -grad tilde{U}(theta)
                       = grad marginal loglike + grad logprior

        Monte Carlo Estimate of gradient (using buffering)

        Args:
            preconditioner (object): preconditioner for gradients
            is_scaled (boolean): scale gradient by 1/T
            **kwargs: arguments for `self._noisy_grad_loglikelihood()`
                For example: minibatch_size, buffer_length, use_analytic

        Returns:
            noisy_gradient (dict): dict of gradient vectors

        """
        noisy_grad_loglike = \
                self._noisy_grad_loglikelihood(**kwargs)
        noisy_grad_prior = self.prior.grad_logprior(
                parameters=self.parameters)
        noisy_gradient = {var: noisy_grad_prior[var] + noisy_grad_loglike[var]
                for var in noisy_grad_prior}

        if preconditioner is None:
            if is_scaled:
                for var in noisy_gradient:
                    noisy_gradient[var] /= self.T
        else:
            scale = 1.0/self.T if is_scaled else 1.0
            noisy_gradient = preconditioner.precondition(noisy_gradient,
                    parameters=self.parameters,
                    scale=scale)

        return noisy_gradient

    def step_sgd(self, epsilon, **kwargs):
        """ One step of Stochastic Gradient Descent

        (Learns the MAP, not a sample from the posterior)

        Args:
            epsilon (double): step size
            **kwargs (kwargs): to pass to self.noisy_gradient
                minibatch_size (int): number of subsequences to sample from
                buffer_length (int): length of buffer to use

        Returns:
            parameters (Parameters): sampled parameters after one step
        """
        delta = self.noisy_gradient(**kwargs)
        for var in self.parameters.var_dict:
            self.parameters.var_dict[var] += epsilon * delta[var]
        return self.parameters

    def step_precondition_sgd(self, epsilon, preconditioner, **kwargs):
        """ One Step of Preconditioned Stochastic Gradient Descent

        Args:
            epsilon (double): step size
            preconditioner (object): preconditioner
            **kwargs (kwargs): to pass to self.noisy_gradient
                minibatch_size (int): number of subsequences to sample from
                buffer_length (int): length of buffer to use

        Returns:
            parameters (Parameters): sampled parameters after one step
        """
        delta = self.noisy_gradient(preconditioner=preconditioner, **kwargs)
        for var in self.parameters.var_dict:
            self.parameters.var_dict[var] += epsilon * delta[var]
        return self.parameters

    def step_adagrad(self, epsilon, **kwargs):
        """ One step of adagrad

        (Learns the MAP, not a sample from the posterior)

        Args:
            epsilon (double): step size
            **kwargs (kwargs): to pass to self.noisy_gradient
        """
        if not hasattr(self, "_adagrad_moments"):
            self._adagrad_moments = dict(t=0, G=0.0)

        g = self.parameters.from_dict_to_vector(self.noisy_gradient(**kwargs))
        t = self._adagrad_moments['t'] + 1
        G = self._adagrad_moments['G'] + g**2

        delta_vec = g/np.sqrt(G + NOISE_NUGGET)
        delta = self.parameters.from_vector_to_dict(delta_vec,
                **self.parameters.dim)
        for var in self.parameters.var_dict:
            self.parameters.var_dict[var] += epsilon * delta[var]
        self._adagrad_moments['t'] = t
        self._adagrad_moments['G'] = G
        return self.parameters

    def _get_sgmcmc_noise(self, is_scaled=True, preconditioner=None,
            **kwargs):
        if is_scaled:
            scale = 1.0 / self.T
        else:
            scale = 1.0

        if preconditioner is not None:
            white_noise = preconditioner.precondition_noise(
                    parameters=self.parameters,
                    scale=scale,
                    )
        else:
            white_noise = {var: np.random.normal(
                loc=0,
                scale=np.sqrt(scale),
                size=value.shape
                ) for var, value in self.parameters.as_dict().items()}

        return white_noise

    def sample_sgld(self, epsilon, **kwargs):
        """ One Step of Stochastic Gradient Langevin Dynamics

        Args:
            epsilon (double): step size
            **kwargs (kwargs): to pass to self.noisy_gradient

        Returns:
            parameters (Parameters): sampled parameters after one step
        """
        if "preconditioner" in kwargs:
            raise ValueError("Use SGRLD instead")
        delta = self.noisy_gradient(**kwargs)
        white_noise = self._get_sgmcmc_noise(**kwargs)

        for var in self.parameters.var_dict:
            self.parameters.var_dict[var] += \
                epsilon * delta[var] + np.sqrt(2.0*epsilon) * white_noise[var]
        return self.parameters

    def sample_sgrld(self, epsilon, preconditioner, **kwargs):
        """ One Step of Stochastic Gradient Riemannian Langevin Dynamics

        theta += epsilon * D(theta) * (grad_logjoint - correction_term) + \
                N(0, 2 epsilon D(theta))

        Args:
            epsilon (double): step size
            preconditioner (object): preconditioner

        Returns:
            parameters (Parameters): sampled parameters after one step
        """
        if kwargs.get("is_scaled", True):
            scale = 1.0 / self.T
        else:
            scale = 1.0

        delta = self.noisy_gradient(preconditioner=preconditioner, **kwargs)
        white_noise = self._get_sgmcmc_noise(
                preconditioner=preconditioner, **kwargs)
        correction = preconditioner.correction_term(
                self.parameters, scale=scale)
        for var in self.parameters.var_dict:
            self.parameters.var_dict[var] += \
                epsilon * (delta[var] + correction[var]) + \
                np.sqrt(2.0*epsilon) * white_noise[var]
        return self.parameters

    def sample_gibbs(self):
        """ One Step of Blocked Gibbs Sampler

        Returns:
            parameters (Parameters): sampled parameters after one step
        """
        raise NotImplementedError()

class SeqSGMCMCSampler(object):
    """ Mixin for handling a list of sequences """
    def setup(self, observations, prior, parameters=None, **kwargs):
        """ Initialize the sampler

        Args:
            observations (list of ndarray): list of L observations ndarrays
            prior (Prior)
            parameters (Parameters)
        """
        super().setup(observations[0], prior, parameters=parameters, **kwargs)
        self.observations = observations
        self.T = np.sum([observation.shape[0] for observation in observations])
        return

    def exact_loglikelihood(self, observations=None):
        """ Return exact loglikelihood over all observation sequences """
        if observations is None:
            observations = self.observations
        loglikelihood = 0
        for observation in observations:
            loglikelihood += self.message_helper.marginal_loglikelihood(
                    observations=observation,
                    parameters=self.parameters,
                    forward_message=self.forward_message,
                    backward_message=self.backward_message,
                    )
        return loglikelihood

    def noisy_loglikelihood(self, num_sequences=-1, observations=None,
            **kwargs):
        """ Subsequence Approximation to loglikelihood

        Args:
            num_sequences (int): how many observation sequences to use
                (default = -1) is to use all observation sequences
        """
        if observations is None:
            observations = self.observations
        loglikelihood = 0
        T = 0.0
        sequence_indices = np.arange(len(observations))
        if num_sequences != -1:
            sequence_indices = np.random.choice(
                    sequence_indices, num_sequences, replace=False,
                    )
        for sequence_index in sequence_indices:
            T += observations[sequence_index].shape[0]
            loglikelihood += super().noisy_loglikelihood(
                    observations=observations[sequence_index],
                    **kwargs)
        if num_sequences != -1:
            loglikelihood *= self.T / T
        return loglikelihood

    def predictive_loglikelihood(self, num_sequences=-1, observations=None, **kwargs):
        """ Return the predictive loglikelihood given the parameters """
        if observations is None:
            observations = self.observations
        predictive_loglikelihood = 0
        T = 0.0
        sequence_indices = np.arange(len(observations))
        if num_sequences != -1:
            sequence_indices = np.random.choice(
                    sequence_indices, num_sequences, replace=False,
                    )
        for sequence_index in sequence_indices:
            T += observations[sequence_index].shape[0]
            predictive_loglikelihood += super().predictive_loglikelihood(
                    observations=observations[sequence_index],
                    **kwargs)
        if num_sequences != -1:
            predictive_loglikelihood *= self.T / T
        return predictive_loglikelihood

    def _noisy_grad_loglikelihood(self, num_sequences=-1, **kwargs):
        """ Subsequence approximation to gradient of loglikelihood

        Args:
            num_sequences (int): how many observation sequences to use
                (default = -1) is to use all observation sequences
        """
        noisy_grad_loglike = None
        T = 0.0

        sequence_indices = np.arange(len(self.observations))
        if num_sequences != -1:
            sequence_indices = np.random.choice(
                    sequence_indices, num_sequences, replace=False,
                    )
        for sequence_index in sequence_indices:
            noisy_grad_index = super()._noisy_grad_loglikelihood(
                    observations=self.observations[sequence_index],
                    **kwargs)
            T += self.observations[sequence_index].shape[0]
            if noisy_grad_loglike is None:
                noisy_grad_loglike = {var: noisy_grad_index[var]
                        for var in noisy_grad_index.keys()
                        }
            else:
                noisy_grad_loglike = {
                        var: noisy_grad_loglike[var] + noisy_grad_index[var]
                        for var in noisy_grad_index.keys()
                        }
        if num_sequences != -1:
            noisy_grad_loglike = {
                    var: noisy_grad_loglike[var] * self.T / T
                    for var in noisy_grad_index.keys()
                    }
        return noisy_grad_loglike

class SGMCMCHelper(object):
    """ Base Class for SGMCMC Helper """
    def __init__(self, **kwargs):
        raise NotImplementedError()

    def forward_message(self, observations, parameters, forward_message=None,
            **kwargs):
        """ Calculate forward messages over the observations

        Pr(u_t | y_{<=t}) for y_t in observations

        Args:
            observations (ndarray): observations
            parameters (parameters): parameters
            forward_message (dict): latent state prior Pr(u_{-1} | y_{<=-1})

        Returns:
            forward_message (dict): same format as forward_message
        """
        if forward_message is None:
            forward_message = self.default_forward_message
        if np.shape(observations)[0] == 0: return forward_message
        forward_message = self._forward_message(
                observations=observations,
                parameters=parameters,
                forward_message=forward_message,
                **kwargs
                )
        return forward_message

    def backward_message(self, observations, parameters, backward_message=None,
            **kwargs):
        """ Calculate backward messages over the observations

        Pr(y_{>t} | u_t) for y_t in observations

        Args:
            observations (ndarray): observations
            parameters (parameters): parameters
            backward_message (dict): backward message Pr(y_{>T-1} | u_{T-1})

        Returns:
            backward_message (dict): same format as forward_message
        """

        if backward_message is None:
            backward_message = self.default_backward_message
        if np.shape(observations)[0] == 0: return backward_message
        backward_message = self._backward_message(
                observations=observations,
                parameters=parameters,
                backward_message=backward_message,
                **kwargs
                )
        return backward_message

    def forward_pass(self, observations, parameters,
            forward_message=None, include_init_message=False, **kwargs):
        """ Calculate forward messages over the observations

        Pr(u_t | y_{<=t}) for y_t in observations

        Args:
            observations (ndarray): observations
            parameters (parameters): parameters
            forward_message (dict): latent state prior Pr(u_{-1} | y_{<=-1})
            include_init_message (boolean) whether to include t = -1

        Returns:
            forward_messages (list of dict): same format as forward_message
        """
        if forward_message is None:
            forward_message = self.default_forward_message

        if np.shape(observations)[0] == 0:
            if include_init_message:
                return [forward_message]
            else:
                return []

        forward_messages = self._forward_messages(
                    observations=observations,
                    parameters=parameters,
                    forward_message=forward_message,
                    **kwargs
                    )
        if include_init_message:
            return forward_messages
        else:
            return forward_messages[1:]

    def backward_pass(self, observations, parameters,
            backward_message=None, include_init_message=False, **kwargs):
        """ Calculate backward message over the observations

        Pr(y_{>t} | u_t) for y_t in observations

        Args:
            observations (ndarray): observations
            parameters (parameters): parameters
            backward_message (dict): backward message Pr(y_{>T-1} | u_{T-1})
            include_init_message (boolean) whether to include t = -1

        Returns:
            backward_messages (list of dict): same format as backward_message
        """
        if backward_message is None:
            backward_message = self.default_backward_message
        if np.shape(observations)[0] == 0:
            if include_init_message:
                return [backward_message]
            else:
                return []
        backward_messages = self._backward_messages(
                observations=observations,
                parameters=parameters,
                backward_message=backward_message,
                **kwargs
                )
        if include_init_message:
            return backward_messages
        else:
            return backward_messages[1:]

    def marginal_loglikelihood(self, observations, parameters,
            forward_message, backward_message):
        raise NotImplementedError()

    def predictive_loglikelihood(self, observations, parameters,
            forward_message, backward_message, lag):
        raise NotImplementedError()

    def gradient_marginal_loglikelihood(self, observations, parameters,
            forward_message, backward_message, **kwargs):
        """ Gradient Calculation

        Gradient of log Pr(y_[0:T) | y_<0, y_>=T, parameters)

        Args:
            observations (ndarray): num_obs observations
            parameters (Parameters): parameters
            forward_message (dict): Pr(u_-1, y_<0 | parameters)
            backward_message (dict): Pr(y_>T | u_T, parameters)

        Returns
            grad (dict): grad of variables in parameters

        """
        raise NotImplementedError()

    def parameters_gibbs_sample(self, observations, latent_vars, prior,
            **kwargs):
        """ Gibbs sample parameters based on data

        Samples parameters from the posterior conditional distribution
            theta ~ Pr(theta | y, u)

        Args:
            observations (ndarray): num_obs observations
            latent_vars (ndarray): num_obs latent variables
            prior (prior): prior

        Returns
            sample_parameters (parameters): sampled parameters
        """

        sufficient_stat = self.calc_gibbs_sufficient_statistic(
            observations, latent_vars, **kwargs,
            )
        sample_parameters = prior.sample_posterior(
                sufficient_stat, **kwargs,
                )
        return sample_parameters

    def calc_gibbs_sufficient_statistic(self, observations, latent_vars,
            **kwargs):
        raise NotImplementedError()

    def latent_var_sample(self, observations, parameters,
            forward_message, backward_message,
            distribution="smoothed", tqdm=None):
        raise NotImplementedError()

    def y_sample(self, observations, parameters,
            forward_message, backward_message,
            latent_var=None, distribution="smoothed", tqdm=None):
        raise NotImplementedError()

    def latent_var_marginal(self, observations, parameters,
            forward_message, backward_message,
            distribution="smoothed", tqdm=None):
        raise NotImplementedError()

    def y_marginal(self, observations, parameters,
            forward_message, backward_message,
            latent_var=None, distribution="smoothed", tqdm=None):
        raise NotImplementedError()

    def _forward_messages(self, observations, parameters, forward_message,
            **kwargs):
        raise NotImplementedError()

    def _backward_messages(self, observations, parameters, backward_message,
            **kwargs):
        raise NotImplementedError()

    def _forward_message(self, observations, parameters, forward_message,
            **kwargs):
        # Override for memory savings
        return self._forward_messages(observations, parameters, forward_message,
                **kwargs)[-1]

    def _backward_message(self, observations, parameters, backward_message,
            **kwargs):
        # Override for memory savings
        return self._backward_messages(observations, parameters,
                backward_message, **kwargs)[0]

    def pf_score_estimate(self, observations, parameters,
            subsequence_start=0, subsequence_end=None,
            pf="poyiadjis_N", N=100, kernel='prior',
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
        raise NotImplementedError()

    def pf_loglikelihood_estimate(self, observations, parameters,
            subsequence_start=0, subsequence_end=None,
            pf="poyiadjis_N", N=1000, kernel='prior',
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
        raise NotImplementedError()

    def pf_predictive_loglikelihood_estimate(self, observations, parameters,
            num_steps_ahead=5, subsequence_start=0, subsequence_end=None,
            pf="pf_filter", N=1000, kernel=None,
            **kwargs):
        """ Particle Filter Predictive Log-Likelihoood Estimate

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
        raise NotImplementedError()

    def pf_latent_var_marginal(self, observations, parameters,
            subsequence_start=0, subsequence_end=None,
            pf="poyiadjis_N", N=100, kernel='prior',
            **kwargs):
        raise NotImplementedError()


def random_subsequence_and_buffers_helper(subsequence_length, T, options):
    if options.get("strict_partition", False):
        if T % subsequence_length != 0:
            raise ValueError(
        "subsequence_length {0} does not evenly divide T {1}".format(
            subsequence_length, T)
        )
        subsequence_start = \
            np.random.choice(np.arange(0, T//subsequence_length)) * \
            subsequence_length
        subsequence_end = subsequence_start + subsequence_length
    elif options.get("naive_partition", False):
        subsequence_start = \
                np.random.randint(0, T-subsequence_length)
        subsequence_end = subsequence_start + subsequence_length
    else:
        left_offset = np.random.choice(np.arange(
                subsequence_length//2,
                subsequence_length + subsequence_length//2))
        right_offset = (T-left_offset) % subsequence_length
        if right_offset < subsequence_length//2:
            right_offset += subsequence_length

        if right_offset + left_offset > T:
            left_offset = T
            right_offset = 0

        # Check if we sample end points
        choice = np.random.choice(
                ['left', 'middle', 'right'],
                p=1.0*np.array([
                    left_offset,
                    T-left_offset-right_offset,
                    right_offset])/T,
                )
        if choice == 'middle':
            T_offset = T-left_offset-right_offset
            subsequence_start = np.random.choice(
                    np.arange(0,T_offset//subsequence_length)
                    ) * subsequence_length + left_offset
            subsequence_end = subsequence_start + subsequence_length
        elif choice == 'left':
            subsequence_start = 0
            subsequence_end = left_offset
        else: # choice == "right"
            subsequence_start = T-right_offset
            subsequence_end = T

    return int(subsequence_start), int(subsequence_end)


