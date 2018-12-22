import numpy as np
import logging
logger = logging.getLogger(name=__name__)
NOISE_NUGGET=1e-9


class SGMCMCSampler(object):
    """ Base Class for SGMCMC with Time Series """
    def __init__(self, **kwargs):
        raise NotImplementedError()

    def setup(self, **kwargs):
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

    def predictive_loglikelihood(self, lag=10,
            parameters=None, observations=None):
        """ Return the predictive loglikelihood given the parameters """
        if parameters is None:
            parameters = self.parameters
        if observations is None:
            observations = self.observations

        pred_loglikelihood = self.message_helper.predictive_loglikelihood(
                observations=observations,
                parameters=parameters,
                forward_message=self.forward_message,
                backward_message=self.backward_message,
                lag=lag
                )
        return pred_loglikelihood

    def noisy_loglikelihood(self,
           subsequence_length=-1,
            minibatch_size=1, buffer_length=10,
            **kwargs):
        """ Subsequence Approximation to loglikelihood

        Args:
            subsequence_length (int): length of subsequence used in evaluation
            minibatch_size (int): number of subsequences
            buffer_length (int): length of each subsequence buffer

        """
        noisy_loglikelihood = 0.0

        for s in range(0, minibatch_size):
            subsequence = self._random_subsequence_and_buffers(buffer_length,
                    subsequence_length=subsequence_length)
            normalization_factor = (subsequence['subsequence_end'] - \
                    subsequence['subsequence_start']
                    )
            # Noisy Loglikelihood should use only forward pass
            # E.g. log Pr(y) \approx \sum_s log Pr(y_s | y<min(s))
            loglikelihood_S = self.message_helper.marginal_loglikelihood(
                observations=subsequence['subsequence'],
                parameters=self.parameters,
                forward_message=subsequence['forward_message'],
                backward_message=self.backward_message,
                ) - subsequence['forward_message']['log_constant']
            noisy_loglikelihood += (
                    loglikelihood_S * self.T/(1.0*normalization_factor)
                    )
        noisy_loglikelihood *= 1.0/minibatch_size
        return noisy_loglikelihood

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

    def project_parameters(self):
        """ Project parameters to valid values + fix constants

        See **kwargs in __init__ for more details
        """
        self.parameters.project_parameters(**self.options)
        return self.parameters

    def _random_subsequence_and_buffers(self, buffer_length,
            subsequence_length):
        """ Get a subsequence and the forward and backward message approx"""
        if buffer_length == -1:
            buffer_length = self.T
        if subsequence_length == -1:
            subsequence_start = 0
            subsequence_end = self.T
        elif self.T - subsequence_length <= 0:
            subsequence_start = 0
            subsequence_end = self.T
        else:
            subsequence_start, subsequence_end = \
                    random_subsequence_and_buffers_helper(
                            subsequence_length=subsequence_length,
                            T=self.T,
                            options=self.options,
                            )

        left_buffer_start = max(0, subsequence_start - buffer_length)
        right_buffer_end = min(self.T, subsequence_end + buffer_length)

        forward_message = self.message_helper.forward_message(
                self.observations[left_buffer_start:subsequence_start],
                self.parameters,
                forward_message=self.forward_message)

        backward_message = self.message_helper.backward_message(
                self.observations[subsequence_end:right_buffer_end],
                self.parameters,
                backward_message=self.backward_message,
                )

        out = dict(
            subsequence_start = subsequence_start,
            subsequence_end = subsequence_end,
            left_buffer_start = left_buffer_start,
            right_buffer_end = right_buffer_end,
            subsequence = self.observations[subsequence_start:subsequence_end],
            forward_message = forward_message,
            backward_message = backward_message,
            )
        return out

    def _noisy_grad_loglikelihood(self, subsequence_length=-1,
            minibatch_size=1, buffer_length=0):
        noisy_grad = {var: np.zeros_like(value)
                for var, value in self.parameters.as_dict().items()}

        for s in range(0, minibatch_size):
            subsequence = self._random_subsequence_and_buffers(buffer_length,
                    subsequence_length=subsequence_length)

            gradient_kwargs = dict(
                observations=subsequence['subsequence'],
                parameters=self.parameters,
                forward_message=subsequence['forward_message'],
                backward_message=subsequence['backward_message'],
                )
            noisy_grad_add = (
                self.message_helper
                .gradient_marginal_loglikelihood(
                    **gradient_kwargs
                    )
                )
            for var in noisy_grad:
                noisy_grad[var] += noisy_grad_add[var] * 1.0*self.T/(
                    subsequence['subsequence_end'] -
                    subsequence['subsequence_start']
                    )

        for var in noisy_grad:
            noisy_grad[var] *= 1.0/minibatch_size

            if np.any(np.isnan(noisy_grad[var])):
                raise ValueError("NaNs in gradient of {0}".format(var))
            if np.linalg.norm(noisy_grad[var]) > 1e16:
                logger.warning("Norm of noisy_grad_loglike[{1} > 1e16: {0}".format(
                    noisy_grad_loglike, var))
        return noisy_grad

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
                size = value.shape
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
                epsilon * (delta[var] - correction[var]) + \
                np.sqrt(2.0*epsilon) * white_noise[var]
        return self.parameters

    def sample_gibbs(self):
        """ One Step of Blocked Gibbs Sampler

        Returns:
            parameters (Parameters): sampled parameters after one step
        """
        raise NotImplementedError()

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
            parameters (GAUSSHMMParameters): parameters of GAUSSHMM
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


