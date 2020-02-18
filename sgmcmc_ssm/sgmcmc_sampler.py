import numpy as np
import pandas as pd
import time
from datetime import timedelta
import logging
from .evaluator import BaseEvaluator
logger = logging.getLogger(name=__name__)
NOISE_NUGGET=1e-9


# SGMCMCSampler
class SGMCMCSampler(object):
    """ Base Class for SGMCMC with Time Series """
    def __init__(self, **kwargs):
        raise NotImplementedError()

    ## Init Functions
    def setup(self, **kwargs):
        # Depreciated
        raise NotImplementedError()

    def prior_init(self):
        self.parameters = self.prior.sample_prior()
        return self.parameters

    ## Loglikelihood Functions
    def exact_loglikelihood(self, tqdm=None):
        """ Return the exact loglikelihood given the current parameters """
        loglikelihood = self.message_helper.marginal_loglikelihood(
                observations=self.observations,
                parameters=self.parameters,
                forward_message=self.forward_message,
                backward_message=self.backward_message,
                tqdm=tqdm,
                )
        return loglikelihood

    def exact_logjoint(self, return_loglike=False, tqdm=None):
        """ Return the loglikelihood + logprior given the current parameters """
        loglikelihood = self.exact_loglikelihood(tqdm=tqdm)
        logprior = self.prior.logprior(self.parameters)
        if return_loglike:
            return dict(
                    logjoint=loglikelihood + logprior,
                    loglikelihood=loglikelihood,
                    )
        else:
            return loglikelihood + logprior

    def predictive_loglikelihood(self, kind='marginal', num_steps_ahead=10,
            subsequence_length=-1, minibatch_size=1, buffer_length=10,
            num_samples=1000, parameters=None, observations=None,
            **kwargs):
        """ Return the predictive loglikelihood given the parameters """
        if parameters is None:
            parameters = self.parameters
        observations = self._get_observations(observations)

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
                        forward_message=self.forward_message,
                        tqdm=kwargs.get('tqdm', None),
                        )
                # Noisy Loglikelihood should use only forward pass
                # E.g. log Pr(y) \approx \sum_s log Pr(y_s | y<min(s))

                pred_loglikelihood_S = (
                        self.message_helper.predictive_loglikelihood(
                        observations=observations,
                        parameters=parameters,
                        forward_message=forward_message,
                        backward_message=self.backward_message,
                        lag=num_steps_ahead,
                        tqdm=kwargs.get('tqdm', None),
                        ))
                pred_loglikelihood += (
                        pred_loglikelihood_S * (T-num_steps_ahead)/(
                        out['subsequence_end'] - out['subsequence_start'] - \
                                num_steps_ahead
                        ))

            pred_loglikelihood *= 1.0/minibatch_size
            return pred_loglikelihood
        elif kind == 'pf':
            if kwargs.get("N", None) is None:
                kwargs['N'] = num_samples
            pred_loglikelihood = np.zeros(num_steps_ahead+1)
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
                        num_steps_ahead=num_steps_ahead,
                        subsequence_start=relative_start,
                        subsequence_end=relative_end,
                        **kwargs)
                    )
                for ll in range(num_steps_ahead+1):
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
        observations = self._get_observations(observations)

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
                        forward_message=self.forward_message,
                        tqdm=kwargs.get('tqdm', None),
                        )
                # Noisy Loglikelihood should use only forward pass
                # E.g. log Pr(y) \approx \sum_s log Pr(y_s | y<min(s))
                noisy_loglikelihood += (
                self.message_helper.marginal_loglikelihood(
                    observations=observations[
                            out['subsequence_start']:out['subsequence_end']
                            ],
                    parameters=self.parameters,
                    weights=out['weights'],
                    forward_message=forward_message,
                    backward_message=self.backward_message,
                    tqdm=kwargs.get('tqdm', None),
                    ) - forward_message['log_constant']
                )

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
                noisy_loglikelihood += \
                    self.message_helper.complete_data_loglikelihood(
                        observations=observations[
                                out['subsequence_start']:out['subsequence_end']
                                ],
                        latent_vars=latent_buffer[relative_start:relative_end],
                        weights=out['weights'],
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
                noisy_loglikelihood += (
                    self.message_helper
                    .pf_loglikelihood_estimate(
                        observations=buffer_,
                        parameters=self.parameters,
                        weights=out['weights'],
                        subsequence_start=relative_start,
                        subsequence_end=relative_end,
                        **kwargs)
                    )
        else:
            raise ValueError("Unrecognized kind = {0}".format(kind))

        noisy_loglikelihood *= 1.0/minibatch_size
        if np.isnan(noisy_loglikelihood):
            raise ValueError("NaNs in loglikelihood")
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

    ## Gradient Functions
    def _random_subsequence_and_buffers(self, buffer_length,
            subsequence_length, T=None):
        """ Get a subsequence and the forward and backward message approx"""
        if T is None:
            T = self._get_T()
        if buffer_length == -1:
            buffer_length = T
        if (subsequence_length == -1) or (T-subsequence_length <= 0):
            subsequence_start = 0
            subsequence_end = T
            weights = None
        else:
            subsequence_start, subsequence_end, weights = \
                random_subsequence_and_weights(
                        S=subsequence_length,
                        T=T,
                        partition_style=self.options.get('partition_style'),
                        )

        left_buffer_start = max(0, subsequence_start - buffer_length)
        right_buffer_end = min(T, subsequence_end + buffer_length)

        out = dict(
            subsequence_start = subsequence_start,
            subsequence_end = subsequence_end,
            left_buffer_start = left_buffer_start,
            right_buffer_end = right_buffer_end,
            weights = weights,
            )
        return out

    def _single_noisy_grad_loglikelihood(self, buffer_dict, kind='marginal',
            num_samples=None, observations=None, parameters=None, **kwargs):
        # buffer_dict is the output of _random_subsequence_and_buffers
        observations = self._get_observations(observations, check_shape=False)

        if parameters is None:
            parameters = self.parameters
        T = observations.shape[0]
        if kind == 'marginal':
            forward_message = self.message_helper.forward_message(
                    observations[
                        buffer_dict['left_buffer_start']:
                        buffer_dict['subsequence_start']
                        ],
                    parameters,
                    forward_message=self.forward_message)

            backward_message = self.message_helper.backward_message(
                    observations[
                        buffer_dict['subsequence_end']:
                        buffer_dict['right_buffer_end']
                        ],
                    parameters,
                    backward_message=self.backward_message,
                    )

            noisy_grad = (
                self.message_helper
                .gradient_marginal_loglikelihood(
                    observations=observations[
                        buffer_dict['subsequence_start']:
                        buffer_dict['subsequence_end']
                        ],
                    parameters=parameters,
                    weights=buffer_dict['weights'],
                    forward_message=forward_message,
                    backward_message=backward_message,
                    **kwargs
                    )
                )
        elif kind == 'complete':
            buffer_ = observations[
                    buffer_dict['left_buffer_start']:
                    buffer_dict['right_buffer_end']
                    ]
            # Draw Samples:
            latent_buffer = self.sample_x(
                    parameters=parameters,
                    observations=buffer_,
                    num_samples=num_samples,
                    )
            relative_start = (buffer_dict['subsequence_start'] -
                    buffer_dict['left_buffer_start'])
            relative_end = (buffer_dict['subsequence_end'] -
                    buffer_dict['left_buffer_start'])
            forward_message = {}
            if relative_start > 0:
                forward_message = dict(
                        x_prev = latent_buffer[relative_start-1]
                        )
            noisy_grad = (
                self.message_helper
                .gradient_complete_data_loglikelihood(
                    observations=observations[
                        buffer_dict['subsequence_start']:
                        buffer_dict['subsequence_end']
                        ],
                    latent_vars=latent_buffer[relative_start:relative_end],
                    parameters=parameters,
                    weights=buffer_dict['weights'],
                    forward_message=forward_message,
                    **kwargs)
                )

        elif kind == 'pf':
            if kwargs.get("N", None) is None:
                kwargs['N'] = num_samples
            relative_start = (buffer_dict['subsequence_start'] -
                    buffer_dict['left_buffer_start'])
            relative_end = (buffer_dict['subsequence_end'] -
                    buffer_dict['left_buffer_start'])
            buffer_ = observations[
                        buffer_dict['left_buffer_start']:
                        buffer_dict['right_buffer_end']
                        ]
            noisy_grad = (
                self.message_helper
                .pf_gradient_estimate(
                    observations=buffer_,
                    parameters=self.parameters,
                    subsequence_start=relative_start,
                    subsequence_end=relative_end,
                    weights=buffer_dict['weights'],
                    **kwargs)
                )
        else:
            raise ValueError("Unrecognized kind = {0}".format(kind))

        return noisy_grad

    def _noisy_grad_loglikelihood(self,
            subsequence_length=-1, minibatch_size=1, buffer_length=0,
            observations=None, buffer_dicts=None, **kwargs):
        observations = self._get_observations(observations, check_shape=False)

        T = observations.shape[0]

        if buffer_dicts is None:
            buffer_dicts = [
                self._random_subsequence_and_buffers(
                    buffer_length=buffer_length,
                    subsequence_length=subsequence_length,
                    T=T)
                for _ in range(minibatch_size)
                ]
        elif len(buffer_dicts) != minibatch_size:
            raise ValueError("len(buffer_dicts != minibatch_size")

        noisy_grad = {var: np.zeros_like(value)
                for var, value in self.parameters.as_dict().items()}

        for s in range(0, minibatch_size):
            noisy_grad_add = self._single_noisy_grad_loglikelihood(
                    buffer_dict=buffer_dicts[s],
                    observations=observations,
                    **kwargs,
                    )
            for var in noisy_grad:
                noisy_grad[var] += noisy_grad_add[var] * 1.0/minibatch_size

        if np.any(np.isnan(noisy_grad[var])):
            raise ValueError("NaNs in gradient of {0}".format(var))
        if np.linalg.norm(noisy_grad[var]) > 1e16:
            logger.warning("Norm of noisy_grad_loglike[{1} > 1e16: {0}".format(
                noisy_grad[var], var))
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
                parameters=kwargs.get('parameters',self.parameters),
                **kwargs
                )
        noisy_gradient = {var: noisy_grad_prior[var] + noisy_grad_loglike[var]
                for var in noisy_grad_prior}

        if preconditioner is None:
            if is_scaled:
                for var in noisy_gradient:
                    noisy_gradient[var] /= self._get_T(**kwargs)
        else:
            scale = 1.0/self._get_T(**kwargs) if is_scaled else 1.0
            noisy_gradient = preconditioner.precondition(noisy_gradient,
                    parameters=kwargs.get('parameters',self.parameters),
                    scale=scale)

        return noisy_gradient

    ## Sampler/Optimizer Step Functions
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
            scale = 1.0 / self._get_T(**kwargs)
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

    def sample_sgld_cv(self, epsilon, centering_parameters, centering_gradient,
            **kwargs):
        """ One Step of Stochastic Gradient Langevin Dynamics with Control Variates

        grad = full_gradient(centering_parameters) + \
                sub_gradient(parameters) - sub_gradient(centering_gradient)

        Args:
            epsilon (double): step size
            centering_parameters (Parameters): centering parameters
            centering_gradient (dict): full data grad of centering_parameters
            **kwargs (kwargs): to pass to self.noisy_gradient

        Returns:
            parameters (Parameters): sampled parameters after one step
        """
        if "preconditioner" in kwargs:
            raise ValueError("Use SGRLD instead")
        buffer_dicts = [
            self._random_subsequence_and_buffers(
                buffer_length=kwargs.get('buffer_length', 0),
                subsequence_length=kwargs.get('subsequence_length', -1),
                T=self._get_T(**kwargs),
                )
            for _ in range(kwargs.get('minibatch_size', 1))
            ]

        cur_subseq_grad = self.noisy_gradient(
                buffer_dicts=buffer_dicts, **kwargs)
        centering_subseq_grad = self.noisy_gradient(
                parameters=centering_parameters,
                buffer_dicts=buffer_dicts, **kwargs)

        delta = {}
        for var in cur_subseq_grad.keys():
            delta[var] = centering_gradient[var] + \
                    cur_subseq_grad[var] - centering_subseq_grad[var]

        white_noise = self._get_sgmcmc_noise(**kwargs)
        for var in self.parameters.var_dict:
            self.parameters.var_dict[var] += \
                epsilon * delta[var] + np.sqrt(2.0*epsilon) * white_noise[var]
        return self.parameters

    def sample_sgrld(self, epsilon, preconditioner, **kwargs):
        """ One Step of Stochastic Gradient Riemannian Langevin Dynamics

        theta += epsilon * (D(theta) * grad_logjoint + correction_term) + \
                N(0, 2 epsilon D(theta))

        Args:
            epsilon (double): step size
            preconditioner (object): preconditioner

        Returns:
            parameters (Parameters): sampled parameters after one step
        """
        if kwargs.get("is_scaled", True):
            scale = 1.0 / self._get_T(**kwargs)
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

    def project_parameters(self, **kwargs):
        """ Project parameters to valid values + fix constants

        See **kwargs in __init__ for more details
        """
        self.parameters.project_parameters(**self.options, **kwargs)
        return self.parameters

    ## Fit Functions
    def fit(self, iter_type, num_iters, output_all=False, observations=None,
            init_parameters=None, tqdm=None, catch_interrupt=False, **kwargs):
        """ Run multiple learning / inference steps

        Args:
            iter_type (string):
                'SGD', 'ADAGRAD', 'SGLD', 'SGRLD', 'Gibbs', etc.
            num_iters (int): number of steps
            output_all (bool): whether to output each iteration's parameters
            observations (ndarray): observations to fit on, optional
            init_parameters (Parameters): initial parameters, optional
            tqdm (tqdm): progress bar wrapper
            catch_interrupt (bool): terminate early on Ctrl-C
            **kwargs: for each iter
                e.g. steps_per_iter, epsilon, minibatch_size,
                    subsequence_length, buffer_length,
                    preconditioner, pf_kwargs, etc.
                see documentation for get_iter_step()

        Returns: (depends on output_all arg)
            parameters (Parameters):
            parameters_list (list of Parameters): length num_iters+1
        """
        if observations is not None:
            self.observations = observations
        if init_parameters is not None:
            self.parameters = init_parameters.copy()

        iter_func_names, iter_func_kwargs = \
                self.get_iter_step(iter_type, tqdm=tqdm, **kwargs)

        if output_all:
            parameters_list = [None]*(num_iters+1)
            parameters_list[0] = self.parameters.copy()

        # Fit Loop
        pbar = range(1, num_iters+1)
        if tqdm is not None:
            pbar = tqdm(pbar)
            pbar.set_description("fit using {0} iters".format(iter_type))
        for it in pbar:
            # Run iter funcs
            try:
                for func_name, func_kwargs in zip(iter_func_names,
                        iter_func_kwargs):
                    getattr(self, func_name)(**func_kwargs)
                if output_all:
                    parameters_list[it] = self.parameters.copy()
            except KeyboardInterrupt as e:
                if catch_interrupt:
                    logger.warning("Interrupt in fit:\n{0}\n".format(e) + \
                            "Stopping early after {0} iters".format(it))
                    if output_all:
                        return parameters_list[:it]
                    else:
                        return self.parameters.copy()
                else:
                    raise e

        if output_all:
            return parameters_list
        else:
            return self.parameters.copy()

    def fit_timed(self, iter_type, max_time=60, min_save_time=1,
            observations=None, init_parameters=None, tqdm=None, tqdm_iter=False,
            catch_interrupt=False,
            **kwargs):
        """ Run multiple learning / inference steps

        Args:
            iter_type (string):
                'SGD', 'ADAGRAD', 'SGLD', 'SGRLD', 'Gibbs', etc.
            max_time (float): maxium time in seconds to run fit
            min_save_time (float): min time between saved parameters
            observations (ndarray): observations to fit on, optional
            init_parameters (Parameters): initial parameters, optional
            tqdm (tqdm): progress bar wrapper
            catch_interrupt (bool): terminate early on Ctrl-C
            **kwargs: for each iter
                e.g. steps_per_iter, epsilon, minibatch_size,
                    subsequence_length, buffer_length,
                    preconditioner, pf_kwargs, etc.
                see documentation for get_iter_step()

        Returns:
            parameters_list (list of Parameters):
            times (list of float): fit time for each parameter
        """
        parameters_list, times, _ = self.fit_evaluate(
                iter_type=iter_type,
                max_time=max_time, min_save_time=min_save_time,
                observations=observations, init_parameters=init_parameters,
                tqdm=tqdm, tqdm_iter=tqdm_iter,
                catch_interrupt=catch_interrupt,
                **kwargs)
        return parameters_list['parameters'].tolist(), times['time'].tolist()

    def fit_evaluate(self, iter_type, metric_functions=None,
            max_num_iters=None, max_time=60, min_save_time=1,
            observations=None, init_parameters=None, tqdm=None, tqdm_iter=False,
            catch_interrupt=False, total_max_time=None,
            **kwargs):
        """ Run multiple learning / inference steps with evaluator

        Args:
            iter_type (string):
                'SGD', 'ADAGRAD', 'SGLD', 'SGRLD', 'Gibbs', etc.
            metric_functions (func or list of funcs): evaluation functions
                Each function takes a sampler and returns a dict or list of dict
                    dict(metric=string, variable=string, value=double) for each
                See metric_functions.py for examples

            max_num_iters (int): maximum number of iterations to save
            max_time (float): maxium time in seconds to run sampler
                does *not* include time used by evaluator
            min_save_time (float): min time between saved parameters
            observations (ndarray): observations to fit on, optional
            init_parameters (Parameters): initial parameters, optional
            tqdm (tqdm): progress bar wrapper
            tqdm_iter (bool): progress bar for each iteration
            catch_interrupt (bool): terminate early on Ctrl-C
            total_max_time (float): maximum time in seconds to run fit_evaluate
            **kwargs: for each iter
                e.g. steps_per_iter, epsilon, minibatch_size,
                    subsequence_length, buffer_length,
                    preconditioner, pf_kwargs, etc.
                see documentation for get_iter_step()

        Returns:
            parameters_list (pd.DataFrame): parameters saved
                columns:
                    iteration: number of iter_func_kwargs steps called
                    parameters: Parameters

            times (pd.DataFrame): fit time for each saved parameter
                columns:
                    iteration: number of iter_func_kwargs steps called
                    time: time used by iter_func_kwargs

            metrics (pd.DataFrame): metric for each saved parameter
                columns:
                    iteration: number of iter_func_kwargs steps called
                    metric: name of metric
                    variable: name of variable
                    value:  value of metric for variable
        """
        if observations is not None:
            self.observations = observations
        if init_parameters is not None:
            self.parameters = init_parameters.copy()


        evaluator = BaseEvaluator(
                sampler=self,
                metric_functions=metric_functions,
                )

        iter_func_names, iter_func_kwargs = \
                self.get_iter_step(iter_type, tqdm=tqdm, **kwargs)
        if tqdm_iter:
            iter_func_kwargs[0]['tqdm'] = tqdm

        num_iters = max_time//min_save_time
        if max_num_iters is not None:
            num_iters = min(num_iters, max_num_iters)


        iteration = 0
        total_time = 0
        parameters_list = [None]*(num_iters+1)
        times = np.zeros(num_iters+1)*np.nan
        iterations = np.zeros(num_iters+1, dtype=int)

        fit_start_time = time.time()
        last_save_time = time.time()

        parameters_list[0] = self.parameters.copy()
        times[0] = total_time
        iterations[0] = iteration
        evaluator.eval_metric_functions(iteration=iteration)

        # Fit Loop
        pbar = range(1, num_iters+1)
        if tqdm is not None:
            pbar = tqdm(pbar)
        for it in pbar:
            if tqdm is not None:
                pbar.set_description("fit using {0}".format(iter_type) + \
                        " on iter {0}".format(iteration)
                        )
            try:
                for step in range(1000):
                    # Run iter funcs
                    for func_name, func_kwargs in zip(iter_func_names,
                            iter_func_kwargs):
                        getattr(self, func_name)(**func_kwargs)
                    if time.time() - last_save_time > min_save_time:
                        parameters_list[it] = self.parameters.copy()

                        total_time += time.time() - last_save_time
                        times[it] = total_time

                        iteration += step + 1
                        iterations[it] = iteration

                        evaluator.eval_metric_functions(iteration=iteration)
                        last_save_time = time.time()
                        break
            except KeyboardInterrupt as e:
                if catch_interrupt:
                    logger.warning("Interrupt in fit_timed:\n{0}\n".format(e) + \
                            "Stopping early after {0} iters".format(it))
                    break
                else:
                    raise e

            if total_time > max_time:
                # Break it total time on iter_funcs exceeds max time
                break
            if total_max_time is not None:
                if fit_start_time - time.time() > total_max_time:
                    # Break it total time on fit_evalute exceeds total max time
                    break

        valid = np.sum(~np.isnan(times))
        parameters_list = pd.DataFrame(dict(
            iteration = iterations[0:valid],
            parameters = parameters_list[0:valid],
            ))
        times = pd.DataFrame(dict(
            iteration = iterations[0:valid],
            time = times[0:valid],
            ))
        metric = evaluator.get_metrics()
        return parameters_list, times, metric

    def get_iter_step(self, iter_type, steps_per_iteration=1, **kwargs):
        # Returns iter_func_names, iter_func_kwargs
        project_kwargs = kwargs.get("project_kwargs",{})
        if iter_type == 'Gibbs':
            iter_func_names = ["sample_gibbs", "project_parameters"]
            iter_func_kwargs = [{}, project_kwargs]
        elif iter_type == 'custom':
            iter_func_names = kwargs.get("iter_func_names")
            iter_func_kwargs = kwargs.get("iter_func_kwargs")
        elif iter_type in ['SGD', 'ADAGRAD', 'SGLD', 'SGRD', 'SGRLD']:
            grad_kwargs = dict(
                epsilon = kwargs['epsilon'],
                subsequence_length = kwargs['subsequence_length'],
                buffer_length = kwargs['buffer_length'],
                minibatch_size = kwargs.get('minibatch_size', 1),
                kind = kwargs.get("kind", "marginal"),
                num_samples = kwargs.get("num_samples", None),
                **kwargs.get("pf_kwargs", {})
            )
            if 'num_sequences' in kwargs:
                grad_kwargs['num_sequences'] = kwargs['num_sequences']
            if 'use_scir' in kwargs:
                grad_kwargs['use_scir'] = kwargs['use_scir']

            if iter_type == 'SGD':
                iter_func_names = ['step_sgd', 'project_parameters']
                iter_func_kwargs = [grad_kwargs, project_kwargs]
            elif iter_type == 'ADAGRAD':
                iter_func_names = ['step_adagrad', 'project_parameters']
                iter_func_kwargs = [grad_kwargs, project_kwargs]
            elif iter_type == 'SGLD':
                iter_func_names = ['sample_sgld', 'project_parameters']
                iter_func_kwargs = [grad_kwargs, project_kwargs]
            elif iter_type == 'SGRD':
                grad_kwargs['preconditioner'] = self._get_preconditioner(
                        kwargs.get('preconditioner')
                        )
                iter_func_names = ['step_precondition_sgd', 'project_parameters']
                iter_func_kwargs = [grad_kwargs, project_kwargs]
            elif iter_type == 'SGRLD':
                grad_kwargs['preconditioner'] = self._get_preconditioner(
                        kwargs.get('preconditioner')
                        )
                iter_func_names = ['sample_sgrld', 'project_parameters']
                iter_func_kwargs = [grad_kwargs, project_kwargs]
        else:
            raise ValueError("Unrecognized iter_type {0}".format(iter_type))

        iter_func_names = iter_func_names * steps_per_iteration
        iter_func_kwargs = iter_func_kwargs * steps_per_iteration

        return iter_func_names, iter_func_kwargs

    def _get_preconditioner(self, preconditioner=None):
        if preconditioner is None:
            raise NotImplementedError("No Default Preconditioner for {}".format(
                self.name))
        return preconditioner

    ## Predict Functions
    def predict(self, target='latent', distr=None, lag=None,
            return_distr=None, num_samples=None,
            kind='analytic', observations=None, parameters=None,
            **kwargs):
        """ Make predictions based on fit

        Args:
            target (string): variable to predict
                'latent' - latent variables
                'y' - observation variables
            distr (string): what distribution to sample/target
                'marginal' - marginal (default for return_distr)
                'joint' - joint (default for sampling)
            lag (int): distribution is p(U_t | Y_{1:t+lag})
                default/None -> use all observations
            return_distr (bool): return distribution
                (default is True if num_samples is None otherwise True)
            num_samples (int): number of samples return
            kind (string): how to calculate distribution
                'analytic' - use message passing
                'pf' - use particle filter/smoother
            observations (ndarray): observations to use
            parameters (Parameters): parameters

            kwargs: key word arguments
                tqdm (tqdm): progress bar
                see message_helper.latent_var_distr,
                    message_helper.y_distr,
                    message_helper.latent_var_sample,
                    message_helper.y_sample,
                    message_helper.pf_latent_var_distr,
                    message_helper.pf_y_distr,
                for more details

        Returns:
            Depends on target, return_distr, num_samples
        """
        observations = self._get_observations(observations)
        if parameters is None:
            parameters = self.parameters
        if return_distr is None:
            if kind == 'pf':
                return_distr = True
            else:
                return_distr = (num_samples is None)

        if kind == 'analytic':
            if return_distr:
                if distr is None:
                    distr = 'marginal'
                if target == 'latent':
                    return self.message_helper.latent_var_distr(
                            distr=distr,
                            lag=lag,
                            observations=observations,
                            parameters=parameters,
                            **kwargs,
                            )
                elif target == 'y':
                    return self.message_helper.y_distr(
                            distr=distr,
                            lag=lag,
                            observations=observations,
                            parameters=parameters,
                            **kwargs,
                            )
                else:
                    raise ValueError("Unrecognized target '{0}'".format(target))
            else:
                if distr is None:
                    distr = 'joint'
                if target == 'latent':
                    return self.message_helper.latent_var_sample(
                            distr=distr,
                            lag=lag,
                            num_samples=num_samples,
                            observations=observations,
                            parameters=parameters,
                            **kwargs,
                            )
                elif target == 'y':
                    return self.message_helper.y_sample(
                            distr=distr,
                            lag=lag,
                            num_samples=num_samples,
                            observations=observations,
                            parameters=parameters,
                            **kwargs,
                            )
                else:
                    raise ValueError("Unrecognized target '{0}'".format(target))
        elif kind == 'pf':
            if return_distr:
                if target == 'latent':
                    return self.message_helper.pf_latent_var_distr(
                            lag=lag,
                            observations=observations,
                            parameters=parameters,
                            **kwargs,
                            )
                elif target == 'y':
                    return self.message_helper.pf_y_distr(
                            distr=distr,
                            lag=lag,
                            observations=observations,
                            parameters=parameters,
                            **kwargs,
                            )
                else:
                    raise ValueError("Unrecognized target '{0}'".format(target))
            else:
                raise ValueError("return_distr must be True for kind = pf")
        else:
            raise ValueError("Unrecognized kind == '{0}'".format(kind))

    def simulate(self, T, init_message=None,
            return_distr=False, num_samples=None,
            kind='analytic', observations=None, parameters=None,
            **kwargs):
        """ Simulate dynamics

        Args:
            T (int): length of simulated data
            init_message (dict): initial forward message
            return_distr (bool): return distribution (default is False)
            num_samples (int): number of samples return
            kind (string): how to calculate distribution
                'analytic' - use message passing
                'pf' - use particle filter/smoother
            observations (ndarray): observations
            parameters (Parameters): parameters

        Returns:
            dict with key values depending on return_distr and num_samples
                latent_vars (ndarray): simulated latent vars
                observations (ndarray): simulated observations
                latent_mean/latent_prob/latent_cov
                observation_mean/observation_prob/observations_cov
        """
        observations = self._get_observations(observations)
        if parameters is None:
            parameters = self.parameters

        if kind == 'analytic':
            if init_message is None:
                init_message = self.message_helper.forward_message(
                        observations=observations,
                        parameters=parameters,
                        )
            if return_distr:
                return self.message_helper.simulate_distr(
                        T=T,
                        parameters=parameters,
                        init_message=init_message,
                        **kwargs
                        )
            else:
                return self.message_helper.simulate(
                        T=T,
                        parameters=parameters,
                        init_message=init_message,
                        num_samples=num_samples,
                        **kwargs
                        )
        elif kind == 'pf':
            raise NotImplementedError()
        else:
            raise ValueError("Unrecognized kind == '{0}'".format(kind))

    ## Attributes + Misc Helper Functions
    @property
    def observations(self):
        return self._observations

    @observations.setter
    def observations(self, observations):
        self._check_observation_shape(observations)
        self._observations = observations
        return

    def _check_observation_shape(self, observations):
        return

    def _get_observations(self, observations, check_shape=True):
        if observations is None:
            observations = self.observations
            if observations is None:
                raise ValueError("observations not specified")
        elif check_shape:
            self._check_observation_shape(observations)
        return observations

    def _get_T(self, **kwargs):
        T = kwargs.get('T')
        if T is None:
            observations = kwargs.get('observations')
            observations = self._get_observations(observations)
            T = observations.shape[0]
        return T

# SeqSGMCMCSampler
class SeqSGMCMCSampler(object):
    """ Mixin for handling a list of sequences """
    def _get_T(self, **kwargs):
        T = kwargs.get('T')
        if T is None:
            observations = kwargs.get('observations')
            observations = self._get_observations(observations)
            T = np.sum(np.shape(observation)[0] for observation in observations)
        return T

    def _check_observation_shape(self, observations):
        if observations is not None:
            for ii, observation in enumerate(observations):
                try:
                    super()._check_observation_shape(observations=observation)
                except ValueError as e:
                    raise ValueError("Error in observations[{0}] :\n{1}".format(
                        ii, e))

    def exact_loglikelihood(self, observations=None, tqdm=None):
        """ Return exact loglikelihood over all observation sequences """
        observations = self._get_observations(observations)
        loglikelihood = 0
        pbar = observations
        if tqdm is not None:
            pbar = tqdm(pbar)
            pbar.set_description("Seq Loglikelihood")
        for observation in pbar:
            loglikelihood += self.message_helper.marginal_loglikelihood(
                    observations=observation,
                    parameters=self.parameters,
                    forward_message=self.forward_message,
                    backward_message=self.backward_message,
                    tqdm=tqdm,
                    )
        return loglikelihood

    def noisy_loglikelihood(self, num_sequences=-1, observations=None,
            tqdm=None, **kwargs):
        """ Subsequence Approximation to loglikelihood

        Args:
            num_sequences (int): how many observation sequences to use
                (default = -1) is to use all observation sequences
        """
        observations = self._get_observations(observations)
        loglikelihood = 0
        S = 0.0
        sequence_indices = np.arange(len(observations))
        if num_sequences != -1:
            sequence_indices = np.random.choice(
                    sequence_indices, num_sequences, replace=False,
                    )
        pbar = sequence_indices
        if tqdm is not None:
            pbar = tqdm(pbar)
            pbar.set_description("Seq Loglikelihood")
        for sequence_index in pbar:
            S += observations[sequence_index].shape[0]
            loglikelihood += super().noisy_loglikelihood(
                    observations=observations[sequence_index],
                    tqdm=tqdm,
                    **kwargs)
        if num_sequences != -1:
            loglikelihood *= self._get_T(**kwargs) / S
        return loglikelihood

    def predictive_loglikelihood(self, num_sequences=-1, observations=None,
            tqdm=None, **kwargs):
        """ Return the predictive loglikelihood given the parameters """
        observations = self._get_observations(observations)
        predictive_loglikelihood = 0
        S = 0.0
        sequence_indices = np.arange(len(observations))
        if num_sequences != -1:
            sequence_indices = np.random.choice(
                    sequence_indices, num_sequences, replace=False,
                    )
        pbar = sequence_indices
        if tqdm is not None:
            pbar = tqdm(pbar)
            pbar.set_description("Seq Pred Loglikelihood")
        for sequence_index in pbar:
            S += observations[sequence_index].shape[0]
            predictive_loglikelihood += super().predictive_loglikelihood(
                    observations=observations[sequence_index],
                    tqdm=tqdm,
                    **kwargs)
        if num_sequences != -1:
            predictive_loglikelihood *= self._get_T(**kwargs) / S
        return predictive_loglikelihood

    def _noisy_grad_loglikelihood(self, num_sequences=-1, **kwargs):
        """ Subsequence approximation to gradient of loglikelihood

        Args:
            num_sequences (int): how many observation sequences to use
                (default = -1) is to use all observation sequences
        """
        noisy_grad_loglike = None
        S = 0.0

        sequence_indices = np.arange(len(self.observations))
        if num_sequences != -1:
            sequence_indices = np.random.choice(
                    sequence_indices, num_sequences, replace=False,
                    )
        for sequence_index in sequence_indices:
            noisy_grad_index = super()._noisy_grad_loglikelihood(
                    observations=self.observations[sequence_index],
                    **kwargs)
            S += self.observations[sequence_index].shape[0]
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
                    var: noisy_grad_loglike[var] * self._get_T(**kwargs) / S
                    for var in noisy_grad_index.keys()
                    }
        return noisy_grad_loglike

    def predict(self, target='latent', distr=None, lag=None,
            return_distr=None, num_samples=None,
            kind='analytic', observations=None, parameters=None,
            tqdm=None,
            **kwargs):
        """ Make predictions based on fit

        Args:
            target (string): variable to predict
                'latent' - latent variables
                'y' - observation variables
            distr (string): what distribution to sample/target
                'marginal' - marginal (default for return_distr)
                'joint' - joint (default for sampling)
            lag (int): distribution is p(U_t | Y_{1:t+lag})
                default/None -> use all observations
            return_distr (bool): return distribution
                (default is True if num_samples is None otherwise True)
            num_samples (int): number of samples return
            kind (string): how to calculate distribution
                'analytic' - use message passing
                'pf' - use particle filter/smoother
            observations (list of ndarray): observations to use
            parameters (Parameters): parameters

            kwargs: key word arguments
                tqdm (tqdm): progress bar
                see message_helper.latent_var_distr,
                    message_helper.y_distr,
                    message_helper.latent_var_sample,
                    message_helper.y_sample,
                    message_helper.pf_latent_var_distr,
                    message_helper.pf_y_distr,
                for more details

        Returns:
            Depends on target, return_distr, num_samples
        """
        observations = self._get_observations(observations)
        if parameters is None:
            parameters = self.parameters
        if return_distr is None:
            if kind == 'pf':
                return_distr = True
            else:
                return_distr = (num_samples is None)

        output = []
        if tqdm is not None:
            kwargs['tqdm'] = tqdm
            observations = tqdm(observations, desc='sequence #')
        if kind == 'analytic':
            if return_distr:
                if distr is None:
                    distr = 'marginal'
                if target == 'latent':
                    for observation in observations:
                        output.append(
                            self.message_helper.latent_var_distr(
                                distr=distr,
                                lag=lag,
                                observations=observation,
                                parameters=parameters,
                                **kwargs,
                                )
                            )
                elif target == 'y':
                    for observation in observations:
                        output.append(
                            self.message_helper.y_distr(
                                distr=distr,
                                lag=lag,
                                observations=observation,
                                parameters=parameters,
                                **kwargs,
                                )
                            )
                else:
                    raise ValueError("Unrecognized target '{0}'".format(target))
            else:
                if distr is None:
                    distr = 'joint'
                if target == 'latent':
                    for observation in observations:
                        output.append(
                            self.message_helper.latent_var_sample(
                                distr=distr,
                                lag=lag,
                                num_samples=num_samples,
                                observations=observation,
                                parameters=parameters,
                                **kwargs,
                                )
                            )
                elif target == 'y':
                    for observation in observations:
                        output.append(
                            self.message_helper.y_sample(
                                distr=distr,
                                lag=lag,
                                num_samples=num_samples,
                                observations=observation,
                                parameters=parameters,
                                **kwargs,
                                )
                            )
                else:
                    raise ValueError("Unrecognized target '{0}'".format(target))
        elif kind == 'pf':
            if return_distr:
                if target == 'latent':
                    for observation in observations:
                        output.append(
                            self.message_helper.pf_latent_var_distr(
                                lag=lag,
                                observations=observation,
                                parameters=parameters,
                                **kwargs,
                                )
                            )
                elif target == 'y':
                    for observation in observations:
                        output.append(
                            self.message_helper.pf_y_distr(
                                distr=distr,
                                lag=lag,
                                observations=observation,
                                parameters=parameters,
                                **kwargs,
                                )
                            )
                else:
                    raise ValueError("Unrecognized target '{0}'".format(target))
            else:
                raise ValueError("return_distr must be True for kind = pf")
        else:
            raise ValueError("Unrecognized kind == '{0}'".format(kind))

        return output


# SGMCMC Helper
class SGMCMCHelper(object):
    """ Base Class for SGMCMC Helper """
    def __init__(self, **kwargs):
        raise NotImplementedError()

    ## Message Passing Functions
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
        forward_message = self._forward_messages(
                observations=observations,
                parameters=parameters,
                forward_message=forward_message,
                only_return_last=True,
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
        backward_message = self._backward_messages(
                observations=observations,
                parameters=parameters,
                backward_message=backward_message,
                only_return_last=True,
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

    def _forward_messages(self, observations, parameters, forward_message,
            weights=None, only_return_last=False, **kwargs):
        raise NotImplementedError()

    def _backward_messages(self, observations, parameters, backward_message,
            weights=None, only_return_last=False, **kwargs):
        raise NotImplementedError()

    def _forward_message(self, observations, parameters, forward_message,
            **kwargs):
        return self._forward_messages(observations, parameters, forward_message,
                only_return_last=True, **kwargs)

    def _backward_message(self, observations, parameters, backward_message,
            **kwargs):
        return self._backward_messages(observations, parameters,
                backward_message, only_return_last=True, **kwargs)

    ## Loglikelihood Functions
    def marginal_loglikelihood(self, observations, parameters,
            forward_message=None, backward_message=None, weights=None,
            tqdm=None):
        """ Calculate the marginal loglikelihood Pr(y | theta)

        Args:
            observations (ndarray): observations
            parameters (Parameters): parameters
            forward_message (dict): latent state forward message
            backward_message (dict): latent state backward message
            weights (ndarray): optional, weights for loglikelihood calculation

        Returns:
            marginal_loglikelihood (float): marginal loglikelihood
        """
        raise NotImplementedError()

    def predictive_loglikelihood(self, observations, parameters,
            forward_message=None, backward_message=None, lag=1):
        """ Calculate the predictive loglikelihood
            pred_loglikelihood = sum_t Pr(y_{t+lag} | y_{<t} theta)

        Args:
            observations (ndarray): observations
            parameters (Parameters): parameters
            forward_message (dict): latent state forward message
            backward_message (dict): latent state backward message
            lag (int): how many steps ahead to predict

        Returns:
            pred_loglikelihood (float): predictive loglikelihood
        """
        raise NotImplementedError()

    def complete_data_loglikelihood(self, observations, latent_vars, parameters,
            forward_message=None, weights=None, **kwargs):
        """ Calculate the complete data loglikelihood Pr(y, u | theta)

        Args:
            observations (ndarray): observations
            latent_vars (ndarray): latent vars
            parameters (Parameters): parameters
            forward_message (dict): latent state forward message
            weights (ndarray): optional, weights for loglikelihood calculation

        Returns:
            complete_data_loglikelihood (float): complete data loglikelihood
        """
        raise NotImplementedError()

    ## Gradient Functions
    def gradient_marginal_loglikelihood(self, observations, parameters,
            forward_message=None, backward_message=None, weights=None, **kwargs):
        """ Gradient Calculation

        Gradient of log Pr(y_[0:T) | y_<0, y_>=T, parameters)

        Args:
            observations (ndarray): num_obs observations
            parameters (Parameters): parameters
            forward_message (dict): Pr(u_-1, y_<0 | parameters)
            backward_message (dict): Pr(y_>T | u_T, parameters)
            weights (ndarray): how to weight terms

        Returns
            grad (dict): grad of variables in parameters

        """
        raise NotImplementedError()

    def gradient_complete_data_loglikelihood(self, observations, latent_vars,
            parameters, forward_message=None, weights=None, **kwargs):
        """ Gradient Calculation

        Gradient of log Pr(y_[0:T), u_[0:T) | y_<0, parameters)

        Args:
            observations (ndarray): num_obs observations
            latent_vars (ndarray): num_obs latent vars
            parameters (Parameters): parameters
            forward_message (dict): Pr(u_-1, y_<0 | parameters)
            weights (ndarray): how to weight terms

        Returns
            grad (dict): grad of variables in parameters

        """
        raise NotImplementedError()

    ## Gibbs Functions
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
        """ Gibbs Sample Sufficient Statistics
        Args:
            observations (ndarray): num_obs observations
            latent_vars (ndarray): latent vars

        Returns:
            sufficient_stat (dict of dict)
                keys are parameter
                values are dict for parameter's sufficient statistics
        """
        raise NotImplementedError()

    ## Predict Functions
    def latent_var_distr(self, observations, parameters,
            distr='marginal', lag=None,
            forward_message=None, backward_message=None,
            tqdm=None, **kwargs):
        """ Sample latent vars distribution conditional on observations

        Returns distribution for (u_t | y_{<= t+lag}, theta)

        Args:
            observations (ndarray): observations
            parameters (LGSSMParameters): parameters
            lag (int): what observations to condition on, None = all
            forward_message (dict): forward message
            backward_message (dict): backward message

        Returns:
            Depends on latent var type, Gaussian -> mean, cov; Discrete -> prob

        """
        raise NotImplementedError()

    def latent_var_sample(self, observations, parameters,
            distr='joint', lag=None, num_samples=None,
            forward_message=None, backward_message=None,
            include_init=False, tqdm=None, **kwargs):
        """ Sample latent vars conditional on observations

        Samples u_t ~ u_t | y_{<= t+lag}, theta

        Args:
            observations (ndarray): observations
            parameters (LGSSMParameters): parameters
            lag (int): what observations to condition on, None = all
            num_samples (int, optional) number of samples
            forward_message (dict): forward message
            backward_message (dict): backward message
            include_init (bool, optional): whether to sample u_{-1} | y

        Returns:
            sampled_latent_vars : shape depends on num_samples parameters
                last dimension is num_samples

        """
        raise NotImplementedError()

    def y_distr(self, observations, parameters,
            distr='marginal', lag=None,
            forward_message=None, backward_message=None,
            latent_var=None, tqdm=None, **kwargs):
        """ Sample observation distribution conditional on observations

        Returns distribution for (y_t* | y_{<= t+lag}, theta)

        Args:
            observations (ndarray): observations
            parameters (LGSSMParameters): parameters
            lag (int): what observations to condition on, None = all
            forward_message (dict): forward message
            backward_message (dict): backward message
            latent_var (ndarray): latent vars
                if provided, will return (y_t* | u_t, theta) instead

        Returns:
            Depends on observation type, Gaussian -> mean, cov; Discrete -> prob

        """
        raise NotImplementedError()

    def y_sample(self, observations, parameters,
            distr='joint', lag=None, num_samples=None,
            forward_message=None, backward_message=None,
            latent_var=None, tqdm=None, **kwargs):
        """ Sample new observations conditional on observations

        Samples y_t* ~ y_t* | y_{<= t+lag}, theta

        Args:
            observations (ndarray): observations
            parameters (LGSSMParameters): parameters
            lag (int): what observations to condition on, None = all
            num_samples (int, optional) number of samples
            forward_message (dict): forward message
            backward_message (dict): backward message
            latent_var (ndarray): latent vars
                if provided, will sample from (y_t* | u_t, theta) instead
                must match num_samples parameters

        Returns:
            sampled_observations : shape depends on num_samples parameters
                last dimension is num_samples

        """
        raise NotImplementedError()

    def simulate_distr(self, T, parameters, init_message=None, tqdm=None):
        raise NotImplementedError()

    def simulate(self, T, parameters, init_message=None, num_samples=None, tqdm=None):
        raise NotImplementedError()

    ## PF Functions
    def pf_loglikelihood_estimate(self, observations, parameters,
            subsequence_start=0, subsequence_end=None, weights=None,
            pf="poyiadjis_N", N=1000, kernel='prior', forward_message=None,
            **kwargs):
        """ Particle Filter Marginal Log-Likelihood Estimate

        Args:
            observations (ndarray): num_obs bufferd observations
            parameters (Parameters): parameters
            weights (ndarray): weights (to correct storchastic approx)
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
                "prior" - bootstrap filter P(u_t | u_{t-1})
                "optimal" - bootstrap filter P(u_t | u_{t-1}, Y_t)
            forward_message (dict): prior for buffered subsequence
            **kwargs - additional keyword args for individual filters

        Return:
            loglikelihood (double): marignal log likelihood estimate

        """
        raise NotImplementedError()

    def pf_predictive_loglikelihood_estimate(self, observations, parameters,
            num_steps_ahead=1,
            subsequence_start=0, subsequence_end=None, weights=None,
            pf="filter", N=1000, kernel=None, forward_message=None,
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
            forward_message (dict): prior for buffered subsequence
            **kwargs - additional keyword args for individual filters

        Return:
            predictive_loglikelihood (num_steps_ahead + 1 ndarray)

        """
        raise NotImplementedError()

    def pf_gradient_estimate(self, observations, parameters,
            subsequence_start=0, subsequence_end=None, weights=None,
            pf="poyiadjis_N", N=1000, kernel=None, forward_message=None,
            **kwargs):
        """ Particle Smoother Gradient Estimate

        Args:
            observations (ndarray): num_obs bufferd observations
            parameters (Parameters): parameters
            subsequence_start (int): relative start of subsequence
                (0:subsequence_start are left buffer)
            subsequence_end (int): relative end of subsequence
                (subsequence_end: is right buffer)
            weights (ndarray): weights (to correct storchastic approx)
            pf (string): particle filter name
                "nemeth" - use Nemeth et al. O(N)
                "poyiadjis_N" - use Poyiadjis et al. O(N)
                "poyiadjis_N2" - use Poyiadjis et al. O(N^2)
                "paris" - use PaRIS Olsson + Westborn O(N log N)
            N (int): number of particles used by particle filter
            kernel (string): kernel to use
                "prior" - bootstrap filter P(u_t | u_{t-1})
                "optimal" - bootstrap filter P(u_t | u_{t-1}, Y_t)
            forward_message (dict): prior for buffered subsequence
            **kwargs - additional keyword args for individual filters

        Return:
            grad (dict): grad of variables in parameters

        """
        raise NotImplementedError()

    def pf_latent_var_distr(self, observations, parameters, lag=None,
            subsequence_start=0, subsequence_end=None,
            pf="poyiadjis_N", N=1000, kernel=None, forward_message=None,
            **kwargs):
        """ Sample latent vars distribution conditional on observations

        Returns distribution for (u_t | y_{<= t+lag}, theta)
        Estimated using particle filter/smoother

        Args:
            observations (ndarray): observations
            parameters (LGSSMParameters): parameters
            lag (int): what observations to condition on, None = all
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
                "prior" - bootstrap filter P(u_t | u_{t-1})
                "optimal" - bootstrap filter P(u_t | u_{t-1}, Y_t)
            forward_message (dict): prior for buffered subsequence
            **kwargs - additional keyword args for individual filters

        Returns:
            Depends on latent var type, Gaussian -> mean, cov; Discrete -> prob

        """
        raise NotImplementedError()

    def pf_y_distr(self, observations, parameters,
            distr='marginal', lag=None,
            subsequence_start=0, subsequence_end=None,
            pf="poyiadjis_N", N=1000, kernel=None, forward_message=None,
            **kwargs):
        """ Sample observation distribution conditional on observations

        Returns distribution for (u_t | y_{<= t+lag}, theta)
        Estimated using particle filter/smoother

        Args:
            observations (ndarray): observations
            parameters (LGSSMParameters): parameters
            lag (int): what observations to condition on, None = all
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
                "prior" - bootstrap filter P(u_t | u_{t-1})
                "optimal" - bootstrap filter P(u_t | u_{t-1}, Y_t)
            forward_message (dict): prior for buffered subsequence
            **kwargs - additional keyword args for individual filters

        Returns:
            Depends on latent var type, Gaussian -> mean, cov; Discrete -> prob

        """
        raise NotImplementedError()



# Helper Function for Sampling Subsequences
def random_subsequence_and_weights(S, T, partition_style=None):
    """ Get Subsequence + Weights
    Args:
        S (int): length of subsequence
        T (int): length of full sequence
        partition_style (string): what type of partition
            'strict' - strict partition, with weights
            'uniform' - uniformly, with weights
            'naive' - uniformly, with incorrect weights (not recommended)

    Returns:
        subsequence_start (int): start of subsequence (inclusive)
        subsequence_end (int): end of subsequence (exclusive)
        weights (ndarray): weights for [start,end)
    """
    if partition_style is None:
        partition_style = 'uniform'

    if partition_style == 'strict':
        if T % S != 0:
            raise ValueError("S {0} does not evenly divide T {1}".format(S, T)
        )
        subsequence_start = np.random.choice(np.arange(0, T//S)) * S
        subsequence_end = subsequence_start + S
        weights = np.ones(S, dtype=float)*T/S
    elif partition_style == 'uniform':
        subsequence_start = np.random.randint(0, T-S+1)
        subsequence_end = subsequence_start + S
        t = np.arange(subsequence_start, subsequence_end)
        if subsequence_end <= 2*S:
            num_sequences = np.min(np.array([
                t+1, np.ones_like(t)*min(S, T-S+1)
                ]), axis=0)
        elif subsequence_start >= T-2*S-1:
            num_sequences = np.min(np.array([
                T-t, np.ones_like(t)*min(S, T-S+1)
                ]), axis=0)
        else:
            num_sequences = np.ones(S)*S
        weights = np.ones(S, dtype=float)*(T-S+1)/num_sequences
    elif partition_style == 'naive':
        # Not recommended because the weights are incorrect
        subsequence_start = np.random.randint(0, T-S+1)
        subsequence_end = subsequence_start + S
        weights = np.ones(S, dtype=float)*T/S
    else:
        raise ValueError("Unrecognized partition_style = '{0}'".format(
            partition_style))
    return int(subsequence_start), int(subsequence_end), weights



