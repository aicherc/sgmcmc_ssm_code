import numpy as np
import logging
logger = logging.getLogger(name=__name__)

from ..sgmcmc_sampler import SGMCMCHelper
from .._utils import random_categorical

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
            pbar = tqdm(pbar, leave=False)
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
            pbar = tqdm(pbar, total=num_obs, leave=False)
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

    def predictive_loglikelihood(self, observations, parameters, lag=1,
            forward_message=None, backward_message=None, tqdm=None, **kwargs):
        if forward_message is None:
            forward_message = self.default_forward_message
        if backward_message is None:
            backward_message = self.default_backward_message

        # Calculate Filtered
        if lag == 0:
            forward_messages = self.forward_pass(observations,
                    parameters, forward_message, tqdm=tqdm, **kwargs)

        else:
            forward_messages = self.forward_pass(observations[0:-lag],
                    parameters, forward_message, tqdm=tqdm, **kwargs)
        loglike = 0.0
        Pi = parameters.pi
        pbar = range(lag, np.shape(observations)[0])
        if tqdm is not None:
            pbar = tqdm(pbar, leave=False)
            pbar.set_description('predictive_loglikelihood')
        for t in pbar:
            # Calculate Pr(z_t | y_{<=t-lag}, theta)
            prob_vector = forward_messages[t-lag]['prob_vector']
            for l in range(lag):
                prob_vector = np.dot(prob_vector, Pi)

            P_t, log_constant = self._likelihoods(observations[t], parameters)
            likelihood = np.dot(prob_vector, P_t)
            loglike += np.log(likelihood) + log_constant
        return loglike

    def complete_data_loglikelihood(self, observations, latent_vars, parameters, forward_message=None, weights=None, **kwargs):
        if forward_message is None:
            forward_message = self.default_forward_message

        log_constant = 0.0
        Pi = parameters.pi

        z_prev = forward_message.get('z_prev')
        for t, (y_t, z_t) in enumerate(zip(observations, latent_vars)):
            weight_t = 1.0 if weights is None else weights[t]

            # Pr(Z_t | Z_t-1)
            if (z_prev is not None):
                log_c = np.log(Pi[z_prev, z_t])
                log_constant += log_c * weight_t

            # Pr(Y_t | Z_t)
            log_c = self._emission_loglikelihood(y_t, z_t, parameters)
            log_constant += log_c * weight_t

            z_prev = z_t

        return log_constant

    def latent_var_distr(self, observations, parameters,
            distr='marginal', lag=None,
            forward_message=None, backward_message=None,
            tqdm=None):
        if distr != 'marginal':
            raise NotImplementedError()
        if forward_message is None:
            forward_message = self.default_forward_message
        if backward_message is None:
            backward_message = self.default_backward_message

        L = np.shape(observations)[0]
        Pi = parameters.pi
        z_prob = np.zeros((L, parameters.num_states), dtype=float)

        # Forward Pass
        forward_messages = self.forward_pass(
                observations=observations,
                parameters=parameters,
                forward_message=forward_message,
                tqdm=tqdm
                )

        pbar = range(L)
        if tqdm is not None:
            pbar = tqdm(pbar, leave=False)
            pbar.set_description('calc latent var distr')

        if lag is None:
            # Smoothing
            backward_messages = self.backward_pass(
                observations=observations,
                parameters=parameters,
                backward_message=backward_message,
                tqdm=tqdm
                )
            for t in pbar:
                log_prob_t = (np.log(forward_messages[t]['prob_vector']) + \
                              np.log(backward_messages[t]['likelihood_vector']))
                log_prob_t -= np.max(log_prob_t)
                z_prob[t] = np.exp(log_prob_t)/np.sum(np.exp(log_prob_t))
            return z_prob

        elif lag <= 0:
            # Prediction/Filtering
            for t in pbar:
                if t+lag >= 0:
                    prob_vector = forward_messages[t+lag]['prob_vector']
                else:
                    prob_vector = forward_message['prob_vector']
                # Forward Simulate
                for _ in range(-lag):
                    prob_vector = np.dot(prob_vector, Pi)
                log_prob_t = np.log(prob_vector)
                log_prob_t -= np.max(log_prob_t)
                z_prob[t] = np.exp(log_prob_t)/np.sum(np.exp(log_prob_t))
            return z_prob

        else:
            # Fixed-lag Smoothing
            for t in pbar:
                # Backward Messages
                back_obs = observations[t:min(t+lag, L)]
                fixed_lag_message = self.backward_message(
                        observations=back_obs,
                        parameters=parameters,
                        backward_message=backward_message,
                        )
                # Output
                log_prob_t = (np.log(forward_messages[t]['prob_vector']) + \
                              np.log(fixed_lag_message[t]['likelihood_vector']))
                log_prob_t -= np.max(log_prob_t)
                z_prob[t] = np.exp(log_prob_t)/np.sum(np.exp(log_prob_t))
            return z_prob

    def latent_var_sample(self, observations, parameters,
            forward_message=None, backward_message=None,
            distr='joint', lag=None, num_samples=None,
            tqdm=None, include_init=False, **kwargs):
        if forward_message is None:
            forward_message = self.default_forward_message
        if backward_message is None:
            backward_message = self.default_backward_message
        if distr == 'joint' and lag is not None:
            raise ValueError("Must set distr to 'marginal' for lag != None")

        Pi = parameters.pi
        L = observations.shape[0]
        if num_samples is not None:
            z = np.zeros((L, num_samples), dtype=int)
        else:
            z = np.zeros((L), dtype=int)

        if distr == 'joint':
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
                pbar = tqdm(pbar, total=len(backward_messages), leave=False)
                pbar.set_description("forward smoothed sampling z")

            for t, backward_t in pbar:
                y_cur = observations[t]
                if t == 0:
                    post_t = np.dot(forward_message['prob_vector'], Pi)
                    if num_samples is not None:
                        post_t = np.outer(np.ones(num_samples), post_t)
                else:
                    post_t = Pi[z[t-1]]
                P_t, _ = self._likelihoods(
                        y_cur=y_cur,
                        parameters=parameters,
                    )
                post_t = post_t * (P_t * backward_t['likelihood_vector'])
                if num_samples is not None:
                    post_t = post_t/np.sum(post_t, axis=-1)[:,np.newaxis]
                    z[t] = np.array([random_categorical(post_t_s)
                        for post_t_s in post_t], dtype=int)
                else:
                    post_t = post_t/np.sum(post_t)
                    z[t] = random_categorical(post_t)
            return z

        elif distr == 'marginal':
            # Calculate Distribution
            z_prob = self.latent_var_distr(observations, parameters,
                    lag=lag, forward_message=forward_message,
                    backward_message=backward_message, tqdm=tqdm,
                    )
            # Sample from Distribution
            L = z_prob.shape[0]
            if num_samples is not None:
                x = np.zeros((L, num_samples), dtype=int)
            else:
                x = np.zeros((L), dtype=int)
            pbar = reversed(range(L))
            if tqdm is not None:
                pbar = tqdm(pbar, leave=False)
                pbar.set_description("sampling z")
            for t in pbar:
                z[t] = random_categorical(z_prob[t], size=num_samples)
            return z

        else:
            raise ValueError("Unrecognized `distr'; {0}".format(distr))
        return

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
        logP_t = self._emission_loglikelihoods(
                y_cur=y_cur, parameters=parameters,
                )
        log_constant = np.max(logP_t)
        logP_t = logP_t - log_constant
        P_t = np.exp(logP_t)
        return P_t, log_constant

    def _emission_loglikelihoods(self, y_cur, parameters):
        # Return loglikelihoods of observation y_cur for each z
        # Override for compute/memory efficiency
        return np.array([self._emission_loglikelihood(y_cur, kk, parameters)
            for kk in range(parameters.num_states)])

    def _emission_loglikelihood(self, y_cur, z_cur, parameters):
        # Return loglikelihoods of observation y_cur for z_cur
        raise NotImplementedError()

    def gradient_marginal_loglikelihood(self, observations, parameters,
            forward_message=None, backward_message=None,
            tqdm=None):
        raise NotImplementedError()

from ..sgmcmc_sampler import SGMCMCSampler
from ..variables.probweight import (
        BernoulliParamHelper,
        TransitionMatrixParamHelper,
        )

class CIRSamplerMixin(SGMCMCSampler):
    # Adjust Sample SGLD and SGRLD to allow for Baker sampling
    def _get_probweight_param_names(self):
        names = []
        for param_helper in self.parameters._param_helper_list:
            if isinstance(param_helper, TransitionMatrixParamHelper):
                names.append(param_helper._logit_name)
                names.append(param_helper._expanded_name)
        return names

    def _sample_cir(self, var, a, epsilon):
        """ Sample from Cox-Ingersoll-Ross Process
            theta_new = 0.5*(1-exp(-epsilon)) * W
            W \sim NoncentralChi2(2a, 2*theta*exp(-epsilon)/(1-exp(-epsilon))

        Args:
            var (string): variable name
            a (float): Dirichlet sufficient statistic
            epsilon (float): stepsize
        """
        if var.startswith("logit_"):
            is_logit = True
            theta = getattr(self.parameters, "expanded_{0}".format(
                var.strip("logit_")))
        elif var.startswith("expanded_"):
            is_logit = False
            theta = getattr(self.parameters, var)
        else:
            raise ValueError("var must be logit_ or expanded_")
        if np.any(a < 0.001):
            raise RuntimeError("Why is a < 0.001")

        W = np.random.noncentral_chisquare(
            df = 2*a,
            nonc = 2*theta*np.exp(-epsilon)/(1-np.exp(-epsilon)),
            )
        theta_new = 0.5*(1-np.exp(-epsilon))*W + 1e-99

        if is_logit:
            logit_new = np.log(np.abs(theta_new) + 1e-99)
            logit_new -= np.outer(
                    np.mean(logit_new, axis=1),
                    np.ones(logit_new.shape[1])
                    )
            return logit_new
        else:
            return theta_new

    def noisy_gradient(self, preconditioner=None, is_scaled=True,
            use_scir=False, **kwargs):
        """ Noisy Gradient Estimate

        noisy_gradient = -grad tilde{U}(theta)
                       = grad marginal loglike + grad logprior

        Monte Carlo Estimate of gradient (using buffering)

        Args:
            preconditioner (object): preconditioner for gradients
            is_scaled (boolean): scale gradient by 1/T
            use_scir (bool): whether to use Cox-Ingersoll-Ross sampling for
                probability simplex variables
            **kwargs: arguments for `self._noisy_grad_loglikelihood()`
                For example: minibatch_size, buffer_length, use_analytic

        Returns:
            noisy_gradient (dict): dict of gradient vectors

        """
        noisy_grad_loglike = \
                self._noisy_grad_loglikelihood(use_scir=use_scir, **kwargs)
        noisy_grad_prior = self.prior.grad_logprior(
                parameters=kwargs.get('parameters',self.parameters),
                use_scir=use_scir,
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
                    scale=scale,
                    use_scir=use_scir,
                    )

        return noisy_gradient

    def sample_sgld(self, epsilon, use_scir=True, **kwargs):
        """ One Step of Stochastic Gradient Langevin Dynamics

        Args:
            epsilon (double): step size
            use_scir (bool): whether to use Cox-Ingersoll-Ross sampling for
                probability simplex variables
            **kwargs (kwargs): to pass to self.noisy_gradient

        Returns:
            parameters (Parameters): sampled parameters after one step
        """
        # Use default sample_sgld if use_scir is False
        if not use_scir:
            return super().sample_sgld(epsilon, **kwargs)

        # Use Stochastic Cox-Ingersoll-Ross Algorithm of Baker et al. 2018
        if kwargs.get("is_scaled", True):
            scale = 1.0 / self._get_T(**kwargs)
        else:
            scale = 1.0
        if "preconditioner" in kwargs:
            raise ValueError("Use SGRLD instead")
        delta = self.noisy_gradient(use_scir=use_scir, **kwargs)
        white_noise = self._get_sgmcmc_noise(**kwargs)

        probweight_param_names = self._get_probweight_param_names()
        for var in self.parameters.var_dict:
            if var in probweight_param_names:
                self.parameters.var_dict[var] = self._sample_cir(
                        var=var, a=delta[var]/scale, epsilon=epsilon)
            else:
                self.parameters.var_dict[var] += \
                    epsilon * delta[var] + np.sqrt(2.0*epsilon) * white_noise[var]
        return self.parameters

    def sample_sgrld(self, epsilon, preconditioner, use_scir=True, **kwargs):
        """ One Step of Stochastic Gradient Riemannian Langevin Dynamics

        theta += epsilon * (D(theta) * grad_logjoint + correction_term) + \
                N(0, 2 epsilon D(theta))

        Args:
            epsilon (double): step size
            preconditioner (object): preconditioner
            use_scir (bool): whether to use Cox-Ingersoll-Ross sampling for
                probability simplex variables

        Returns:
            parameters (Parameters): sampled parameters after one step
        """
        # Use default sample_sgrld if use_scir is False
        if not use_scir:
            return super().sample_sgrld(epsilon, preconditioner, **kwargs)

        # Use Stochastic Cox-Ingersoll-Ross Algorithm of Baker et al. 2018
        if kwargs.get("is_scaled", True):
            scale = 1.0 / self._get_T(**kwargs)
        else:
            scale = 1.0

        delta = self.noisy_gradient(preconditioner=preconditioner,
                use_scir=use_scir, **kwargs)
        white_noise = self._get_sgmcmc_noise(
                preconditioner=preconditioner, **kwargs)
        correction = preconditioner.correction_term(
                self.parameters, scale=scale)

        probweight_param_names = self._get_probweight_param_names()
        for var in self.parameters.var_dict:
            if var in probweight_param_names:
                self.parameters.var_dict[var] = self._sample_cir(
                        var=var, a=delta[var]/scale, epsilon=epsilon)
            else:
                self.parameters.var_dict[var] += \
                    epsilon * (delta[var] + correction[var]) + \
                    np.sqrt(2.0*epsilon) * white_noise[var]
        return self.parameters


