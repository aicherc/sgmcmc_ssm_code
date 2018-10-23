import numpy as np
import scipy.stats
from scipy.misc import logsumexp
import logging
logger = logging.getLogger(name=__name__)

class PiMixin(object):
    # Mixin for Pi variable
    def _set_dim(self, **kwargs):
        if 'logit_pi' in kwargs:
            num_states, num_states2 = np.shape(kwargs['logit_pi'])
            self.dim['pi_type'] = 'logit'
        elif 'expanded_pi' in kwargs:
            num_states, num_states2 = np.shape(kwargs['expanded_pi'])
            self.dim['pi_type'] = 'expanded'
        else:
            raise ValueError("Either logit_pi or expanded_pi must be provided")

        if num_states != num_states2:
            raise ValueError("pi must be square, not {0}".format(
                (num_states, num_states2)))
        if "num_states" in self.dim:
            if num_states != self.dim['num_states']:
                raise ValueError("dims of pi do not match existing dims")
        else:
            self.dim['num_states'] = num_states
        super()._set_dim(**kwargs)
        return

    def _set_var_dict(self, **kwargs):
        if self.pi_type == 'logit':
            self.var_dict['logit_pi'] = np.array(kwargs['logit_pi']).astype(float)
        elif self.pi_type == 'expanded':
            self.var_dict['expanded_pi'] = np.array(kwargs['expanded_pi']).astype(float)
        else:
            raise RuntimeError()
        super()._set_var_dict(**kwargs)
        return

    @classmethod
    def _from_dict_to_vector(cls, vector_list, var_dict, **kwargs):
        if 'logit_pi' in var_dict:
            vector_list.append(var_dict['logit_pi'].flatten())
        elif 'expanded_pi' in var_dict:
            vector_list.append(var_dict['expanded_pi'].flatten())
        else:
            raise RuntimeError("Missing logit_pi + expanded_pi in var_dict")
        return super()._from_dict_to_vector(vector_list, var_dict, **kwargs)

    @classmethod
    def _from_vector_to_dict(cls, var_dict, vector, **kwargs):
        num_states = kwargs['num_states']
        pi_vec = np.reshape(vector[0:num_states**2], (num_states, num_states))
        pi_type = kwargs.get('pi_type', 'logit')
        if pi_type == 'logit':
            var_dict['logit_pi'] = pi_vec
        elif pi_type == 'expanded':
            var_dict['expanded_pi'] = pi_vec
        else:
            raise ValueError("Unrecognized pi_type {0}".format(pi_type))
        var_dict = super()._from_vector_to_dict(
                var_dict, vector[num_states**2:], **kwargs)
        return var_dict

    def _project_parameters(self, **kwargs):
        if self.pi_type == 'logit' and kwargs.get('center_pi', False):
            # Center logit_pi to be stable
            logit_pi = self.logit_pi
            logit_pi -= np.outer(np.mean(logit_pi, axis=1),
                    np.ones(self.num_states))
        if self.pi_type == "expanded":
            self.expanded_pi = np.abs(self.expanded_pi)
            if kwargs.get('center_pi', False):
                self.expanded_pi = np.abs(self.expanded_pi) / \
                        np.sum(np.abs(self.expanded_pi), axis=1)
        return super()._project_parameters(**kwargs)

    @property
    def pi_type(self):
        return self.dim['pi_type']
    @pi_type.setter
    def pi_type(self, pi_type):
        if pi_type == self.pi_type:
            return
        else:
            if pi_type == "logit":
                logit_pi = np.log(self.pi + 1e-9)
                logit_pi -= np.outer(np.mean(logit_pi, axis=1),
                        np.ones(self.num_states))
                self.var_dict['logit_pi'] = logit_pi
                self.var_dict.pop('expanded_pi')
                self.dim['pi_type'] = pi_type
            elif pi_type == 'expanded':
                expanded_pi = self.pi
                self.var_dict['expanded_pi'] = expanded_pi
                self.var_dict.pop('logit_pi')
                self.dim['pi_type'] = pi_type
            else:
                raise ValueError("Unrecognized pi_type: {0}".format(pi_type))
        return

    @property
    def logit_pi(self):
        if self.pi_type == 'logit':
            logit_pi = self.var_dict['logit_pi']
            return logit_pi
        else:
            logit_pi = np.log(self.pi + 1e-9)
            logit_pi -= np.outer(np.mean(logit_pi, axis=1),
                    np.ones(self.num_states))
            return logit_pi
    @logit_pi.setter
    def logit_pi(self, logit_pi):
        if self.pi_type != 'logit':
            raise RuntimeError("pi_type != logit")
        self.var_dict['logit_pi'] = logit_pi
        return
    @property
    def expanded_pi(self):
        if self.pi_type == 'expanded':
            expanded_pi = self.var_dict['expanded_pi']
            return expanded_pi
        else:
            return self.pi
    @expanded_pi.setter
    def expanded_pi(self, expanded_pi):
        if self.pi_type != 'expanded':
            raise RuntimeError("pi_type != expanded")
        self.var_dict['expanded_pi'] = np.abs(expanded_pi)
        return
    @property
    def pi(self):
        if self.pi_type == 'logit':
            pi = np.exp(self.logit_pi - np.outer(
                logsumexp(self.logit_pi, axis=1), np.ones(self.num_states))
                    )
        elif self.pi_type == 'expanded':
            pi = np.abs(self.expanded_pi) / np.outer(
                np.sum(np.abs(self.expanded_pi), axis=1),
                np.ones(self.num_states)
                    )
        return pi
    @property
    def num_states(self):
        return self.dim['num_states']


class PiPrior(object):
    # Mixin for Pi variable
    def _set_hyperparams(self, **kwargs):
        if 'alpha_Pi' in kwargs:
            num_states, num_states2 = np.shape(kwargs['alpha_Pi'])
        else:
            raise ValueError("alpha_Pi must be provided")
        if num_states != num_states2:
            raise ValueError("alpha_Pi must be square, not {0}".format(
                (num_states, num_states2)))
        if "num_states" in self.dim:
            if num_states != self.dim['num_states']:
                raise ValueError("num_states do not match existing dims")
        else:
            self.dim['num_states'] = num_states
        self.hyperparams['alpha_Pi'] = kwargs['alpha_Pi']
        super()._set_hyperparams(**kwargs)
        return

    def _sample_prior_var_dict(self, var_dict, **kwargs):
        pi_type = kwargs.get("pi_type", "logit")
        alpha_Pi = self.hyperparams['alpha_Pi']
        Pi = np.array([np.random.dirichlet(alpha_Pi_k)
            for alpha_Pi_k in alpha_Pi])
        if pi_type == "logit":
            logit_pi = np.log(Pi)
            var_dict['logit_pi'] = logit_pi
        elif pi_type == "expanded":
            var_dict['expanded_pi'] = Pi
        else:
            raise ValueError("Unrecognized pi_type {0}".format(
                pi_type))
        var_dict = super()._sample_prior_var_dict(var_dict, **kwargs)
        return var_dict

    def _sample_post_var_dict(self, var_dict, sufficient_stat, **kwargs):
        pi_type = kwargs.get("pi_type", "logit")
        alpha_Pi = \
                self.hyperparams['alpha_Pi'] + sufficient_stat['alpha_Pi']
        Pi = np.array([np.random.dirichlet(alpha_Pi_k)
            for alpha_Pi_k in alpha_Pi])

        if pi_type == "logit":
            logit_pi = np.log(Pi)
            var_dict['logit_pi'] = logit_pi
        elif pi_type == "expanded":
            var_dict['expanded_pi'] = Pi
        else:
            raise ValueError("Unrecognized pi_type {0}".format(
                pi_type))
        var_dict = super()._sample_post_var_dict(
                var_dict, sufficient_stat, **kwargs)
        return var_dict

    def _logprior(self, logprior, parameters, **kwargs):
        alpha_Pi = self.hyperparams['alpha_Pi']
        for pi_k, alpha_Pi_k in zip(parameters.pi, alpha_Pi):
            logprior += scipy.stats.dirichlet.logpdf(pi_k/np.sum(pi_k),
                alpha=alpha_Pi_k)

        logprior = super()._logprior(logprior, parameters, **kwargs)
        return logprior

    def _grad_logprior(self, grad, parameters, **kwargs):
        alpha_Pi = self.hyperparams['alpha_Pi']
        if parameters.pi_type == "logit":
            grad['logit_pi'] = np.array([
                -pi_k*np.sum(alpha_Pi_k-1.0) + (alpha_Pi_k-1.0)
                for pi_k, alpha_Pi_k in zip(parameters.pi, alpha_Pi)
                ])
        elif parameters.pi_type == "expanded":
            grad['expanded_pi'] = np.array([
                (-pi_k*np.sum(alpha_Pi_k-1.0) + (alpha_Pi_k-1.0)) * exp_pi_k
                for pi_k, exp_pi_k, alpha_Pi_k in zip(
                    parameters.pi, parameters.expanded_pi, alpha_Pi)
                ])
        else:
            RuntimeError("Unrecognized pi_type")

        grad = super()._grad_logprior(grad, parameters, **kwargs)
        return grad

    @classmethod
    def _get_default_kwargs(cls, default_kwargs, **kwargs):
        num_states = kwargs['num_states']
        default_kwargs['alpha_Pi'] = np.ones((num_states, num_states))
        default_kwargs = super()._get_default_kwargs(default_kwargs, **kwargs)
        return default_kwargs

    @classmethod
    def _get_prior_kwargs(cls, prior_kwargs, parameters, **kwargs):
        num_states = parameters.num_states
        prior_kwargs['alpha_Pi'] = np.ones((num_states, num_states))
        prior_kwargs = super()._get_prior_kwargs(
                prior_kwargs, parameters, **kwargs)
        return prior_kwargs


class PiPreconditioner(object):
    def __init__(self, expanded_pi_base = 0.0, **kwargs):
        self.expanded_pi_base = expanded_pi_base
        super().__init__(**kwargs)
        return

    def _precondition(self, precond_grad, grad, parameters, **kwargs):
        if parameters.pi_type == 'logit':
            precond_grad['logit_pi'] = grad['logit_pi']
        elif parameters.pi_type == 'expanded':
            precond_grad['expanded_pi'] =  (grad['expanded_pi'] *
                    (self.expanded_pi_base + np.abs(parameters.expanded_pi)))
        else:
            raise RuntimeError("Unrecognized pi_type")
        precond_grad = super()._precondition(precond_grad, grad,
                parameters, **kwargs)
        return precond_grad

    def _precondition_noise(self, noise, parameters, **kwargs):
        if parameters.pi_type == 'logit':
            noise['logit_pi'] = np.random.normal(
                    loc=0, size=parameters.logit_pi.shape)
        elif parameters.pi_type == 'expanded':
            noise['expanded_pi'] = (
                    (self.expanded_pi_base +
                        np.abs(parameters.expanded_pi))**0.5 *
                    np.random.normal(loc=0, size=parameters.logit_pi.shape)
                )
        else:
            raise RuntimeError("Unrecognized pi_type")
        super()._precondition_noise(noise, parameters, **kwargs)
        return noise

    def _correction_term(self, correction, parameters, **kwargs):
        if parameters.pi_type == 'logit':
            correction['logit_pi'] = parameters.logit_pi * 0.0
        elif parameters.pi_type == 'expanded':
            correction['expanded_pi'] = np.ones(
                    (parameters.num_states, parameters.num_states),
                    dtype=float)
        else:
            raise RuntimeError("Unrecognized pi_type")
        super()._correction_term(correction, parameters, **kwargs)
        return correction




