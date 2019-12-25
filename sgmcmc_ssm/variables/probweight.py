import numpy as np
import scipy.stats
from scipy.special import logit, expit, logsumexp
from ..base_parameters import (
        ParamHelper, PriorHelper, PrecondHelper,
        get_value_func, get_hyperparam_func, get_dim_func,
        set_value_func, set_hyperparam_func,
        )
import logging
logger = logging.getLogger(name=__name__)

## Implementations of Bernoulli + Transition Matrix Parameters

# Bernoulli Var
class BernoulliParamHelper(ParamHelper):
    def __init__(self, name='pi'):
        self.name = name
        self._logit_name = 'logit_{}'.format(name)
        self.dim_names = []
        return

    def set_var(self, param, **kwargs):
        if self._logit_name in kwargs:
            if np.size(kwargs[self._logit_name]) != 1:
                raise ValueError("{} must be 1D scalar".format(self._logit_name))
            param.var_dict[self._logit_name] = np.atleast_1d(
                    kwargs[self._logit_name]).astype(float)
        else:
            raise ValueError("{} not provided".format(self.name))
        return

    def project_parameters(self, param, **kwargs):
        name_kwargs = kwargs.get(self.name, {})
        if name_kwargs.get('fixed') is not None:
            param.var_dict[self._logit_name] = name_kwargs['fixed'].copy()

        return

    def from_dict_to_vector(self, vector_list, var_dict, **kwargs):
        vector_list.append(var_dict[self._logit_name].flatten())
        return

    def from_vector_to_dict(self, var_dict, vector, vector_index, **kwargs):
        logit_pi = np.reshape(vector[vector_index:vector_index+1], (1))
        var_dict[self._logit_name] = logit_pi
        return vector_index+1

    def get_properties(self):
        properties = {}
        properties[self._logit_name] = property(
                fget=get_value_func(self._logit_name),
                fset=set_value_func(self._logit_name),
                doc="{0} is a scalar log-odds".format(
                    self._logit_name),
                )
        properties[self.name] = property(
                fget=get_pi_func(self.name),
                doc="{0} is a probability in [0,1]".format(self.name),
                )
        return properties

def get_pi_func(name):
    def fget(self):
        logit_pi = getattr(self, "logit_{0}".format(name))
        pi = expit(logit_pi)
        return pi
    return fget

class BernoulliPriorHelper(PriorHelper):
    def __init__(self, name='pi', dim_names=None):
        self.name = name
        self._logit_name = 'logit_{0}'.format(name)
        self._alpha_name = 'alpha_{0}'.format(name)
        self._beta_name = 'beta_{0}'.format(name)
        self.dim_names = [] if dim_names is None else dim_names
        return

    def set_hyperparams(self, prior, **kwargs):
        if self._alpha_name in kwargs:
            alpha = kwargs[self._alpha_name]
            if np.size(alpha) != 1 or alpha < 0:
                raise ValueError("{} must be a nonnegative scalar".format(
                    self._alpha_name))
            prior.hyperparams[self._alpha_name] = alpha
        else:
            raise ValueError("{} must be provided".format(self._alpha_name))
        if self._beta_name in kwargs:
            beta = kwargs[self._beta_name]
            if np.size(beta) != 1 or beta < 0:
                raise ValueError("{} must be a nonnegative scalar".format(
                    self._beta_name))
            prior.hyperparams[self._beta_name] = beta
        else:
            raise ValueError("{} must be provided".format(self._beta_name))
        return

    def sample_prior(self, prior, var_dict, **kwargs):
        alpha = prior.hyperparams[self._alpha_name]
        beta = prior.hyperparams[self._beta_name]
        pi = scipy.stats.beta(a=alpha, b=beta).rvs()
        var_dict[self._logit_name] = np.array(logit(pi))
        return

    def sample_posterior(self, prior, var_dict, sufficient_stat, **kwargs):
        raise NotImplementedError()
        return

    def logprior(self, prior, logprior, parameters, **kwargs):
        alpha = prior.hyperparams[self._alpha_name]
        beta = prior.hyperparams[self._beta_name]
        logprior += scipy.stats.beta.logpdf(
                getattr(parameters, self.name),
                a=alpha,
                b=beta,
                )
        return np.asscalar(logprior)

    def grad_logprior(self, prior, grad, parameters, **kwargs):
        alpha = prior.hyperparams[self._alpha_name]
        beta = prior.hyperparams[self._beta_name]
        pi = getattr(parameters, self.name)
        grad[self._logit_name] = np.array((alpha-1)*(1-pi) - (beta-1)*pi)
        return

    def get_prior_kwargs(self, prior_kwargs, parameters, **kwargs):
        pi = getattr(parameters, self.name)
        if kwargs.get('from_mean', False):
            alpha = pi, 1-pi
        else:
            alpha, beta = 0.5, 0.5
        prior_kwargs[self._alpha_name] = alpha
        prior_kwargs[self._beta_name] = beta
        return

    def get_default_kwargs(self, default_kwargs, **kwargs):
        default_kwargs[self._alpha_name] = 0.5
        default_kwargs[self._beta_name] = 0.5
        return

class BernoulliPrecondHelper(PrecondHelper):
    def __init__(self, name='pi', dim_names=None):
        self.name = name
        self._logit_name = 'logit_{}'.format(self.name)
        return

    def precondition(self, preconditioner,
            precond_grad, grad, parameters, **kwargs):
        precond_grad[self._logit_name] = grad[self._logit_name]
        return

    def precondition_noise(self, preconditioner,
            noise, parameters, **kwargs):
        noise[self._logit_name] = np.random.normal(loc=0, size=(1))
        return

    def correction_term(self, preconditioner, correction, parameters, **kwargs):
        correction[self._logit_name] = np.zeros((1), dtype=float)
        return

# Transition Matrix
class TransitionMatrixParamHelper(ParamHelper):
    def __init__(self, name='pi', dim_names=None):
        self.name = name
        self._logit_name = "logit_{0}".format(name)
        self._expanded_name = "expanded_{0}".format(name)
        self.dim_names = ['num_states', '{0}_type'.format(name)]
        if dim_names is not None:
            self.dim_names = dim_names
        return

    def set_var(self, param, **kwargs):
        if self._logit_name in kwargs:
            num_states, num_states2 = np.shape(kwargs[self._logit_name])
            if num_states != num_states2:
                raise ValueError("{} must be square matrix".format(self._logit_name))
            param.var_dict[self._logit_name] = np.array(kwargs[self._logit_name]).astype(float)
            param._set_check_dim(**{self.dim_names[0]: num_states,
                                    self.dim_names[1]: 'logit',
                                    })

        elif self._expanded_name in kwargs:
            num_states, num_states2 = np.shape(kwargs[self._expanded_name])
            if num_states != num_states2:
                raise ValueError("{} must be square matrix".format(self._expanded_name))
            param.var_dict[self._expanded_name] = np.array(kwargs[self._expanded_name]).astype(float)
            param._set_check_dim(**{self.dim_names[0]: num_states,
                                    self.dim_names[1]: 'expanded',
                                    })
        else:
            raise ValueError("{} not provided".format(self.name))
        return

    def project_parameters(self, param, **kwargs):
        name_kwargs = kwargs.get(self.name, {})
        pi_type = getattr(param, self.dim_names[1])
        if pi_type == 'logit' and name_kwargs.get('center', False):
            # Center logit_pi to be stable
            logit_pi = param.var_dict[self._logit_name]
            logit_pi -= np.outer(np.mean(logit_pi, axis=1),
                    np.ones(self.num_states))
        if pi_type == 'expanded':
            param.var_dict[self._expanded_name] = \
                    np.abs(param.var_dict[self._expanded_name])
            if name_kwargs.get('center', False):
                param.var_dict[self._expanded_name] /= \
                        np.sum(param.var_dict[self._expanded_name], axis=1)
        if name_kwargs.get('fixed') is not None:
            if pi_type == 'logit':
                param.var_dict[self._logit_name] = name_kwargs['fixed'].copy()
            elif pi_type == 'expanded':
                param.var_dict[self._expanded_name] = name_kwargs['fixed'].copy()
        return

    def from_dict_to_vector(self, vector_list, var_dict, **kwargs):
        if self._logit_name in var_dict:
            vector_list.append(var_dict[self._logit_name].flatten())
        elif self._expanded_name in var_dict:
            vector_list.append(var_dict[self._expanded_name].flatten())
        else:
            raise RuntimeError("Missin either {0} or {1} in var_dict".format(
                self._logit_name, self._expanded_name))
        return

    def from_vector_to_dict(self, var_dict, vector, vector_index, **kwargs):
        num_states = kwargs[self.dim_names[0]]
        pi_type = kwargs[self.dim_names[1]]
        pi_mat = np.reshape(
                vector[vector_index:vector_index+num_states**2],
                (num_states, num_states))
        if pi_type == 'logit':
            var_dict[self._logit_name] = pi_mat
        elif pi_type == 'expanded':
            var_dict[self._expanded_name] = pi_mat
        else:
            raise ValueError("Unrecognized {0} {1}".format(
                self.dim_names[1], pi_type))
        return vector_index+num_states**2

    def get_properties(self):
        properties_kwargs = dict(
                pi_type_name=self.dim_names[1],
                logit_pi_name=self._logit_name,
                expanded_pi_name=self._expanded_name,
                )

        # Parameter Values
        properties = {}
        properties[self.name] = property(
                fget=get_pi_func2(**properties_kwargs),
                doc="{0} is a {1} by {1} stochastic matrix".format(
                    self.name, self.dim_names[0]),
                )
        properties[self._logit_name] = property(
                fget=get_logit_pi_func(**properties_kwargs),
                fset=set_logit_pi_func(**properties_kwargs),
                doc="{0} is the row-wise logit of {1}".format(
                    self._logit_name, self.name),
                )
        properties[self._expanded_name] = property(
                fget=get_expanded_pi_func(**properties_kwargs),
                fset=set_expanded_pi_func(**properties_kwargs),
                doc="{0} is the row-wise expanded mean of {1}".format(
                    self._logit_name, self.name),
                )

        # Dims
        properties[self.dim_names[0]] = property(
                fget=get_dim_func(self.dim_names[0]),
                )
        properties[self.dim_names[1]] = property(
                fget=get_dim_func(self.dim_names[1]),
                fset=set_pi_type(**properties_kwargs),
                )
        return properties

def get_pi_func2(pi_type_name, logit_pi_name, expanded_pi_name):
    def fget(self):
        pi_type = getattr(self, pi_type_name)
        if pi_type == 'logit':
            logit_pi = self.var_dict[logit_pi_name]
            pi = np.exp(logit_pi - np.outer(
                logsumexp(logit_pi, axis=1),
                np.ones(logit_pi.shape[1])
                ))
        elif pi_type == 'expanded':
            expanded_pi = self.var_dict[expanded_pi_name]
            pi = np.abs(expanded_pi) / np.outer(
                    np.sum(np.abs(expanded_pi), axis=1),
                    np.ones(expanded_pi.shape[1])
                    )
        else:
            raise ValueError("Unrecognized {0} {1}".format(
                pi_type_name, pi_type))
        return pi
    return fget

def get_logit_pi_func(pi_type_name, logit_pi_name, expanded_pi_name):
    def fget(self):
        pi_type = getattr(self, pi_type_name)
        if pi_type == 'logit':
            logit_pi = self.var_dict[logit_pi_name]
        elif pi_type == 'expanded':
            logit_pi = np.log(np.abs(self.var_dict[expanded_pi_name]) + 1e-99)
            logit_pi -= np.outer(
                    np.mean(logit_pi, axis=1),
                    np.ones(logit_pi.shape[1])
                    )
        else:
            raise ValueError("Unrecognized {0} {1}".format(
                pi_type_name, pi_type))
        return logit_pi
    return fget

def set_logit_pi_func(pi_type_name, logit_pi_name, expanded_pi_name):
    def fset(self, value):
        pi_type = getattr(self, pi_type_name)
        if pi_type == 'logit':
            self.var_dict[logit_pi_name] = value
        else:
            raise ValueError("{0} != 'logit'".format(pi_type_name))
        return
    return fset

def get_expanded_pi_func(pi_type_name, logit_pi_name, expanded_pi_name):
    def fget(self):
        pi_type = getattr(self, pi_type_name)
        if pi_type == 'logit':
            logit_pi = self.var_dict[logit_pi_name]
            expanded_pi = np.exp(logit_pi - np.outer(
                logsumexp(logit_pi, axis=1),
                np.ones(logit_pi.shape[1])
                ))
        elif pi_type == 'expanded':
            expanded_pi = self.var_dict[expanded_pi_name]
        else:
            raise ValueError("Unrecognized {0} {1}".format(
                pi_type_name, pi_type))
        return expanded_pi
    return fget

def set_expanded_pi_func(pi_type_name, logit_pi_name, expanded_pi_name):
    def fset(self, value):
        pi_type = getattr(self, pi_type_name)
        if pi_type == 'expanded':
            self.var_dict[expanded_pi_name] = value
        else:
            raise ValueError("{0} != 'expanded'".format(pi_type_name))
        return
    return fset

def set_pi_type(pi_type_name, logit_pi_name, expanded_pi_name):
    def fset(self, value):
        pi_type = getattr(self, pi_type_name)
        if pi_type == value:
            return
        else:
            if value == 'logit':
                logit_pi = getattr(self, logit_pi_name)
                self.var_dict[logit_pi_name] = logit_pi
                self.var_dict.pop(expanded_pi_name)
                self.dim[pi_type_name] = value
            elif value == 'expanded':
                expanded_pi = getattr(self, expanded_pi_name)
                self.var_dict[expanded_pi_name] = expanded_pi
                self.var_dict.pop(logit_pi_name)
                self.dim[pi_type_name] = value
            else:
                raise ValueError("Unrecognized {0} {1}".format(
                    pi_type_name, value))
        return
    return fset

class TransitionMatrixPriorHelper(PriorHelper):
    def __init__(self, name='pi', dim_names=None, var_row_name=None):
        self.name = name
        self._logit_name = "logit_{0}".format(name)
        self._expanded_name = "expanded_{0}".format(name)
        self._type_name = "{0}_type".format(name)
        self._alpha = 'alpha_{0}'.format(name)
        self.dim_names = ['num_states'] if dim_names is None else dim_names
        return

    def set_hyperparams(self, prior, **kwargs):
        if self._alpha in kwargs:
            num_states, num_states2 = np.shape(kwargs[self._alpha])
        else:
            raise ValueError("{} must be provided".format(self._alpha))
        if num_states != num_states2:
            raise ValueError("{} must be square".format(self._alpha))

        prior._set_check_dim(**{self.dim_names[0]: num_states})
        prior.hyperparams[self._alpha] = kwargs[self._alpha]
        return

    def sample_prior(self, prior, var_dict, **kwargs):
        pi_type = kwargs.get(self._type_name, 'logit')
        alpha = prior.hyperparams[self._alpha]
        pi = np.array([np.random.dirichlet(alpha_k) for alpha_k in alpha])
        if pi_type == 'logit':
            var_dict[self._logit_name] = np.log(pi+1e-99)
        elif pi_type == 'expanded':
            var_dict[self._expanded_name] = pi
        else:
            raise ValueError("Unrecognized {0} {1}".format(
                self._type_name, pi_type))
        return

    def sample_posterior(self, prior, var_dict, sufficient_stat, **kwargs):
        pi_type = kwargs.get(self._type_name, 'logit')
        alpha = prior.hyperparams[self._alpha] + \
                sufficient_stat[self.name]['alpha']
        pi = np.array([np.random.dirichlet(alpha_k) for alpha_k in alpha])
        if pi_type == 'logit':
            var_dict[self._logit_name] = np.log(pi+1e-99)
        elif pi_type == 'expanded':
            var_dict[self._expanded_name] = pi
        else:
            raise ValueError("Unrecognized {0} {1}".format(
                self._type_name, pi_type))
        return

    def logprior(self, prior, logprior, parameters, **kwargs):
        alpha = prior.hyperparams[self._alpha]
        pi = getattr(parameters, self.name)
        for pi_k, alpha_k in zip(pi, alpha):
            logprior += scipy.stats.dirichlet.logpdf(pi_k+1e-16, alpha=alpha_k)
        return logprior

    def grad_logprior(self, prior, grad, parameters, use_scir=False, **kwargs):
        pi_type = getattr(parameters, self._type_name)
        alpha = prior.hyperparams[self._alpha]
        if use_scir:
            if pi_type == "logit":
                grad[self._logit_name] = alpha
            elif pi_type == "expanded":
                grad[self._expanded_name] = alpha
        else:
            if pi_type == "logit":
                grad[self._logit_name] = np.array([
                    -pi_k*np.sum(alpha_k-1.0) + (alpha_k-1.0)
                    for pi_k, alpha_k in zip(getattr(parameters, self.name),
                        alpha)
                    ])
            elif pi_type == "expanded":
                grad[self._expanded_name] = np.array([
                    (-exp_pi_k*np.sum(alpha_k-1.0)/np.sum(exp_pi_k) + \
                            (alpha_k-1.0)) * exp_pi_k
                    for exp_pi_k, alpha_k in zip(
                        getattr(parameters, self._expanded_name), alpha)
                    ])
            else:
                RuntimeError("Unrecognized pi_type")
        return

    def get_prior_kwargs(self, prior_kwargs, parameters, **kwargs):
        var = kwargs['var']
        num_states = getattr(parameters, self.dim_names[0])
        if kwargs.get('from_mean', False):
            alpha = getattr(parameters, self.name)/(var/num_states)
        else:
            alpha = np.ones((num_states, num_states))/var
        prior_kwargs[self._alpha] = alpha
        return

    def get_default_kwargs(self, default_kwargs, **kwargs):
        num_states = kwargs[self.dim_names[0]]
        var = kwargs['var']
        alpha = np.ones((num_states, num_states))/var

        default_kwargs[self._alpha] = alpha
        return

class TransitionMatrixPrecondHelper(PrecondHelper):
    def __init__(self, name='pi', dim_names=None):
        self.name = name
        self._logit_name = "logit_{0}".format(name)
        self._expanded_name = "expanded_{0}".format(name)
        self._type_name = "{0}_type".format(name)
        self.dim_names = ['num_states'] if dim_names is None else dim_names
        return

    def precondition(self, preconditioner,
            precond_grad, grad, parameters, **kwargs):
        pi_type = getattr(parameters, self._type_name)
        if pi_type == 'logit':
            precond_grad[self._logit_name] = grad[self._logit_name]
        elif pi_type == 'expanded':
            if kwargs.get('use_scir', False):
                # Don't precondition if using SCIR
                precond_grad[self._expanded_name] = grad[self._expanded_name]
            else:
                precond_grad[self._expanded_name] = (grad[self._expanded_name] *
                    (1e-99 + np.abs(getattr(parameters, self._expanded_name))))
        else:
            raise RuntimeError("Unrecognized {0} {1}".format(
                self._type_name, pi_type))
        return

    def precondition_noise(self, preconditioner,
            noise, parameters, **kwargs):
        pi_type = getattr(parameters, self._type_name)
        num_states = getattr(parameters, self.dim_names[0])
        if pi_type == 'logit':
            noise[self._logit_name] = np.random.normal(loc=0,
                    size=(num_states, num_states))
        elif pi_type == 'expanded':
            noise[self._expanded_name] = (
                (1e-99 + np.abs(getattr(parameters, self._expanded_name)))**0.5 *
                np.random.normal(loc=0, size=(num_states, num_states))
                )
        else:
            raise RuntimeError("Unrecognized {0} {1}".format(
                self._type_name, pi_type))
        return

    def correction_term(self, preconditioner, correction, parameters, **kwargs):
        pi_type = getattr(parameters, self._type_name)
        num_states = getattr(parameters, self.dim_names[0])

        if pi_type == 'logit':
            correction[self._logit_name] = \
                    np.zeros((num_states, num_states), dtype=float)
        elif pi_type == 'expanded':
            correction[self._expanded_name] = \
                    np.ones((num_states, num_states), dtype=float)
        else:
            raise RuntimeError("Unrecognized {0} {1}".format(
                self._type_name, pi_type))
        return


