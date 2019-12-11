import numpy as np
import scipy.stats
from scipy.special import expit, logit
from ..base_parameters import (
        ParamHelper, PriorHelper, PrecondHelper,
        get_value_func, get_hyperparam_func, get_dim_func,
        set_value_func, set_hyperparam_func,
        )
from .._utils import (
        normal_logpdf,
        matrix_normal_logpdf,
        pos_def_mat_inv,
        varp_stability_projection,
        )
import logging
logger = logging.getLogger(name=__name__)

## Implementations of GARCH

# Single Square
class GARCHParamHelper(ParamHelper):
    def __init__(self):
        self.names = ['log_mu', 'logit_phi', 'logit_lambduh']
        self.dim_names = []
        return

    def set_var(self, param, **kwargs):
        for name in self.names:
            if name in kwargs:
                param.var_dict[name] = np.atleast_1d(kwargs[name]).astype(float)
            else:
                raise ValueError("{} not provided".format(self.name))
        return

    def project_parameters(self, param, **kwargs):
        for name in self.names:
            name_kwargs = kwargs.get(name, {})
            if name_kwargs.get('fixed') is not None:
                param.var_dict[name] = name_kwargs['fixed'].copy()
        return

    def from_dict_to_vector(self, vector_list, var_dict, **kwargs):
        for name in self.names:
            vector_list.append(var_dict[name].flatten())
        return

    def from_vector_to_dict(self, var_dict, vector, vector_index, **kwargs):
        for name in self.names:
            var = np.reshape(vector[vector_index:vector_index+1], (1))
            var_dict[name] = var
            vector_index += 1
        return vector_index

    def get_properties(self):
        properties = {}
        for name in self.names:
            properties[name] = property(
                    fget=get_value_func(name),
                    fset=set_value_func(name),
                    )
        properties['mu'] = property(fget=fget_mu)
        properties['phi'] = property(fget=fget_phi)
        properties['lambduh'] = property(fget=fget_lambduh)
        properties['alpha'] = property(fget=fget_alpha)
        properties['beta'] = property(fget=fget_beta)
        properties['gamma'] = property(fget=fget_gamma)
        return properties

def fget_mu(self):
    mu = np.exp(self.var_dict['log_mu'])
    return mu

def fget_phi(self):
    phi = expit(self.var_dict['logit_phi'])
    return phi

def fget_lambduh(self):
    lambduh = expit(self.var_dict['logit_lambduh'])
    return lambduh

def fget_alpha(self):
    alpha = self.mu * (1-self.phi)
    return alpha

def fget_beta(self):
    beta = self.phi * self.lambduh
    return beta

def fget_gamma(self):
    gamma = self.phi * (1-self.lambduh)
    return gamma

class GARCHPriorHelper(PriorHelper):
    def __init__(self):
        self.names = ['log_mu', 'logit_phi', 'logit_lambduh']
        self.hyperparam_names = [
                'scale_mu', 'shape_mu',
                'alpha_phi', 'beta_phi',
                'alpha_lambduh', 'beta_lambduh',
                ]
        return

    def set_hyperparams(self, prior, **kwargs):
        for name in self.hyperparam_names:
            if name in kwargs:
                prior.hyperparams[name] = kwargs[name]
            else:
                raise ValueError("{} must be provided".format(name))
        return

    def sample_prior(self, prior, var_dict, **kwargs):
        # mu
        mu = scipy.stats.invgamma(
                a=prior.hyperparams['shape_mu'],
                scale=prior.hyperparams['scale_mu']
                ).rvs()
        var_dict['log_mu'] = np.log(mu)

        # phi
        phi = scipy.stats.beta(
                a=prior.hyperparams['alpha_phi'],
                b=prior.hyperparams['beta_phi'],
                ).rvs()
        var_dict['logit_phi'] = logit(phi)

        # lambduh
        lambduh = scipy.stats.beta(
                a=prior.hyperparams['alpha_lambduh'],
                b=prior.hyperparams['beta_lambduh'],
                ).rvs()
        var_dict['logit_lambduh'] = logit(lambduh)
        return

    def sample_posterior(self, prior, var_dict, sufficient_stat, **kwargs):
        raise NotImplementedError("GARCH is not conjugate")

    def logprior(self, prior, logprior, parameters, **kwargs):
        logprior += scipy.stats.invgamma(
                a=prior.hyperparams['shape_mu'],
                scale=prior.hyperparams['scale_mu']
                ).logpdf(parameters.mu)
        logprior += scipy.stats.beta(
                a=prior.hyperparams['alpha_phi'],
                b=prior.hyperparams['beta_phi']
                ).logpdf((1+parameters.phi)/2.0)
        logprior += scipy.stats.beta(
                a=prior.hyperparams['alpha_lambduh'],
                b=prior.hyperparams['beta_lambduh']
                ).logpdf((1+parameters.lambduh)/2.0)
        return logprior

    def grad_logprior(self, prior, grad, parameters, **kwargs):
        grad['log_mu'] = - prior.hyperparams['shape_mu'] - 1 + \
            prior.hyperparams['scale_mu'] / parameters.mu

        grad['logit_phi'] = (
            (prior.hyperparams['alpha_phi'] - 1) / (1 + parameters.phi) -
            (prior.hyperparams['beta_phi'] - 1) / (1 - parameters.phi)
            ) * parameters.phi * (1-parameters.phi)

        grad['logit_lambduh'] = (
            (prior.hyperparams['alpha_lambduh'] - 1) / (1 + parameters.lambduh) -
            (prior.hyperparams['beta_lambduh'] - 1) / (1 - parameters.lambduh)
            ) * parameters.lambduh * (1-parameters.lambduh)
        return

    def get_prior_kwargs(self, prior_kwargs, parameters, **kwargs):
        var = kwargs['var']
        if var > 1:
            var = 1
        prior_kwargs['scale_mu'] = var + 2
        prior_kwargs['shape_mu'] = prior_kwargs['scale_mu'] + 1
        prior_kwargs['alpha_phi'] = 1 + 19*var**-1
        prior_kwargs['beta_phi'] = prior_kwargs['alpha_phi'] / 9
        prior_kwargs['alpha_lambduh'] = 1 + 19*var**-1
        prior_kwargs['beta_lambduh'] = prior_kwargs['alpha_lambduh'] / 9
        return

    def get_default_kwargs(self, default_kwargs, **kwargs):
        var = kwargs['var']
        if var > 1:
            var = 1
        default_kwargs['scale_mu'] = var + 2
        default_kwargs['shape_mu'] = default_kwargs['scale_mu'] + 1
        default_kwargs['alpha_phi'] = 1 + 19*var**-1
        default_kwargs['beta_phi'] = default_kwargs['alpha_phi'] / 9
        default_kwargs['alpha_lambduh'] = 1 + 19*var**-1
        default_kwargs['beta_lambduh'] = default_kwargs['alpha_lambduh'] / 9
        return

