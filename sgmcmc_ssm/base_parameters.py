"""

Base Parameters

"""
import numpy as np
from copy import deepcopy

# Param and ParamHelper Class
class BaseParameters(object):
    """ Base Class for Parameters """
    # List of ParamHelpers
    _param_helper_list = []

    # Set Attributes (must be copied in each subclass)
    for param_helper in _param_helper_list:
        properties = param_helper.get_properties()
        for name, prop in properties.items():
            vars()[name] = prop

    # Methods
    def __init__(self, **kwargs):
        self.dim = {}
        self.var_dict = {}
        for param_helper in self._param_helper_list:
            param_helper.set_var(self, **kwargs)
        return

    def _set_check_dim(self, **kwargs):
        for dim_key, dim_value in kwargs.items():
            if dim_key in self.dim:
                if dim_value != self.dim[dim_key]:
                    raise ValueError(
                        "{0} does not match existing dims {1} != {2}".format(
                            dim_key, dim_value, self.dim[dim_key])
                        )
            else:
                self.dim[dim_key] = dim_value
        return

    def as_dict(self, copy=True):
        """ Return Dict """
        if copy:
            return self.var_dict.copy()
        else:
            return self.var_dict

    def as_vector(self):
        """ Return flatten vector representation """
        return self.from_dict_to_vector(self.var_dict, **self.dim)

    def from_vector(self, vector):
        """ Set Vars from vector """
        var_dict = self.from_vector_to_dict(vector, **self.dim)
        self.set_var_dict(**var_dict)
        return

    @classmethod
    def from_dict_to_vector(cls, var_dict, **dim):
        """ Convert dict of variable-values to flattened vector """
        vector_list = []
        for param_helper in cls._param_helper_list:
            param_helper.from_dict_to_vector(vector_list, var_dict, **dim)
        return np.concatenate([vec for vec in vector_list])

    @classmethod
    def from_vector_to_dict(cls, vector, **dim):
        """ Convert flattened vector to dict of variable-values """
        var_dict = {}
        vector_index = 0
        for param_helper in cls._param_helper_list:
            vector_index = param_helper.from_vector_to_dict(
                var_dict, vector, vector_index, **dim)
        return var_dict

    def __iadd__(self, other):
        if isinstance(other, dict):
            for key in self.var_dict:
                self.var_dict[key] += other[key]
        else:
            raise TypeError("Addition only defined for dict not {0}".format(
                type(other)))
        return self
    def __add__(self, other):
        out = self.copy()
        out += other
        return out
    def __radd__(self, other):
        return self + other

    def copy(self):
        new_obj = type(self)(**deepcopy(self.var_dict))
        return new_obj

    def project_parameters(self, **kwargs):
        """ Project Parameters using passed options """
        for param_helper in self._param_helper_list:
            param_helper.project_parameters(self, **kwargs)
        return self

class ParamHelper(object):
    """ Base Class for ParamHelper """
    def __init__(self, name='theta', dim_names=None):
        self.name = name
        self.dim_names = [] if dim_names is None else dim_names
        return

    def set_var(self, param, **kwargs):
        raise NotImplementedError()

    def project_parameters(self, param, **kwargs):
        raise NotImplementedError()

    def from_dict_to_vector(self, vector_list, var_dict, **kwargs):
        vector_list.append(var_dict[self.name].flatten())
        return

    def from_vector_to_dict(self, var_dict, vector, vector_index, **kwargs):
        raise NotImplementedError()
        return vector_index

    def get_properties(self):
        # Returns a dict with key-property
        raise NotImplementedError()
        return {}

# Prior and PriorHelper Class
class BasePrior(object):
    """ Base Class for Priors """
    # Set Parameters class (should be overriden in subclass)
    _Parameters = BaseParameters

    # List of PriorHelpers
    _prior_helper_list = []

    # Methods
    def __init__(self, **kwargs):
        self.dim = {}
        self.hyperparams = {}
        for prior_helper in self._prior_helper_list:
            prior_helper.set_hyperparams(self, **kwargs)
        return

    def sample_prior(self, **kwargs):
        """ sample parameters from prior """
        var_dict = {}
        for prior_helper in self._prior_helper_list:
            prior_helper.sample_prior(self, var_dict, **kwargs)
        parameters = self._Parameters(**var_dict)
        return parameters

    def sample_posterior(self, sufficient_stat, **kwargs):
        """ sample parameters from posterior (for conjugate models) """
        var_dict = {}
        for prior_helper in self._prior_helper_list:
            prior_helper.sample_posterior(
                    self, var_dict, sufficient_stat, **kwargs)
        parameters = self._Parameters(**var_dict)
        return parameters

    def logprior(self, parameters, **kwargs):
        """ Return the log prior density for parameters

        Args:
            parameters (Parameters)

        Returns:
            logprior (double)
        """
        logprior = 0.0
        for prior_helper in self._prior_helper_list:
            logprior = prior_helper.logprior(
                    self, logprior, parameters, **kwargs)
        return np.asscalar(logprior)

    def grad_logprior(self, parameters, **kwargs):
        """ Return the gradient of log prior density for parameters

        Args:
            parameters (Parameters)

        Returns:
            grad (dict)
        """
        grad = {}
        for prior_helper in self._prior_helper_list:
            prior_helper.grad_logprior(self, grad, parameters, **kwargs)
        return grad

    @classmethod
    def generate_prior(cls, parameters, from_mean=False, var=1.0):
        """ Generate Prior to have parameters as its mean """
        prior_kwargs = {}
        for prior_helper in cls._prior_helper_list:
            prior_helper.get_prior_kwargs(prior_kwargs, parameters,
                    from_mean=from_mean, var=var)
        return cls(**prior_kwargs)

    @classmethod
    def generate_default_prior(cls, var=100.0, **kwargs):
        """ Generate Default Prior """
        default_kwargs = {}
        for prior_helper in cls._prior_helper_list:
            prior_helper.get_default_kwargs(default_kwargs, var=var, **kwargs)
        return cls(**default_kwargs)

    def _set_check_dim(self, **kwargs):
        for dim_key, dim_value in kwargs.items():
            if dim_key in self.dim:
                if dim_value != self.dim[dim_key]:
                    raise ValueError(
                        "{0} does not match existing dims {1} != {2}".format(
                            dim_key, dim_value, self.dim[dim_key])
                        )
            else:
                self.dim[dim_key] = dim_value
        return

class PriorHelper(object):
    """ Base Class for PriorHelper """
    def __init__(self, name='theta', dim_names=None):
        self.name = name
        self.dim_names = [] if dim_names is None else dim_names
        return

    def set_hyperparams(self, prior, **kwargs):
        raise NotImplementedError()

    def sample_prior(self, prior, var_dict, **kwargs):
        raise NotImplementedError()

    def sample_posterior(self, prior, var_dict, set_value_func, **kwargs):
        raise NotImplementedError()

    def logprior(self, prior, logprior, parameters, **kwargs):
        raise NotImplementedError()
        return logprior

    def grad_logprior(self, prior, grad, parameters, **kwargs):
        raise NotImplementedError()

    def get_prior_kwargs(self, prior_kwargs, parameters, **kwargs):
        raise NotImplementedError()

    def get_default_kwargs(self, default_kwargs, **kwargs):
        raise NotImplementedError()


# Preconditioner and PrecondHelper Class
class BasePreconditioner(object):
    """ Base Class for Preconditioner """
    # List of PrecondHelpers
    _precond_helper_list = []

    # Methods
    def __init__(self, **kwargs):
        for precond_helper in self._precond_helper_list:
            precond_helper.set_preconditioner_vars(self, **kwargs)
        return

    def precondition(self, grad, parameters, scale=1.0, **kwargs):
        """ Return dict with precondition gradients """
        precond_grad = {}
        for precond_helper in self._precond_helper_list:
            precond_helper.precondition(self, precond_grad, grad, parameters,
                    **kwargs)
        for var in precond_grad:
            precond_grad[var] *= scale
        return precond_grad

    def precondition_noise(self, parameters, scale=1.0):
        """ Return dict with precondition noise """
        noise = {}
        for precond_helper in self._precond_helper_list:
            precond_helper.precondition_noise(self, noise, parameters)
        for var in noise:
            noise[var] *= scale**0.5
        return noise

    def correction_term(self, parameters, scale=1.0):
        """ Return dict with correction term """
        correction = {}
        for precond_helper in self._precond_helper_list:
            precond_helper.correction_term(self, correction, parameters)
        for var in correction:
            correction[var] = correction[var]*scale
        return correction

class PrecondHelper(object):
    """ Base Class for PrecondHelper """
    def __init__(self, name='theta', dim_names=None):
        self.name = name
        self.dim_names = [] if dim_names is None else dim_names
        return

    def set_preconditioner_vars(self, preconditioner, **kwargs):
        return

    def precondition(self, preconditioner, precond_grad, grad, parameters,
        **kwargs):
        return

    def precondition_noise(self, preconditioner, noise, parameters, **kwargs):
        return

    def correction_term(self, preconditioner, correction, parameters, **kwargs):
        return

# Property Helper Functions
def get_value_func(name):
    def fget(self):
        return self.var_dict[name]
    return fget

def set_value_func(name):
    def fset(self, value):
        self.var_dict[name] = value
        return
    return fset

def get_dim_func(name):
    def fget(self):
        return self.dim[name]
    return fget

def get_hyperparam_func(name):
    def fget(self):
        return self.hyperparams[name]
    return fget

def set_hyperparam_func(name):
    def fset(self, value):
        self.hyperparams[name] = value
        return
    return fset


# Temporary Demo









