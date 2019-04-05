"""

Base Parameters

"""
import numpy as np
from copy import deepcopy

class BaseParameters(object):
    """ Base Class for Parameters Classes """
    def __init__(self, **kwargs):
        # kwargs should be (variable, value) pairs
        self.set_dim(**kwargs)
        self.set_var_dict(**kwargs)
        return

    def set_dim(self, **kwargs):
        """ Set Dimensions based on kwargs: (variable, value) pairs """
        self.dim = {}
        self._set_dim(**kwargs)

    def _set_dim(self, **kwargs):
        return

    def set_var_dict(self, **kwargs):
        """ Set variable values based on kwargs: (variable, value) pairs """
        self.var_dict = {}
        self._set_var_dict(**kwargs)

    def _set_var_dict(self, **kwargs):
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
    def from_vector_to_dict(cls, vector, **dim):
        """ Convert flattened vector to Dict of variabls-values """
        var_dict = {}
        var_dict = cls._from_vector_to_dict(var_dict, vector, **dim)
        return var_dict

    @classmethod
    def from_dict_to_vector(cls, var_dict, **dim):
        """ Convert dict of variabls-values to flattened vector """
        vector_list = []
        vector_list = cls._from_dict_to_vector(vector_list, var_dict, **dim)
        return np.concatenate([vec for vec in vector_list])

    @classmethod
    def _from_vector_to_dict(cls, var_dict, vector, **kwargs):
        return var_dict

    @classmethod
    def _from_dict_to_vector(cls, vector_list, var_dict, **kwargs):
        return vector_list

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
        """ Project Parameters to valid values + fix constants """
        self._project_parameters(**kwargs)
        return self

    def _project_parameters(self, **kwargs):
        return

    def is_dim_valid(self):
        """ Check that var_dict's variables are correct dimension """
        try:
            self.set_dim(**self.var_dict)
        except Exception as e:
            print(e)
            return False
        return True

class BasePrior(object):
    """ Base Class for Priors """
    def __init__(self, **kwargs):
        self.dim = {}
        self.hyperparams = {}
        self._set_hyperparams(**kwargs)
        return

    def sample_prior(self, **kwargs):
        """ sample parameters from prior """
        var_dict = {}
        var_dict = self._sample_prior_var_dict(var_dict, **kwargs)
        parameters = self._parameters(**var_dict)
        return parameters

    def sample_posterior(self, sufficient_stat, **kwargs):
        """ sample parameters from posterior (for Gibbs) """
        var_dict = {}
        var_dict = self._sample_post_var_dict(var_dict, sufficient_stat, **kwargs)
        parameters = self._parameters(**var_dict)
        return parameters

    def logprior(self, parameters, **kwargs):
        """ Return the log prior density for parameters

        Args:
            parameters (parameters)

        Returns:
            logprior (double)
        """
        logprior = 0.0
        logprior = self._logprior(logprior, parameters, **kwargs)
        return logprior

    def grad_logprior(self, parameters, **kwargs):
        """ Return the gradient of log prior density for parameters

        Args:
            parameters (parameters)

        Returns:
            grad (dict)
        """
        grad = {}
        grad = self._grad_logprior(grad, parameters, **kwargs)
        return grad

    @classmethod
    def generate_prior(cls, parameters, from_mean=False, var=1.0):
        """ Generate Prior to have parameters as its mean """
        prior_kwargs = {}
        prior_kwargs = cls._get_prior_kwargs(prior_kwargs,
                parameters=parameters, from_mean=from_mean, var=var)
        return cls(**prior_kwargs)

    @classmethod
    def generate_default_prior(cls, var=100.0, **kwargs):
        """ Generate Default Prior """
        default_kwargs = {}
        default_kwargs = cls._get_default_kwargs(default_kwargs,
                var=var, **kwargs)
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


    # Functions for Mixins
    def _set_hyperparams(self, **kwargs):
        return

    def _sample_prior_var_dict(self, var_dict, **kwargs):
        return var_dict

    def _sample_post_var_dict(self, var_dict, sufficient_stat, **kwargs):
        return var_dict

    def _logprior(self, logprior, parameters, **kwargs):
        return logprior

    def _grad_logprior(self, grad, parameters, **kwargs):
        return grad

    @classmethod
    def _get_default_kwargs(cls, default_kwargs, **kwargs):
        return default_kwargs

    @classmethod
    def _get_prior_kwargs(cls, prior_kwargs, parameters, **kwargs):
        return prior_kwargs

    # Function for child class
    @staticmethod
    def _parameters(**kwargs):
        raise NotImplementedError("should return Parameters(**kwargs)")

class BasePreconditioner(object):
    """ Base Class for Preconditioners """
    def __init__(self):
        return

    def precondition(self, grad, parameters, scale=1.0):
        """ Return dict with precondition gradients """
        precond_grad = {}
        precond_grad = self._precondition(precond_grad, grad, parameters)
        for var in precond_grad:
            precond_grad[var] *= scale
        return precond_grad

    def precondition_noise(self, parameters, scale=1.0):
        """ Return dict with precondition noise """
        noise = {}
        noise = self._precondition_noise(noise, parameters)
        for var in noise:
            noise[var] *= scale**0.5
        return noise

    def correction_term(self, parameters, scale=1.0):
        """ Return dict with correction term """
        correction = {}
        correction = self._correction_term(correction, parameters, scale=scale)
        for var in correction:
            correction[var] = correction[var]*scale
        return correction

    # For Mixins to implement
    def _precondition(self, precond_grad, grad, parameters, **kwargs):
        return precond_grad

    def _precondition_noise(self, noise, parameters, **kwargs):
        return noise

    def _correction_term(self, correction, parameters, **kwargs):
        return correction

