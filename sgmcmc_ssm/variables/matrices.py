import numpy as np
import scipy.stats
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
        tril_vector_to_mat,
        )
import logging
logger = logging.getLogger(name=__name__)

## Implementations of Vector, Square, Rectangular Parameters

# Single Square
class VectorParamHelper(ParamHelper):
    def __init__(self, name='mu', dim_names=None):
        self.name = name
        self.dim_names = ['n'] if dim_names is None else dim_names
        return

    def set_var(self, param, **kwargs):
        if self.name in kwargs:
            n = np.shape(kwargs[self.name])
            if np.ndim(kwargs[self.name]) != 1:
                raise ValueError("{} must be vector".format(self.name))
            param.var_dict[self.name] = np.array(kwargs[self.name]).astype(float)
            param._set_check_dim(**{self.dim_names[0]: n})
        else:
            raise ValueError("{} not provided".format(self.name))
        return

    def project_parameters(self, param, **kwargs):
        name_kwargs = kwargs.get(self.name, {})
        if name_kwargs.get('fixed') is not None:
            param.var_dict[self.name] = name_kwargs['fixed'].copy()
        return

    def from_dict_to_vector(self, vector_list, var_dict, **kwargs):
        vector_list.append(var_dict[self.name].flatten())
        return

    def from_vector_to_dict(self, var_dict, vector, vector_index, **kwargs):
        n = kwargs[self.dim_names[0]]
        mu = np.reshape(vector[vector_index:vector_index+n], (n))
        var_dict[self.name] = mu
        return vector_index+n

    def get_properties(self):
        properties = {}
        properties[self.name] = property(
                fget=get_value_func(self.name),
                fset=set_value_func(self.name),
                doc="{0} is a {1} vector".format(
                    self.name, self.dim_names[0]),
                )
        for dim_name in self.dim_names:
            properties[dim_name] = property(
                    fget=get_dim_func(dim_name),
                    )
        return properties

class VectorPriorHelper(PriorHelper):
    def __init__(self, name='mu', dim_names=None, var_row_name=None):
        self.name = name
        self._mean_name = 'mean_{0}'.format(name)
        self._var_col_name = 'var_col_{0}'.format(name)
        self._var_row_name = var_row_name
        self._lt_vec_name = 'L{0}inv_vec'.format(var_row_name)
        self.dim_names = ['n'] if dim_names is None else dim_names
        return

    def set_hyperparams(self, prior, **kwargs):
        if self._mean_name in kwargs:
            n = np.shape(kwargs[self._mean_name])
        else:
            raise ValueError("{} must be provided".format(self._mean_name))
        if self._var_col_name in kwargs:
            if not np.isscalar(kwargs[self._var_col_name]):
                raise ValueError("{} must be scalar".format(self._var_col_name))
        else:
            raise ValueError("{} must be provided".format(self._var_col_name))
        prior._set_check_dim(**{self.dim_names[0]: n})
        prior.hyperparams[self._mean_name] = kwargs[self._mean_name]
        prior.hyperparams[self._var_col_name] = kwargs[self._var_col_name]
        return

    def sample_prior(self, prior, var_dict, **kwargs):
        mean_mu = prior.hyperparams[self._mean_name]
        var_col_mu = prior.hyperparams[self._var_col_name]
        if self._var_row_name is not None:
            if self._lt_vec_name in var_dict:
                LQinv = tril_vector_to_mat(var_dict[self._lt_vec_name])
                Qinv = LQinv.dot(LQinv.T) + \
                        1e-9*np.eye(prior.dim[self.dim_names[0]])
            else:
                raise ValueError("Missing {}\n".format(self._lt_vec_name) +
                    "Perhaps {} must be earlier in _prior_helper_list".format(
                        self._var_row_name)
                        )
        else:
            Qinv = np.eye(prior.dim[self.dim_names[0]])

        var_dict[self.name] = np.random.multivariate_normal(
                mean=mean_mu,
                cov=var_col_mu*pos_def_mat_inv(Qinv),
                )
        return

    def sample_posterior(self, prior, var_dict, sufficient_stat, **kwargs):
        mean_mu = prior.hyperparams[self._mean_name]
        var_col_mu = prior.hyperparams[self._var_col_name]
        if self._var_row_name is not None:
            if self._lt_vec_name in var_dict:
                LQinv = tril_vector_to_mat(var_dict[self._lt_vec_name])
                Qinv = LQinv.dot(LQinv.T) + \
                        1e-9*np.eye(prior.dim[self.dim_names[0]])
            else:
                raise ValueError("Missing {}\n".format(self._lt_vec_name) +
                    "Perhaps {} must be earlier in _prior_helper_list".format(
                        self._var_row_name)
                    )
        else:
            Qinv = np.eye(self.prior[self.dim_names[0]])

        S_prevprev = var_col_mu**-1 + \
                sufficient_stat[self.name]['S_prevprev']
        S_curprev = mean_mu * var_col_mu**-1  + \
                sufficient_stat[self.name]['S_curprev']
        post_mean_mu = S_curprev/S_prevprev
        var_dict[self.name] = np.random.multivariate_normal(
                mean=post_mean_mu,
                cov=pos_def_mat_inv(Qinv)/S_prevprev,
                )
        return

    def logprior(self, prior, logprior, parameters, **kwargs):
        mean_mu = prior.hyperparams[self._mean_name]
        var_col_mu = prior.hyperparams[self._var_col_name]
        if self._var_row_name is not None:
            LQinv = tril_vector_to_mat(parameters.var_dict[self._lt_vec_name])
        else:
            LQinv = np.eye(prior.dim[self.dim_names[0]])
        logprior += normal_logpdf(parameters.var_dict[self.name],
                mean=mean_mu,
                Lprec=var_col_mu_k**-0.5 * LQinv,
                )
        return logprior

    def grad_logprior(self, prior, grad, parameters, **kwargs):
        mean_mu = prior.hyperparams[self._mean_name]
        var_col_mu = prior.hyperparams[self._var_col_name]
        mu = getattr(parameters, self.name)
        if self._var_row_name is not None:
            Qinv = getattr(parameters, '{}inv'.format(self._var_row_name))
        else:
            Qinv = np.eye(prior.dim[self.dim_names[0]])

        grad[self.name] = -1.0 * np.dot(var_col_mu**-1 * Qinv, mu - mean_mu)
        return

    def get_prior_kwargs(self, prior_kwargs, parameters, **kwargs):
        var = kwargs['var']
        mu = getattr(parameters, self.name)
        if kwargs.get('from_mean', False):
            mean_mu = mu.copy()
        else:
            mean_mu = np.zeros_like(mu)
        var_col_mu = var

        prior_kwargs[self._mean_name] = mean_mu
        prior_kwargs[self._var_col_name] = var_col_mu
        return

    def get_default_kwargs(self, default_kwargs, **kwargs):
        n = kwargs[self.dim_names[0]]
        var = kwargs['var']
        mean_mu = np.zeros((n))
        var_col_mu = var

        default_kwargs[self._mean_name] = mean_mu
        default_kwargs[self._var_col_name] = var_col_mu
        return

class VectorPrecondHelper(PrecondHelper):
    def __init__(self, name='mu', dim_names=None, var_row_name='Q'):
        self.name = name
        self._var_row_name = var_row_name
        self.dim_names = ['n'] if dim_names is None else dim_names
        return

    def precondition(self, preconditioner,
            precond_grad, grad, parameters, **kwargs):
        Q = getattr(parameters, self._var_row_name)
        precond_grad[self.name] = np.dot(Q, grad[self.name])
        return

    def precondition_noise(self, preconditioner,
            noise, parameters, **kwargs):
        LQinv = getattr(parameters, "L{}inv".format(self._var_row_name))
        noise[self.name] = np.linalg.solve(LQinv.T,
                np.random.normal(loc=0, size=(LQinv.shape[0]))
                )
        return

    def correction_term(self, preconditioner, correction, parameters, **kwargs):
        correction[self.name] = np.zeros_like(getattr(parameters, self.name),
                dtype=float)
        return

# Multiple Square
class VectorsParamHelper(ParamHelper):
    def __init__(self, name='mu', dim_names=None):
        self.name = name
        self.dim_names = ['n', 'num_states'] if dim_names is None else dim_names
        return

    def set_var(self, param, **kwargs):
        if self.name in kwargs:
            num_states, n = np.shape(kwargs[self.name])
            param.var_dict[self.name] = np.array(kwargs[self.name]).astype(float)
            param._set_check_dim(**{self.dim_names[0]: n,
                                    self.dim_names[1]: num_states})
        else:
            raise ValueError("{} not provided".format(self.name))
        return

    def project_parameters(self, param, **kwargs):
        name_kwargs = kwargs.get(self.name, {})
        if name_kwargs.get('fixed') is not None:
            param.var_dict[self.name] = name_kwargs['fixed'].copy()
        return

    def from_dict_to_vector(self, vector_list, var_dict, **kwargs):
        vector_list.append(var_dict[self.name].flatten())
        return

    def from_vector_to_dict(self, var_dict, vector, vector_index, **kwargs):
        n = kwargs[self.dim_names[0]]
        num_states = kwargs[self.dim_names[1]]
        mu = np.reshape(vector[vector_index:vector_index+num_states*n],
                (num_states, n))
        var_dict[self.name] = mu
        return vector_index+num_states*n

    def get_properties(self):
        properties = {}
        properties[self.name] = property(
                fget=get_value_func(self.name),
                fset=set_value_func(self.name),
                doc="{0} is {2} {1} vectors".format(
                    self.name, self.dim_names[0], self.dim_names[1]),
                )
        for dim_name in self.dim_names:
            properties[dim_name] = property(
                    fget=get_dim_func(dim_name),
                    )
        return properties

class VectorsPriorHelper(PriorHelper):
    def __init__(self, name='mu', dim_names=None, var_row_name=None):
        self.name = name
        self._mean_name = 'mean_{0}'.format(name) # num_states by n
        self._var_col_name = 'var_col_{0}'.format(name) # num_states by n
        self._var_row_name = var_row_name
        self._lt_vec_name = 'L{0}inv_vec'.format(var_row_name)
        self.dim_names = ['n', 'num_states'] if dim_names is None else dim_names
        return

    def set_hyperparams(self, prior, **kwargs):
        if self._mean_name in kwargs:
            num_states, n = np.shape(kwargs[self._mean_name])
        else:
            raise ValueError("{} must be provided".format(self._mean_name))
        if self._var_col_name in kwargs:
            num_states2 = np.size(kwargs[self._var_col_name])
        else:
            raise ValueError("{} must be provided".format(self._var_col_name))

        if (num_states != num_states2):
            raise ValueError("prior dimensions don't match")

        prior._set_check_dim(**{self.dim_names[0]: n,
                                self.dim_names[1]: num_states})
        prior.hyperparams[self._mean_name] = kwargs[self._mean_name]
        prior.hyperparams[self._var_col_name] = kwargs[self._var_col_name]
        return

    def sample_prior(self, prior, var_dict, **kwargs):
        mean_mu = prior.hyperparams[self._mean_name]
        var_col_mu = prior.hyperparams[self._var_col_name]
        if self._var_row_name is not None:
            if self._lt_vec_name in var_dict:
                LQinvs = np.array([tril_vector_to_mat(LQinv_vec_k)
                    for LQinv_vec_k in var_dict[self._lt_vec_name]])
                Qinvs = np.array([LQinv_k.dot(LQinv_k.T) + \
                        1e-9*np.eye(prior.dim[self.dim_names[0]])
                        for LQinv_k in LQinvs])
            else:
                raise ValueError("Missing {}\n".format(self._lt_vec_name) +
                    "Perhaps {} must be earlier in _prior_helper_list".format(
                        self._var_row_name)
                        )
        else:
            Qinvs = np.array([np.eye(prior.dim[self.dim_names[0]])
                for _ in prior.dim[self.dim_names[1]]])

        mus = [None for k in range(prior.dim[self.dim_names[1]])]
        for k in range(len(mus)):
            mus[k] = np.random.multivariate_normal(
                mean=mean_mu[k],
                cov=var_col_mu[k]*pos_def_mat_inv(Qinvs[k]),
                )
        var_dict[self.name] = np.array(mus)
        return

    def sample_posterior(self, prior, var_dict, sufficient_stat, **kwargs):
        mean_mu = prior.hyperparams[self._mean_name]
        var_col_mu = prior.hyperparams[self._var_col_name]
        num_states, n = np.shape(mean_mu)
        if self._var_row_name is not None:
            if self._lt_vec_name in var_dict:
                LQinvs = np.array([tril_vector_to_mat(LQinv_vec_k)
                    for LQinv_vec_k in var_dict[self._lt_vec_name]])
                Qinvs = np.array([LQinv_k.dot(LQinv_k.T) + 1e-9*np.eye(n)
                        for LQinv_k in LQinvs])
            else:
                raise ValueError("Missing {}\n".format(self._lt_vec_name) +
                    "Perhaps {} must be earlier in _prior_helper_list".format(
                        self._var_row_name)
                    )
        else:
            Qinvs = np.array([np.eye(n) for _ in range(num_states)])

        mus = [None for k in range(num_states)]
        for k in range(len(mus)):
            S_prevprev = var_col_mu[k]**-1 + \
                    sufficient_stat[self.name]['S_prevprev'][k]
            S_curprev = mean_mu[k] * var_col_mu[k]**-1  + \
                    sufficient_stat[self.name]['S_curprev'][k]
            post_mean_mu_k = S_curprev/S_prevprev
            mus[k] = np.random.multivariate_normal(
                    mean=post_mean_mu_k,
                    cov=pos_def_mat_inv(Qinvs[k])/S_prevprev,
                    )
        var_dict[self.name] = np.array(mus)
        return

    def logprior(self, prior, logprior, parameters, **kwargs):
        mean_mu = prior.hyperparams[self._mean_name]
        var_col_mu = prior.hyperparams[self._var_col_name]
        num_states, n = np.shape(mean_mu)
        if self._var_row_name is not None:
            LQinvs = np.array([tril_vector_to_mat(LQinv_vec_k)
                for LQinv_vec_k in parameters.var_dict[self._lt_vec_name]])
        else:
            LQinvs = np.array([np.eye(n)
                for _ in range(num_states)])
        for mu_k, mean_mu_k, var_col_mu_k, LQinv_k in zip(
                parameters.var_dict[self.name], mean_mu, var_col_mu, LQinvs):
            logprior += normal_logpdf(mu_k,
                    mean=mean_mu_k,
                    Lprec=var_col_mu_k**-0.5 * LQinv_k,
                    )
        return logprior

    def grad_logprior(self, prior, grad, parameters, **kwargs):
        mu = parameters.var_dict[self.name]
        mean_mu = prior.hyperparams[self._mean_name]
        var_col_mu = prior.hyperparams[self._var_col_name]
        num_states, n = np.shape(mean_mu)
        if self._var_row_name is not None:
            Qinvs = getattr(parameters, '{}inv'.format(self._var_row_name))
        else:
            Qinvs = np.array([np.eye(n)
                for _ in range(num_states)])

        grad[self.name] = np.array([
            -1.0 * np.dot(var_col_mu[k]**-1 * Qinvs[k], mu[k] - mean_mu[k])
            for k in range(num_states)])
        return

    def get_prior_kwargs(self, prior_kwargs, parameters, **kwargs):
        var = kwargs['var']
        mu = getattr(parameters, self.name)
        if kwargs.get('from_mean', False):
            mean_mu = mu.copy()
        else:
            mean_mu = np.zeros_like(mu)
        var_col_mu = np.array([
            var for _ in range(A.shape[0])
            ])

        prior_kwargs[self._mean_name] = mean_mu
        prior_kwargs[self._var_col_name] = var_col_mu
        return

    def get_default_kwargs(self, default_kwargs, **kwargs):
        n = kwargs[self.dim_names[0]]
        num_states = kwargs[self.dim_names[1]]
        var = kwargs['var']
        mean_mu = np.zeros((num_states, n))
        var_col_mu = np.ones((num_states))*var

        default_kwargs[self._mean_name] = mean_mu
        default_kwargs[self._var_col_name] = var_col_mu
        return

class VectorsPrecondHelper(PrecondHelper):
    def __init__(self, name='mu', dim_names=None, var_row_name='Q'):
        self.name = name
        self._var_row_name = var_row_name
        self.dim_names = ['n', 'num_states'] if dim_names is None else dim_names
        return

    def precondition(self, preconditioner,
            precond_grad, grad, parameters, **kwargs):
        Q = getattr(parameters, self._var_row_name)
        precond_grad[self.name] = np.array([
            np.dot(Q[k], grad[self.name][k])
            for k in range(Q.shape[0])
            ])
        return

    def precondition_noise(self, preconditioner,
            noise, parameters, **kwargs):
        LQinv = getattr(parameters, "L{}inv".format(self._var_row_name))
        noise[self.name] = np.array([
            np.linalg.solve(LQinv[k].T,
                np.random.normal(loc=0, size=LQinv.shape[-1])
                )
            for k in range(LQinv.shape[0])
            ])
        return

    def correction_term(self, preconditioner, correction, parameters, **kwargs):
        correction[self.name] = np.zeros_like(getattr(parameters, self.name),
                dtype=float)
        return


# Single Square
class SquareMatrixParamHelper(ParamHelper):
    def __init__(self, name='A', dim_names=None):
        self.name = name
        self.dim_names = ['n'] if dim_names is None else dim_names
        return

    def set_var(self, param, **kwargs):
        if self.name in kwargs:
            n, n2 = np.shape(kwargs[self.name])
            if n != n2:
                raise ValueError("{} must be square matrices".format(self.name))
            param.var_dict[self.name] = np.array(kwargs[self.name]).astype(float)
            param._set_check_dim(**{self.dim_names[0]: n})
        else:
            raise ValueError("{} not provided".format(self.name))
        return

    def project_parameters(self, param, **kwargs):
        name_kwargs = kwargs.get(self.name, {})
        if name_kwargs.get('thresh', True):
            A = param.var_dict[self.name]
            A = varp_stability_projection(A,
                    eigenvalue_cutoff=name_kwargs.get(
                        'eigenvalue_cutoff', 0.9999),
                    var_name=self.name,
                    logger=logger)
            param.var_dict[self.name] = A
        if name_kwargs.get('fixed') is not None:
            param.var_dict[self.name] = name_kwargs['fixed'].copy()
        return

    def from_dict_to_vector(self, vector_list, var_dict, **kwargs):
        vector_list.append(var_dict[self.name].flatten())
        return

    def from_vector_to_dict(self, var_dict, vector, vector_index, **kwargs):
        n = kwargs[self.dim_names[0]]
        A = np.reshape(vector[vector_index:vector_index+n**2], (n, n))
        var_dict[self.name] = A
        return vector_index+n**2

    def get_properties(self):
        properties = {}
        properties[self.name] = property(
                fget=get_value_func(self.name),
                fset=set_value_func(self.name),
                doc="{0} is a {1} by {1} matrix".format(
                    self.name, self.dim_names[0]),
                )
        for dim_name in self.dim_names:
            properties[dim_name] = property(
                    fget=get_dim_func(dim_name),
                    )
        return properties

class SquareMatrixPriorHelper(PriorHelper):
    def __init__(self, name='A', dim_names=None, var_row_name=None):
        self.name = name
        self._mean_name = 'mean_{0}'.format(name)
        self._var_col_name = 'var_col_{0}'.format(name)
        self._var_row_name = var_row_name
        self._lt_vec_name = 'L{0}inv_vec'.format(var_row_name)
        self.dim_names = ['n'] if dim_names is None else dim_names
        return

    def set_hyperparams(self, prior, **kwargs):
        if self._mean_name in kwargs:
            n, n2 = np.shape(kwargs[self._mean_name])
        else:
            raise ValueError("{} must be provided".format(self._mean_name))
        if self._var_col_name in kwargs:
            n3 = np.size(kwargs[self._var_col_name])
        else:
            raise ValueError("{} must be provided".format(self._var_col_name))

        if n != n2:
            raise ValueError("{} must be square".format(self._mean_name))
        if n != n3:
            raise ValueError("prior dimensions don't match")

        prior._set_check_dim(**{self.dim_names[0]: n})
        prior.hyperparams[self._mean_name] = kwargs[self._mean_name]
        prior.hyperparams[self._var_col_name] = kwargs[self._var_col_name]
        return

    def sample_prior(self, prior, var_dict, **kwargs):
        mean_A = prior.hyperparams[self._mean_name]
        var_col_A = prior.hyperparams[self._var_col_name]
        if self._var_row_name is not None:
            if self._lt_vec_name in var_dict:
                LQinv = tril_vector_to_mat(var_dict[self._lt_vec_name])
                Qinv = LQinv.dot(LQinv.T) + \
                        1e-9*np.eye(prior.dim[self.dim_names[0]])
            else:
                raise ValueError("Missing {}\n".format(self._lt_vec_name) +
                    "Perhaps {} must be earlier in _prior_helper_list".format(
                        self._var_row_name)
                        )
        else:
            Qinv = np.eye(prior.dim[self.dim_names[0]])

        var_dict[self.name] =  scipy.stats.matrix_normal(
                mean=mean_A,
                rowcov=pos_def_mat_inv(Qinv),
                colcov=np.diag(var_col_A),
                ).rvs()
        return

    def sample_posterior(self, prior, var_dict, sufficient_stat, **kwargs):
        mean_A = prior.hyperparams[self._mean_name]
        var_col_A = prior.hyperparams[self._var_col_name]
        if self._var_row_name is not None:
            if self._lt_vec_name in var_dict:
                LQinv = tril_vector_to_mat(var_dict[self._lt_vec_name])
                Qinv = LQinv.dot(LQinv.T) + \
                        1e-9*np.eye(prior.dim[self.dim_names[0]])
            else:
                raise ValueError("Missing {}\n".format(self._lt_vec_name) +
                    "Perhaps {} must be earlier in _prior_helper_list".format(
                        self._var_row_name)
                    )
        else:
            Qinv = np.eye(self.prior[self.dim_names[0]])

        S_prevprev = np.diag(var_col_A**-1) + \
                sufficient_stat[self.name]['S_prevprev']
        S_curprev = mean_A * var_col_A**-1  + \
                sufficient_stat[self.name]['S_curprev']
        var_dict[self.name] = scipy.stats.matrix_normal(
                mean=np.linalg.solve(S_prevprev, S_curprev.T).T,
                rowcov=pos_def_mat_inv(Qinv),
                colcov=pos_def_mat_inv(S_prevprev),
                ).rvs()
        return

    def logprior(self, prior, logprior, parameters, **kwargs):
        mean_A = prior.hyperparams[self._mean_name]
        var_col_A = prior.hyperparams[self._var_col_name]
        if self._var_row_name is not None:
            LQinv = tril_vector_to_mat(parameters.var_dict[self._lt_vec_name])
        else:
            LQinv = np.eye(prior.dim[self.dim_names[0]])
        logprior += matrix_normal_logpdf(parameters.var_dict[self.name],
                mean=mean_A,
                Lrowprec=LQinv,
                Lcolprec=np.diag(var_col_A**-0.5),
                )
        return logprior

    def grad_logprior(self, prior, grad, parameters, **kwargs):
        mean_A = prior.hyperparams[self._mean_name]
        var_col_A = prior.hyperparams[self._var_col_name]
        A = getattr(parameters, self.name)
        if self._var_row_name is not None:
            Qinv = getattr(parameters, '{}inv'.format(self._var_row_name))
        else:
            Qinv = np.eye(prior.dim[self.dim_names[0]])

        grad[self.name] = -1.0 * np.dot(Qinv, A - mean_A) * var_col_A**-1
        return

    def get_prior_kwargs(self, prior_kwargs, parameters, **kwargs):
        var = kwargs['var']
        A = getattr(parameters, self.name)
        if kwargs.get('from_mean', False):
            mean_A = A.copy()
        else:
            mean_A = np.zeros_like(A)
        var_col_A = np.ones(A.shape[0])*var

        prior_kwargs[self._mean_name] = mean_A
        prior_kwargs[self._var_col_name] = var_col_A
        return

    def get_default_kwargs(self, default_kwargs, **kwargs):
        n = kwargs[self.dim_names[0]]
        var = kwargs['var']
        mean_A = np.zeros((n,n))
        var_col_A = np.ones(n)*var

        default_kwargs[self._mean_name] = mean_A
        default_kwargs[self._var_col_name] = var_col_A
        return

class SquareMatrixPrecondHelper(PrecondHelper):
    def __init__(self, name='A', dim_names=None, var_row_name='Q'):
        self.name = name
        self._var_row_name = var_row_name
        self.dim_names = ['n'] if dim_names is None else dim_names
        return

    def precondition(self, preconditioner,
            precond_grad, grad, parameters, **kwargs):
        Q = getattr(parameters, self._var_row_name)
        precond_grad[self.name] = np.dot(Q, grad[self.name])
        return

    def precondition_noise(self, preconditioner,
            noise, parameters, **kwargs):
        LQinv = getattr(parameters, "L{}inv".format(self._var_row_name))
        noise[self.name] = np.linalg.solve(LQinv.T,
                np.random.normal(loc=0, size=LQinv.shape)
                )
        return

    def correction_term(self, preconditioner, correction, parameters, **kwargs):
        correction[self.name] = np.zeros_like(getattr(parameters, self.name),
                dtype=float)
        return

# Multiple Square
class SquareMatricesParamHelper(ParamHelper):
    def __init__(self, name='A', dim_names=None):
        self.name = name
        self.dim_names = ['n', 'num_states'] if dim_names is None else dim_names
        return

    def set_var(self, param, **kwargs):
        if self.name in kwargs:
            num_states, n, n2 = np.shape(kwargs[self.name])
            if n != n2:
                raise ValueError("{} must be square matrices".format(self.name))
            param.var_dict[self.name] = np.array(kwargs[self.name]).astype(float)
            param._set_check_dim(**{self.dim_names[0]: n,
                                    self.dim_names[1]: num_states})
        else:
            raise ValueError("{} not provided".format(self.name))
        return

    def project_parameters(self, param, **kwargs):
        name_kwargs = kwargs.get(self.name, {})
        if name_kwargs.get('thresh', True):
            A = param.var_dict[self.name]
            for k, A_k in enumerate(A):
                A_k = varp_stability_projection(A_k,
                        eigenvalue_cutoff=name_kwargs.get(
                            'eigenvalue_cutoff', 0.9999),
                        var_name=self.name,
                        logger=logger)
                A[k] = A_k
            param.var_dict[self.name] = A
        if name_kwargs.get('fixed') is not None:
            param.var_dict[self.name] = name_kwargs['fixed'].copy()
        return

    def from_dict_to_vector(self, vector_list, var_dict, **kwargs):
        vector_list.append(var_dict[self.name].flatten())
        return

    def from_vector_to_dict(self, var_dict, vector, vector_index, **kwargs):
        n = kwargs[self.dim_names[0]]
        num_states = kwargs[self.dim_names[1]]
        A = np.reshape(vector[vector_index:vector_index+num_states*n**2],
                (num_states, n, n))
        var_dict[self.name] = A
        return vector_index+num_states*n**2

    def get_properties(self):
        properties = {}
        properties[self.name] = property(
                fget=get_value_func(self.name),
                fset=set_value_func(self.name),
                doc="{0} is a {2} of {1} by {1} matrices".format(
                    self.name, self.dim_names[0], self.dim_names[1]),
                )
        for dim_name in self.dim_names:
            properties[dim_name] = property(
                    fget=get_dim_func(dim_name),
                    )
        return properties

class SquareMatricesPriorHelper(PriorHelper):
    def __init__(self, name='A', dim_names=None, var_row_name=None):
        self.name = name
        self._mean_name = 'mean_{0}'.format(name)
        self._var_col_name = 'var_col_{0}'.format(name)
        self._var_row_name = var_row_name
        self._lt_vec_name = 'L{0}inv_vec'.format(var_row_name)
        self.dim_names = ['n', 'num_states'] if dim_names is None else dim_names
        return

    def set_hyperparams(self, prior, **kwargs):
        if self._mean_name in kwargs:
            num_states, n, n2 = np.shape(kwargs[self._mean_name])
        else:
            raise ValueError("{} must be provided".format(self._mean_name))
        if self._var_col_name in kwargs:
            num_states2, n3 = np.shape(kwargs[self._var_col_name])
        else:
            raise ValueError("{} must be provided".format(self._var_col_name))

        if n != n2:
            raise ValueError("{} must be square".format(self._mean_name))
        if (n != n3) or (num_states != num_states2):
            raise ValueError("prior dimensions don't match")

        prior._set_check_dim(**{self.dim_names[0]: n,
                                self.dim_names[1]: num_states})
        prior.hyperparams[self._mean_name] = kwargs[self._mean_name]
        prior.hyperparams[self._var_col_name] = kwargs[self._var_col_name]
        return

    def sample_prior(self, prior, var_dict, **kwargs):
        n = prior.dim[self.dim_names[0]]
        num_states = prior.dim[self.dim_names[1]]
        mean_A = prior.hyperparams[self._mean_name]
        var_col_A = prior.hyperparams[self._var_col_name]
        if self._var_row_name is not None:
            if self._lt_vec_name in var_dict:
                LQinvs = np.array([tril_vector_to_mat(LQinv_vec_k)
                    for LQinv_vec_k in var_dict[self._lt_vec_name]])
                Qinvs = np.array([LQinv_k.dot(LQinv_k.T) + 1e-9*np.eye(n)
                        for LQinv_k in LQinvs])
            else:
                raise ValueError("Missing {}\n".format(self._lt_vec_name) +
                    "Perhaps {} must be earlier in _prior_helper_list".format(
                        self._var_row_name)
                        )
        else:
            Qinvs = np.array([np.eye(n) for _ in range(num_states)])

        As = [None for k in range(num_states)]
        for k in range(len(As)):
            As[k] = scipy.stats.matrix_normal(
                mean=mean_A[k],
                rowcov=pos_def_mat_inv(Qinvs[k]),
                colcov=np.diag(var_col_A[k]),
                ).rvs()
        var_dict[self.name] = np.array(As)
        return

    def sample_posterior(self, prior, var_dict, sufficient_stat, **kwargs):
        n = prior.dim[self.dim_names[0]]
        num_states = prior.dim[self.dim_names[1]]
        mean_A = prior.hyperparams[self._mean_name]
        var_col_A = prior.hyperparams[self._var_col_name]
        if self._var_row_name is not None:
            if self._lt_vec_name in var_dict:
                LQinvs = np.array([tril_vector_to_mat(LQinv_vec_k)
                    for LQinv_vec_k in var_dict[self._lt_vec_name]])
                Qinvs = np.array([LQinv_k.dot(LQinv_k.T) + 1e-9*np.eye(n)
                        for LQinv_k in LQinvs])
            else:
                raise ValueError("Missing {}\n".format(self._lt_vec_name) +
                    "Perhaps {} must be earlier in _prior_helper_list".format(
                        self._var_row_name)
                    )
        else:
            Qinvs = np.array([np.eye(n) for _ in range(num_states)])

        As = [None for k in range(num_states)]
        for k in range(len(As)):
            S_prevprev = np.diag(var_col_A[k]**-1) + \
                    sufficient_stat[self.name]['S_prevprev'][k]
            S_curprev = mean_A[k] * var_col_A[k]**-1  + \
                    sufficient_stat[self.name]['S_curprev'][k]
            As[k] = scipy.stats.matrix_normal(
                    mean=np.linalg.solve(S_prevprev, S_curprev.T).T,
                    rowcov=pos_def_mat_inv(Qinvs[k]),
                    colcov=pos_def_mat_inv(S_prevprev),
                    ).rvs()
        var_dict[self.name] = np.array(As)
        return

    def logprior(self, prior, logprior, parameters, **kwargs):
        n = prior.dim[self.dim_names[0]]
        num_states = prior.dim[self.dim_names[1]]
        mean_A = prior.hyperparams[self._mean_name]
        var_col_A = prior.hyperparams[self._var_col_name]
        if self._var_row_name is not None:
            LQinv_vec = getattr(parameters, self._lt_vec_name)
            LQinvs = np.array([tril_vector_to_mat(LQinv_vec_k)
                for LQinv_vec_k in LQinv_vec])
        else:
            LQinvs = np.array([np.eye(n) for _ in range(num_states)])
        for A_k, mean_A_k, var_col_A_k, LQinv_k in zip(
                parameters.var_dict[self.name], mean_A, var_col_A, LQinvs):
            logprior += matrix_normal_logpdf(A_k,
                    mean=mean_A_k,
                    Lrowprec=LQinv_k,
                    Lcolprec=np.diag(var_col_A_k**-0.5),
                    )
        return logprior

    def grad_logprior(self, prior, grad, parameters, **kwargs):
        mean_A = prior.hyperparams[self._mean_name]
        var_col_A = prior.hyperparams[self._var_col_name]
        A = getattr(parameters, self.name)
        if self._var_row_name is not None:
            Qinvs = getattr(parameters, '{}inv'.format(self._var_row_name))
        else:
            Qinvs = np.array([np.eye(prior.dim[self.dim_names[0]])
                for _ in prior.dim[self.dim_names[1]]])

        grad[self.name] = np.array([
            -1.0 * np.dot(Qinvs[k], A[k] - mean_A[k]) * var_col_A[k]**-1
            for k in range(prior.dim[self.dim_names[1]])
            ])
        return

    def get_prior_kwargs(self, prior_kwargs, parameters, **kwargs):
        var = kwargs['var']
        A = getattr(parameters, self.name)
        if kwargs.get('from_mean', False):
            mean_A = A.copy()
        else:
            mean_A = np.zeros_like(A)
        var_col_A = np.array([
            np.ones(A.shape[0])*var for _ in range(A.shape[0])
            ])

        prior_kwargs[self._mean_name] = mean_A
        prior_kwargs[self._var_col_name] = var_col_A
        return

    def get_default_kwargs(self, default_kwargs, **kwargs):
        n = kwargs[self.dim_names[0]]
        num_states = kwargs[self.dim_names[1]]
        var = kwargs['var']
        mean_A = np.zeros((num_states, n,n))
        var_col_A = np.ones((num_states,n))*var

        default_kwargs[self._mean_name] = mean_A
        default_kwargs[self._var_col_name] = var_col_A
        return

class SquareMatricesPrecondHelper(PrecondHelper):
    def __init__(self, name='A', dim_names=None, var_row_name='Q'):
        self.name = name
        self._var_row_name = var_row_name
        self.dim_names = ['n', 'num_states'] if dim_names is None else dim_names
        return

    def precondition(self, preconditioner,
            precond_grad, grad, parameters, **kwargs):
        Q = getattr(parameters, self._var_row_name)
        precond_grad[self.name] = np.array([
            np.dot(Q[k], grad[self.name][k])
            for k in range(Q.shape[0])
            ])
        return

    def precondition_noise(self, preconditioner,
            noise, parameters, **kwargs):
        LQinv = getattr(parameters, "L{}inv".format(self._var_row_name))
        noise[self.name] = np.array([
            np.linalg.solve(LQinv[k].T,
                np.random.normal(loc=0, size=LQinv[k].shape)
                )
            for k in range(LQinv.shape[0])
            ])
        return

    def correction_term(self, preconditioner, correction, parameters, **kwargs):
        correction[self.name] = np.zeros_like(getattr(parameters, self.name),
                dtype=float)
        return

# Single Rectangular (m by n)
class RectMatrixParamHelper(ParamHelper):
    def __init__(self, name='A', dim_names=None):
        self.name = name
        self.dim_names = ['m','n'] if dim_names is None else dim_names
        return

    def set_var(self, param, **kwargs):
        if self.name in kwargs:
            m, n = np.shape(kwargs[self.name])
            param.var_dict[self.name] = np.array(kwargs[self.name]).astype(float)
            param._set_check_dim(**{
                self.dim_names[0]: m,
                self.dim_names[1]: n,
                })
        else:
            raise ValueError("{} not provided".format(self.name))
        return

    def project_parameters(self, param, **kwargs):
        name_kwargs = kwargs.get(self.name, {})
        if name_kwargs.get('thresh', False):
            A = param.var_dict[self.name]
            A = varp_stability_projection(A,
                    eigenvalue_cutoff=name_kwargs.get(
                        'eigenvalue_cutoff', 0.9999),
                    var_name=self.name,
                    logger=logger)
            param.var_dict[self.name] = A
        if name_kwargs.get('fixed') is not None:
            param.var_dict[self.name] = name_kwargs['fixed'].copy()
        if name_kwargs.get('fixed_eye', False):
            k = min(param.dim[self.dim_names[0]], param.dim[self.dim_names[1]])
            A = param.var_dict[self.name]
            A[0:k, 0:k] = np.eye(k)
            param.var_dict[self.name] = A
        return

    def from_dict_to_vector(self, vector_list, var_dict, **kwargs):
        vector_list.append(var_dict[self.name].flatten())
        return

    def from_vector_to_dict(self, var_dict, vector, vector_index, **kwargs):
        m = kwargs[self.dim_names[0]]
        n = kwargs[self.dim_names[1]]
        A = np.reshape(vector[vector_index:vector_index+m*n], (m, n))
        var_dict[self.name] = A
        return vector_index+m*n

    def get_properties(self):
        properties = {}
        properties[self.name] = property(
                fget=get_value_func(self.name),
                fset=set_value_func(self.name),
                doc="{0} is a {1} by {2} matrix".format(
                    self.name, self.dim_names[0], self.dim_names[1]),
                )
        for dim_name in self.dim_names:
            properties[dim_name] = property(
                    fget=get_dim_func(dim_name),
                    )
        return properties

class RectMatrixPriorHelper(PriorHelper):
    def __init__(self, name='A', dim_names=None, var_row_name=None):
        self.name = name
        self._mean_name = 'mean_{0}'.format(name) # m by n ndarray
        self._var_col_name = 'var_col_{0}'.format(name) # n ndarray
        self._var_row_name = var_row_name
        self._lt_vec_name = 'L{0}inv_vec'.format(var_row_name) # m by m ndarray
        self.dim_names = ['m', 'n'] if dim_names is None else dim_names
        return

    def set_hyperparams(self, prior, **kwargs):
        if self._mean_name in kwargs:
            m, n = np.shape(kwargs[self._mean_name])
        else:
            raise ValueError("{} must be provided".format(self._mean_name))
        if self._var_col_name in kwargs:
            n2 = np.size(kwargs[self._var_col_name])
        else:
            raise ValueError("{} must be provided".format(self._var_col_name))
        if n != n2:
            raise ValueError("prior dimensions don't match")

        prior._set_check_dim(**{
            self.dim_names[0]: m,
            self.dim_names[1]: n,
            })
        prior.hyperparams[self._mean_name] = kwargs[self._mean_name]
        prior.hyperparams[self._var_col_name] = kwargs[self._var_col_name]
        return

    def sample_prior(self, prior, var_dict, **kwargs):
        mean_A = prior.hyperparams[self._mean_name]
        var_col_A = prior.hyperparams[self._var_col_name]
        if self._var_row_name is not None:
            if self._lt_vec_name in var_dict:
                LQinv = tril_vector_to_mat(var_dict[self._lt_vec_name])
                Qinv = LQinv.dot(LQinv.T) + \
                        1e-9*np.eye(prior.dim[self.dim_names[0]])
            else:
                raise ValueError("Missing {}\n".format(self._lt_vec_name) +
                    "Perhaps {} must be earlier in _prior_helper_list".format(
                        self._var_row_name)
                        )
        else:
            Qinv = np.eye(prior.dim[self.dim_names[0]])

        var_dict[self.name] =  scipy.stats.matrix_normal(
                mean=mean_A,
                rowcov=pos_def_mat_inv(Qinv),
                colcov=np.diag(var_col_A),
                ).rvs()
        return

    def sample_posterior(self, prior, var_dict, sufficient_stat, **kwargs):
        mean_A = prior.hyperparams[self._mean_name]
        var_col_A = prior.hyperparams[self._var_col_name]
        if self._var_row_name is not None:
            if self._lt_vec_name in var_dict:
                LQinv = tril_vector_to_mat(var_dict[self._lt_vec_name])
                Qinv = LQinv.dot(LQinv.T) + \
                        1e-9*np.eye(prior.dim[self.dim_names[0]])
            else:
                raise ValueError("Missing {}\n".format(self._lt_vec_name) +
                    "Perhaps {} must be earlier in _prior_helper_list".format(
                        self._var_row_name)
                    )
        else:
            Qinv = np.eye(self.prior[self.dim_names[0]])

        S_prevprev = np.diag(var_col_A**-1) + \
                sufficient_stat[self.name]['S_prevprev']
        S_curprev = mean_A * var_col_A**-1  + \
                sufficient_stat[self.name]['S_curprev']
        var_dict[self.name] = scipy.stats.matrix_normal(
                mean=np.linalg.solve(S_prevprev, S_curprev.T).T,
                rowcov=pos_def_mat_inv(Qinv),
                colcov=pos_def_mat_inv(S_prevprev),
                ).rvs()
        return

    def logprior(self, prior, logprior, parameters, **kwargs):
        mean_A = prior.hyperparams[self._mean_name]
        var_col_A = prior.hyperparams[self._var_col_name]
        if self._var_row_name is not None:
            LQinv = tril_vector_to_mat(parameters.var_dict[self._lt_vec_name])
        else:
            LQinv = np.eye(prior.dim[self.dim_names[0]])
        logprior += matrix_normal_logpdf(parameters.var_dict[self.name],
                mean=mean_A,
                Lrowprec=LQinv,
                Lcolprec=np.diag(var_col_A**-0.5),
                )
        return logprior

    def grad_logprior(self, prior, grad, parameters, **kwargs):
        mean_A = prior.hyperparams[self._mean_name]
        var_col_A = prior.hyperparams[self._var_col_name]
        A = getattr(parameters, self.name)
        if self._var_row_name is not None:
            Qinv = getattr(parameters, '{}inv'.format(self._var_row_name))
        else:
            Qinv = np.eye(prior.dim[self.dim_names[0]])

        grad[self.name] = -1.0 * np.dot(Qinv, A - mean_A) * var_col_A**-1
        return

    def get_prior_kwargs(self, prior_kwargs, parameters, **kwargs):
        var = kwargs['var']
        A = getattr(parameters, self.name)
        if kwargs.get('from_mean', False):
            mean_A = A.copy()
        else:
            mean_A = np.zeros_like(A)
        var_col_A = np.ones(A.shape[1])*var

        prior_kwargs[self._mean_name] = mean_A
        prior_kwargs[self._var_col_name] = var_col_A
        return

    def get_default_kwargs(self, default_kwargs, **kwargs):
        m = kwargs[self.dim_names[0]]
        n = kwargs[self.dim_names[1]]
        var = kwargs['var']
        mean_A = np.zeros((m,n))
        var_col_A = np.ones(n)*var

        default_kwargs[self._mean_name] = mean_A
        default_kwargs[self._var_col_name] = var_col_A
        return

class RectMatrixPrecondHelper(PrecondHelper):
    def __init__(self, name='A', dim_names=None, var_row_name='Q'):
        self.name = name
        self._var_row_name = var_row_name
        self.dim_names = ['m', 'n'] if dim_names is None else dim_names
        return

    def precondition(self, preconditioner,
            precond_grad, grad, parameters, **kwargs):
        Q = getattr(parameters, self._var_row_name)
        precond_grad[self.name] = np.dot(Q, grad[self.name])
        return

    def precondition_noise(self, preconditioner,
            noise, parameters, **kwargs):
        m = parameters.dim[self.dim_names[0]]
        n = parameters.dim[self.dim_names[1]]
        LQinv = getattr(parameters, "L{}inv".format(self._var_row_name))
        noise[self.name] = np.linalg.solve(LQinv.T,
                np.random.normal(loc=0, size=(m, n))
                )
        return

    def correction_term(self, preconditioner, correction, parameters, **kwargs):
        correction[self.name] = np.zeros_like(getattr(parameters, self.name),
                dtype=float)
        return

# Multiple Rectangular
class RectMatricesParamHelper(ParamHelper):
    def __init__(self, name='A', dim_names=None):
        self.name = name
        self.dim_names = ['m', 'n', 'num_states'] \
                if dim_names is None else dim_names
        return

    def set_var(self, param, **kwargs):
        if self.name in kwargs:
            num_states, m, n = np.shape(kwargs[self.name])
            param.var_dict[self.name] = np.array(kwargs[self.name]).astype(float)
            param._set_check_dim(**{
                    self.dim_names[0]: m,
                    self.dim_names[1]: n,
                    self.dim_names[2]: num_states,
                    })
        else:
            raise ValueError("{} not provided".format(self.name))
        return

    def project_parameters(self, param, **kwargs):
        name_kwargs = kwargs.get(self.name, {})
        if name_kwargs.get('thresh', False):
            A = param.var_dict[self.name]
            for k, A_k in enumerate(A):
                A_k = varp_stability_projection(A_k,
                        eigenvalue_cutoff=name_kwargs.get(
                            'eigenvalue_cutoff', 0.9999),
                        var_name=self.name,
                        logger=logger)
                A[k] = A_k
            param.var_dict[self.name] = A
        if name_kwargs.get('fixed') is not None:
            param.var_dict[self.name] = name_kwargs['fixed'].copy()
        if name_kwargs.get('fixed_eye', False):
            k = min(param.dim[self.dim_names[0]], param.dim[self.dim_names[1]])
            A = param.var_dict[self.name]
            for kk in range(self.num_states):
                A[kk, 0:k, 0:k] = np.eye(k)
            param.var_dict[self.name] = A

        return

    def from_dict_to_vector(self, vector_list, var_dict, **kwargs):
        vector_list.append(var_dict[self.name].flatten())
        return

    def from_vector_to_dict(self, var_dict, vector, vector_index, **kwargs):
        m = kwargs[self.dim_names[0]]
        n = kwargs[self.dim_names[1]]
        num_states = kwargs[self.dim_names[2]]
        A = np.reshape(vector[vector_index:vector_index+num_states*m*n],
                (num_states, m, n))
        var_dict[self.name] = A
        return vector_index+num_states*m*n

    def get_properties(self):
        properties = {}
        properties[self.name] = property(
                fget=get_value_func(self.name),
                fset=set_value_func(self.name),
                doc="{0} is a {3} by {1} by {2} matrices".format(
                    self.name, self.dim_names[0],
                    self.dim_names[1], self.dim_names[2]),
                )
        for dim_name in self.dim_names:
            properties[dim_name] = property(
                    fget=get_dim_func(dim_name),
                    )
        return properties

class RectMatricesPriorHelper(PriorHelper):
    def __init__(self, name='A', dim_names=None, var_row_name=None):
        self.name = name
        self._mean_name = 'mean_{0}'.format(name) # num_states x m x n
        self._var_col_name = 'var_col_{0}'.format(name) # num_states x n
        self._var_row_name = var_row_name # num_states x m x m
        self._lt_vec_name = 'L{0}inv_vec'.format(var_row_name)
        self.dim_names = ['m', 'n', 'num_states'] \
                if dim_names is None else dim_names
        return

    def set_hyperparams(self, prior, **kwargs):
        if self._mean_name in kwargs:
            num_states, m, n = np.shape(kwargs[self._mean_name])
        else:
            raise ValueError("{} must be provided".format(self._mean_name))
        if self._var_col_name in kwargs:
            num_states2, n2 = np.shape(kwargs[self._var_col_name])
        else:
            raise ValueError("{} must be provided".format(self._var_col_name))

        if (n != n2) or (num_states != num_states2):
            raise ValueError("prior dimensions don't match")

        prior._set_check_dim(**{
                self.dim_names[0]: m,
                self.dim_names[1]: n,
                self.dim_names[2]: num_states})
        prior.hyperparams[self._mean_name] = kwargs[self._mean_name]
        prior.hyperparams[self._var_col_name] = kwargs[self._var_col_name]
        return

    def sample_prior(self, prior, var_dict, **kwargs):
        mean_A = prior.hyperparams[self._mean_name]
        var_col_A = prior.hyperparams[self._var_col_name]
        num_states, m, n = np.shape(mean_A)
        if self._var_row_name is not None:
            if self._lt_vec_name in var_dict:
                LQinvs = np.array([tril_vector_to_mat(LQinv_vec_k)
                    for LQinv_vec_k in var_dict[self._lt_vec_name]])
                Qinvs = np.array([LQinv_k.dot(LQinv_k.T) + 1e-9*np.eye(m)
                        for LQinv_k in LQinvs])
            else:
                raise ValueError("Missing {}\n".format(self._lt_vec_name) +
                    "Perhaps {} must be earlier in _prior_helper_list".format(
                        self._var_row_name)
                        )
        else:
            Qinvs = np.array([np.eye(m) for _ in range(num_states)])

        As = [None for k in range(num_states)]
        for k in range(len(As)):
            As[k] = scipy.stats.matrix_normal(
                mean=mean_A[k],
                rowcov=pos_def_mat_inv(Qinvs[k]),
                colcov=np.diag(var_col_A[k]),
                ).rvs()
        var_dict[self.name] = np.array(As)
        return

    def sample_posterior(self, prior, var_dict, sufficient_stat, **kwargs):
        mean_A = prior.hyperparams[self._mean_name]
        var_col_A = prior.hyperparams[self._var_col_name]
        num_states, m, n = np.shape(mean_A)
        if self._var_row_name is not None:
            if self._lt_vec_name in var_dict:
                LQinvs = np.array([tril_vector_to_mat(LQinv_vec_k)
                    for LQinv_vec_k in var_dict[self._lt_vec_name]])
                Qinvs = np.array([LQinv_k.dot(LQinv_k.T) + 1e-9*np.eye(m)
                        for LQinv_k in LQinvs])
            else:
                raise ValueError("Missing {}\n".format(self._lt_vec_name) +
                    "Perhaps {} must be earlier in _prior_helper_list".format(
                        self._var_row_name)
                    )
        else:
            Qinvs = np.array([np.eye(m) for _ in range(num_states)])

        As = [None for k in range(num_states)]
        for k in range(len(As)):
            S_prevprev = np.diag(var_col_A[k]**-1) + \
                    sufficient_stat[self.name]['S_prevprev'][k]
            S_curprev = mean_A[k] * var_col_A[k]**-1  + \
                    sufficient_stat[self.name]['S_curprev'][k]
            As[k] = scipy.stats.matrix_normal(
                    mean=np.linalg.solve(S_prevprev, S_curprev.T).T,
                    rowcov=pos_def_mat_inv(Qinvs[k]),
                    colcov=pos_def_mat_inv(S_prevprev),
                    ).rvs()
        var_dict[self.name] = np.array(As)
        return

    def logprior(self, prior, logprior, parameters, **kwargs):
        mean_A = prior.hyperparams[self._mean_name]
        var_col_A = prior.hyperparams[self._var_col_name]
        num_states, m, n = np.shape(mean_A)
        if self._var_row_name is not None:
            LQinvs = np.array([tril_vector_to_mat(LQinv_vec_k)
                    for LQinv_vec_k in parameters.var_dict[self._lt_vec_name]])
        else:
            LQinvs = np.array([np.eye(m) for _ in range(num_states)])
        for A_k, mean_A_k, var_col_A_k, LQinv_k in zip(
                parameters.var_dict[self.name], mean_A, var_col_A, LQinvs):
            logprior += matrix_normal_logpdf(A_k,
                    mean=mean_A_k,
                    Lrowprec=LQinv_k,
                    Lcolprec=np.diag(var_col_A_k**-0.5),
                    )
        return logprior

    def grad_logprior(self, prior, grad, parameters, **kwargs):
        mean_A = prior.hyperparams[self._mean_name]
        var_col_A = prior.hyperparams[self._var_col_name]
        A = getattr(parameters, self.name)
        if self._var_row_name is not None:
            Qinvs = getattr(parameters, '{}inv'.format(self._var_row_name))
        else:
            Qinvs = np.array([np.eye(prior.dim[self.dim_names[0]])
                for _ in prior.dim[self.dim_names[2]]])

        grad[self.name] = np.array([
            -1.0 * np.dot(Qinvs[k], A[k] - mean_A[k]) * var_col_A[k]**-1
            for k in range(prior.dim[self.dim_names[2]])
            ])
        return

    def get_prior_kwargs(self, prior_kwargs, parameters, **kwargs):
        var = kwargs['var']
        A = getattr(parameters, self.name)
        if kwargs.get('from_mean', False):
            mean_A = A.copy()
        else:
            mean_A = np.zeros_like(A)
        var_col_A = np.array([
            np.ones(A.shape[2])*var for _ in range(A.shape[0])
            ])

        prior_kwargs[self._mean_name] = mean_A
        prior_kwargs[self._var_col_name] = var_col_A
        return

    def get_default_kwargs(self, default_kwargs, **kwargs):
        m = kwargs[self.dim_names[0]]
        n = kwargs[self.dim_names[1]]
        num_states = kwargs[self.dim_names[2]]
        var = kwargs['var']
        mean_A = np.zeros((num_states,m,n))
        var_col_A = np.ones((num_states,n))*var

        default_kwargs[self._mean_name] = mean_A
        default_kwargs[self._var_col_name] = var_col_A
        return

class RectMatricesPrecondHelper(PrecondHelper):
    def __init__(self, name='A', dim_names=None, var_row_name='Q'):
        self.name = name
        self._var_row_name = var_row_name
        self.dim_names = ['m', 'n', 'num_states'] \
                if dim_names is None else dim_names
        return

    def precondition(self, preconditioner,
            precond_grad, grad, parameters, **kwargs):
        Q = getattr(parameters, self._var_row_name)
        precond_grad[self.name] = np.array([
            np.dot(Q[k], grad[self.name][k])
            for k in range(Q.shape[0])
            ])
        return

    def precondition_noise(self, preconditioner,
            noise, parameters, **kwargs):
        m = parameters.dim[self.dim_names[0]]
        n = parameters.dim[self.dim_names[1]]
        LQinv = getattr(parameters, "L{}inv".format(self._var_row_name))
        noise[self.name] = np.array([
            np.linalg.solve(LQinv[k].T,
                np.random.normal(loc=0, size=(m,n))
                )
            for k in range(LQinv.shape[0])
            ])
        return

    def correction_term(self, preconditioner, correction, parameters, **kwargs):
        correction[self.name] = np.zeros_like(getattr(parameters, self.name),
                dtype=float)
        return


if __name__ == "__main__":
    # Demo of Parameters
    class SquareParameters(BaseParameters):
        """ Square Parameters """
        _param_helper_list = [
                SquareMatrixParamHelper(name='A', dim_names=['n'])
                ]
        for param_helper in _param_helper_list:
            properties = param_helper.get_properties()
            for name, prop in properties.items():
                vars()[name] = prop

        def __str__(self):
            my_str = "SquareParameters:"
            my_str += "\nA:\n" + str(self.A)
            return my_str

    class SquareMatrixPrior(BasePrior):
        """ Square Prior """
        _Parameters = SquareParameters
        _prior_helper_list = [
            SquareMatrixPriorHelper(name='A', dim_names=['n'], var_row_name=None)
                ]


