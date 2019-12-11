import numpy as np
import scipy.stats
import logging
from ..base_parameters import (
        ParamHelper, PriorHelper, PrecondHelper,
        get_value_func, get_hyperparam_func, get_dim_func,
        set_value_func, set_hyperparam_func,
        )
from .._utils import (
        array_wishart_rvs,
        pos_def_mat_inv,
        )
logger = logging.getLogger(name=__name__)

# Implementations of Covariance Parameters

# Single Covariance
class CovarianceParamHelper(ParamHelper):
    def __init__(self, name='Q', dim_names=None):
        self.name = name
        self._lt_prec_name = 'L{}inv'.format(name)
        self._inv_name = '{}inv'.format(name)
        self.dim_names = ['n'] if dim_names is None else dim_names
        return

    def set_var(self, param, **kwargs):
        if self._lt_prec_name in kwargs:
            n, n2 = np.shape(kwargs[self._lt_prec_name])
            if n != n2:
                raise ValueError("{} must be square matrix".format(
                    self._lt_prec_name))
            LQinv = np.array(kwargs[self._lt_prec_name]).astype(float)
            LQinv[np.triu_indices_from(LQinv, 1)] = 0.0 # zero out upper tri

            param.var_dict[self._lt_prec_name] = LQinv
            param._set_check_dim(**{self.dim_names[0]: n})
        else:
            raise ValueError("{} not provided".format(self._lt_prec_name))
        return

    def project_parameters(self, param, **kwargs):
        name_kwargs = kwargs.get(self.name, {})
        if name_kwargs.get('thresh', True):
            LQinv = param.var_dict[self._lt_prec_name]
            if np.any(np.diag(LQinv) < 0.0):
                logger.info(
                    "Reflecting {0}: {1} < 0.0".format(
                        self._lt_prec_name, LQinv)
                    )
                LQinv[:] = np.linalg.cholesky(
                    np.dot(LQinv, LQinv.T) + \
                            np.eye(param.dim[self.dim_names[0]])*1e-9
                    )
            param.var_dict[self._lt_prec_name] = LQinv

        if name_kwargs.get('fixed') is not None:
            param.var_dict[self._lt_prec_name] = name_kwargs['fixed'].copy()
        return

    def from_dict_to_vector(self, vector_list, var_dict, **kwargs):
        LQinv = var_dict[self._lt_prec_name]
        if np.isscalar(LQinv):
            vector_list.append([LQinv])
        else:
            vector_list.append(LQinv[np.tril_indices_from(LQinv)])
        return

    def from_vector_to_dict(self, var_dict, vector, vector_index, **kwargs):
        n = kwargs[self.dim_names[0]]
        LQinv = np.zeros((n, n))
        LQinv[np.tril_indices(n)] = vector[vector_index:vector_index+(n+1)*n//2]
        var_dict[self._lt_prec_name] = LQinv
        return vector_index+(n+1)*n//2

    def get_properties(self):
        properties = {}
        properties[self._lt_prec_name] = property(
                fget=get_value_func(self._lt_prec_name),
                fset=set_value_func(self._lt_prec_name),
                doc="{0} is a {1} by {1} lower triangular matrix".format(
                    self._lt_prec_name, self.dim_names[0]),
                )
        properties[self._inv_name] = property(
                fget=get_Qinv_func(self.name),
                doc="{0} is a {1} by {1} precision matrix".format(
                    self._inv_name, self.dim_names[0]),
                )
        properties[self.name] = property(
                fget=get_Q_func(self.name),
                doc="{0} is a {1} by {1} covariance matrix".format(
                    self.name, self.dim_names[0]),
                )
        for dim_name in self.dim_names:
            properties[dim_name] = property(
                    fget=get_dim_func(dim_name),
                    )
        return properties

def get_Qinv_func(name):
    def fget(self):
        LQinv = getattr(self, "L{0}inv".format(name))
        Qinv = LQinv.dot(LQinv.T) + 1e-9*np.eye(LQinv.shape[0])
        return Qinv
    return fget

def get_Q_func(name):
    def fget(self):
        Qinv = getattr(self, "{0}inv".format(name))
        if np.size(Qinv) == 1:
            Q = Qinv**-1
        else:
            Q = pos_def_mat_inv(Qinv)
        return Q
    return fget

class CovariancePriorHelper(PriorHelper):
    def __init__(self, name='Q', dim_names=None, matrix_name=None):
        self.name = name
        self._scale_name = 'scale_{0}inv'.format(name)
        self._df_name = 'df_{0}inv'.format(name)
        self._inv_name = '{0}inv'.format(name)
        self._lt_prec_name = 'L{0}inv'.format(name)
        self.dim_names = ['n'] if dim_names is None else dim_names
        self.matrix_name = matrix_name
        return

    def set_hyperparams(self, prior, **kwargs):
        if self._scale_name in kwargs:
            n, n2 = np.shape(kwargs[self._scale_name])
        else:
            raise ValueError("{} must be provided".format(self._scale_name))
        if self._df_name not in kwargs:
            raise ValueError("{} must be provided".format(self._df_name))

        if n != n2:
            raise ValueError("{} must be square".format(self._scale_name))

        prior._set_check_dim(**{self.dim_names[0]: n})
        prior.hyperparams[self._scale_name] = kwargs[self._scale_name]
        prior.hyperparams[self._df_name] = kwargs[self._df_name]
        return

    def sample_prior(self, prior, var_dict, **kwargs):
        scale_Qinv = prior.hyperparams[self._scale_name]
        df_Qinv = prior.hyperparams[self._df_name]

        Qinv = array_wishart_rvs(df=df_Qinv, scale=scale_Qinv)
        LQinv = np.linalg.cholesky(Qinv)
        var_dict[self._lt_prec_name] = LQinv
        return

    def _get_matrix_hyperparam(self, prior):
        if self.matrix_name is not None:
            mean = prior.hyperparams['mean_{0}'.format(self.matrix_name)]
            mean_prec = (mean *
                prior.hyperparams['var_col_{0}'.format(self.matrix_name)]**-1)
            prec = np.diag(
                    prior.hyperparams['var_col_{0}'.format(self.matrix_name)]**-1)
        else:
            raise RuntimeError("matrix_name not specified for {0}".format(
                self.name))
        return mean, mean_prec, prec

    def sample_posterior(self, prior, var_dict, sufficient_stat, **kwargs):
        mean, mean_prec, prec = self._get_matrix_hyperparam(prior)
        scale_Qinv = prior.hyperparams[self._scale_name]
        df_Qinv = prior.hyperparams[self._df_name]

        if len(np.shape(prec)) == 1:
            S_prevprev = \
                prec + sufficient_stat[self.name]['S_prevprev']
            S_curprev = \
                mean_prec + sufficient_stat[self.name]['S_curprev']
            S_curcur =  np.outer(mean, mean_prec) + \
                    sufficient_stat[self.name]['S_curcur']
            S_schur = S_curcur - np.outer(S_curprev, S_curprev)/S_prevprev
            df_Q = df_Qinv + sufficient_stat[self.name]['S_count']
            scale_Qinv = \
                np.linalg.inv(np.linalg.inv(scale_Qinv) + S_schur)
            Qinv = array_wishart_rvs(df=df_Q, scale=scale_Qinv)
        else:
            S_prevprev = \
                np.diag(prec) + sufficient_stat[self.name]['S_prevprev']
            S_curprev = \
                mean_prec + sufficient_stat[self.name]['S_curprev']
            S_curcur =  np.matmul(mean, mean_prec) + \
                    sufficient_stat[self.name]['S_curcur']
            S_schur = S_curcur - np.matmul(S_curprev,
                    np.linalg.solve(S_prevprev, S_curprev.T))
            df_Q = df_Qinv + sufficient_stat[self.name]['S_count']
            scale_Qinv = \
                np.linalg.inv(np.linalg.inv(scale_Qinv) + S_schur)
            Qinv = array_wishart_rvs(df=df_Q, scale=scale_Qinv)

        LQinv = np.linalg.cholesky(Qinv)
        var_dict[self._lt_prec_name] = LQinv
        return

    def logprior(self, prior, logprior, parameters, **kwargs):
        scale_Qinv = prior.hyperparams[self._scale_name]
        df_Qinv = prior.hyperparams[self._df_name]

        logprior += scipy.stats.wishart.logpdf(
                getattr(parameters, self._inv_name),
            df=df_Qinv, scale=scale_Qinv)

        return logprior

    def grad_logprior(self, prior, grad, parameters, **kwargs):
        scale_Qinv = prior.hyperparams[self._scale_name]
        df_Qinv = prior.hyperparams[self._df_name]
        LQinv = getattr(parameters, self._lt_prec_name)
        grad_LQinv = \
            (df_Qinv - LQinv.shape[0] - 1) * np.linalg.inv(LQinv.T) - \
            np.linalg.solve(scale_Qinv, LQinv)
        grad_LQinv[np.triu_indices_from(LQinv, 1)] = 0
        grad[self._lt_prec_name] = grad_LQinv
        return

    def get_prior_kwargs(self, prior_kwargs, parameters, **kwargs):
        var = kwargs['var']
        if kwargs.get('from_mean', False):
            Qinv = getattr(parameters, self._inv_name)
        else:
            Qinv = np.eye(getattr(parameters.self._lt_prec_name).shape[0])
        df_Qinv = np.shape(Qinv)[-1] + 1.0 + var**-1
        scale_Qinv = Qinv/df_Qinv

        prior_kwargs[self._scale_name] = scale_Qinv
        prior_kwargs[self._df_name] = df_Qinv
        return

    def get_default_kwargs(self, default_kwargs, **kwargs):
        n = kwargs[self.dim_names[0]]
        var = kwargs['var']
        Qinv = np.eye(n)
        df_Qinv = np.shape(Qinv)[-1] + 1.0 + var**-1
        scale_Qinv = Qinv/df_Qinv

        default_kwargs[self._scale_name] = scale_Qinv
        default_kwargs[self._df_name] = df_Qinv
        return

class CovariancePrecondHelper(PrecondHelper):
    def __init__(self, name='Q', dim_names=None):
        self.name = name
        self._inv_name = '{0}inv'.format(name)
        self._lt_prec_name = 'L{0}inv'.format(name)
        self.dim_names = ['n'] if dim_names is None else dim_names
        return

    def precondition(self, preconditioner,
            precond_grad, grad, parameters, **kwargs):
        Qinv = getattr(parameters, self._inv_name)
        grad[self._lt_prec_name][np.triu_indices_from(Qinv, 1)] = 0
        precond_grad[self._lt_prec_name] = np.dot(0.5*Qinv,
                grad[self._lt_prec_name])
        return

    def precondition_noise(self, preconditioner,
            noise, parameters, **kwargs):
        LQinv = getattr(parameters, self._lt_prec_name)
        noise[self._lt_prec_name] = np.dot(np.sqrt(0.5)*LQinv,
                np.random.normal(loc=0, size=LQinv.shape)
                )
        noise[self._lt_prec_name][np.triu_indices_from(LQinv, 1)] = 0
        return

    def correction_term(self, preconditioner, correction, parameters, **kwargs):
        LQinv = getattr(parameters, self._lt_prec_name)
        correction[self._lt_prec_name] = 0.5 * (LQinv.shape[0]+1) * LQinv
        return


# Multiple Covariance
class CovariancesParamHelper(ParamHelper):
    def __init__(self, name='Q', dim_names=None):
        self.name = name
        self._lt_prec_name = 'L{}inv'.format(name)
        self._inv_name = '{}inv'.format(name)
        self.dim_names = ['n', 'num_states'] if dim_names is None else dim_names
        return

    def set_var(self, param, **kwargs):
        if self._lt_prec_name in kwargs:
            num_states, n, n2 = np.shape(kwargs[self._lt_prec_name])
            if n != n2:
                raise ValueError("{} must be square matrix".format(
                    self._lt_prec_name))
            LQinv = np.array(kwargs[self._lt_prec_name]).astype(float)
            for LQinv_k in LQinv:
                LQinv_k[np.triu_indices_from(LQinv_k, 1)] = 0.0 # zero upper tri

            param.var_dict[self._lt_prec_name] = LQinv
            param._set_check_dim(**{
                self.dim_names[0]: n,
                self.dim_names[1]: num_states,
                })
        else:
            raise ValueError("{} not provided".format(self._lt_prec_name))
        return

    def project_parameters(self, param, **kwargs):
        name_kwargs = kwargs.get(self.name, {})
        if name_kwargs.get('thresh', True):
            LQinv = param.var_dict[self._lt_prec_name]
            for k, LQinv_k in enumerate(LQinv):
                if np.any(np.diag(LQinv_k) < 0.0):
                    logger.info(
                        "Reflecting {0}[{2}]: {1} < 0.0".format(
                            self._lt_prec_name, LQinv_k, k)
                        )
                    LQinv_k[:] = np.linalg.cholesky(
                        np.dot(LQinv_k, LQinv_k.T) + \
                                np.eye(param.dim[self.dim_names[0]])*1e-9
                        )
            param.var_dict[self._lt_prec_name] = LQinv
        if name_kwargs.get('fixed') is not None:
            param.var_dict[self._lt_prec_name] = name_kwargs['fixed'].copy()
        return

    def from_dict_to_vector(self, vector_list, var_dict, **kwargs):
        LQinv = var_dict[self._lt_prec_name]
        vector_list.extend([LQinv_k[np.tril_indices_from(LQinv_k)]
            for LQinv_k in LQinv])
        return

    def from_vector_to_dict(self, var_dict, vector, vector_index, **kwargs):
        n = kwargs[self.dim_names[0]]
        num_states = kwargs[self.dim_names[1]]
        LQinv = np.zeros((num_states, n, n))
        for k in range(num_states):
            LQinv[k][np.tril_indices(n)] = \
                    vector[vector_index + k*(n+1)*n//2:
                           vector_index + (k+1)*(n+1)*n//2]
        var_dict[self._lt_prec_name] = LQinv
        return vector_index+num_states*(n+1)*n//2

    def get_properties(self):
        properties = {}
        properties[self._lt_prec_name] = property(
                fget=get_value_func(self._lt_prec_name),
                fset=set_value_func(self._lt_prec_name),
                doc="{0} is {2} of {1} by {1} lower triangular matrices".format(
                    self._lt_prec_name, self.dim_names[0], self.dim_names[1]),
                )
        properties[self._inv_name] = property(
                fget=get_Qinvs_func(self.name),
                doc="{0} is {2} of {1} by {1} precision matrices".format(
                    self._inv_name, self.dim_names[0], self.dim_names[1]),
                )
        properties[self.name] = property(
                fget=get_Qs_func(self.name),
                doc="{0} is {2} of {1} by {1} covariance matrices".format(
                    self.name, self.dim_names[0], self.dim_names[1]),
                )
        for dim_name in self.dim_names:
            properties[dim_name] = property(
                    fget=get_dim_func(dim_name),
                    )
        return properties

def get_Qinvs_func(name):
    def fget(self):
        LQinv = getattr(self, "L{0}inv".format(name))
        Qinv = np.array([LQinv_k.dot(LQinv_k.T) + 1e-9*np.eye(LQinv_k.shape[0])
            for LQinv_k in LQinv])
        return Qinv
    return fget

def get_Qs_func(name):
    def fget(self):
        Qinv = getattr(self, "{0}inv".format(name))
        if Qinv.shape[1] == 1:
            Q = np.array([Qinv_k**-1 for Qinv_k in Qinv])
        else:
            Q = np.array([pos_def_mat_inv(Qinv_k) for Qinv_k in Qinv])
        return Q
    return fget

class CovariancesPriorHelper(PriorHelper):
    def __init__(self, name='Q', dim_names=None, matrix_name=None):
        self.name = name
        self._scale_name = 'scale_{0}inv'.format(name)
        self._df_name = 'df_{0}inv'.format(name)
        self._inv_name = '{0}inv'.format(name)
        self._lt_prec_name = 'L{0}inv'.format(name)
        self.dim_names = ['n', 'num_states'] if dim_names is None else dim_names
        self.matrix_name = matrix_name
        return

    def set_hyperparams(self, prior, **kwargs):
        if self._scale_name in kwargs:
            num_states, n, n2 = np.shape(kwargs[self._scale_name])
        else:
            raise ValueError("{} must be provided".format(self._scale_name))
        if self._df_name in kwargs:
            num_states2 = np.shape(kwargs[self._df_name])[0]
        else:
            raise ValueError("{} must be provided".format(self._df_name))

        if n != n2:
            raise ValueError("{} must be square".format(self._scale_name))
        if num_states != num_states2:
            raise ValueError("scale and df for {} don't match".format(self.name))

        prior._set_check_dim(**{
            self.dim_names[0]: n,
            self.dim_names[1]: num_states,
            })
        prior.hyperparams[self._scale_name] = kwargs[self._scale_name]
        prior.hyperparams[self._df_name] = kwargs[self._df_name]
        return

    def sample_prior(self, prior, var_dict, **kwargs):
        scale_Qinv = prior.hyperparams[self._scale_name]
        df_Qinv = prior.hyperparams[self._df_name]

        Qinvs = [array_wishart_rvs(df=df_Qinv_k, scale=scale_Qinv_k)
                for df_Qinv_k, scale_Qinv_k in zip(df_Qinv, scale_Qinv)]
        LQinv = np.array([np.linalg.cholesky(Qinv_k) for Qinv_k in Qinvs])
        var_dict[self._lt_prec_name] = LQinv
        return

    def _get_matrix_hyperparam(self, prior):
        if self.matrix_name is not None:
            mean = prior.hyperparams['mean_{0}'.format(self.matrix_name)]
            prec = prior.hyperparams['var_col_{0}'.format(self.matrix_name)]**-1
        else:
            raise RuntimeError("matrix_name not specified for {0}".format(
                self.name))
        return mean, prec

    def sample_posterior(self, prior, var_dict, sufficient_stat, **kwargs):
        mean, prec = self._get_matrix_hyperparam(prior)
        scale_Qinv = prior.hyperparams[self._scale_name]
        df_Qinv = prior.hyperparams[self._df_name]
        n = prior.dim[self.dim_names[0]]
        num_states = prior.dim[self.dim_names[1]]

        Qinvs = [None for _ in range(num_states)]
        if len(np.shape(prec)) == 1:
            for k in range(num_states):
                S_prevprev = prec[k] + \
                        sufficient_stat[self.name]['S_prevprev'][k]
                S_curprev = prec[k] * mean[k] + \
                        sufficient_stat[self.name]['S_curprev'][k]
                S_curcur =  np.outer(mean[k], prec[k]*mean[k]) + \
                        sufficient_stat[self.name]['S_curcur'][k]
                S_schur = S_curcur - np.outer(S_curprev, S_curprev)/S_prevprev
                df_Q_k = df_Qinv[k] + sufficient_stat[self.name]['S_count'][k]
                scale_Qinv_k = \
                    np.linalg.inv(np.linalg.inv(scale_Qinv[k]) + S_schur)
                Qinvs[k] = array_wishart_rvs(df=df_Q_k, scale=scale_Qinv_k)
        else:
            for k in range(num_states):
                S_prevprev = np.diag(prec[k]) + \
                        sufficient_stat[self.name]['S_prevprev'][k]
                S_curprev = prec[k] * mean[k] + \
                        sufficient_stat[self.name]['S_curprev'][k]
                S_curcur =  np.matmul(mean[k], (prec[k] * mean[k]).T) + \
                        sufficient_stat[self.name]['S_curcur'][k]
                S_schur = S_curcur - np.matmul(S_curprev,
                        np.linalg.solve(S_prevprev, S_curprev.T))
                df_Q_k = df_Qinv[k] + sufficient_stat[self.name]['S_count'][k]
                scale_Qinv_k = \
                    np.linalg.inv(np.linalg.inv(scale_Qinv[k]) + S_schur)
                Qinvs[k] = array_wishart_rvs(df=df_Q_k, scale=scale_Qinv_k)

        LQinv = np.array([np.linalg.cholesky(Qinv_k) for Qinv_k in Qinvs])
        var_dict[self._lt_prec_name] = LQinv
        return

    def logprior(self, prior, logprior, parameters, **kwargs):
        scale_Qinv = prior.hyperparams[self._scale_name]
        df_Qinv = prior.hyperparams[self._df_name]
        Qinv = getattr(parameters, self._inv_name)

        for Qinv_k, df_Qinv_k, scale_Qinv_k in zip(
                Qinv, df_Qinv, scale_Qinv):
            logprior += scipy.stats.wishart.logpdf(Qinv_k,
                    df=df_Qinv_k, scale=scale_Qinv_k)

        return logprior

    def grad_logprior(self, prior, grad, parameters, **kwargs):
        scale_Qinv = prior.hyperparams[self._scale_name]
        df_Qinv = prior.hyperparams[self._df_name]
        LQinv = getattr(parameters, self._lt_prec_name)
        grad_LQinv = np.array([
            (df_Qinv_k - LQinv.shape[1] - 1) * np.linalg.inv(LQinv_k.T) - \
            np.linalg.solve(scale_Qinv_k, LQinv_k)
             for LQinv_k, df_Qinv_k, scale_Qinv_k in zip(
                 LQinv, df_Qinv, scale_Qinv)
             ])
        for grad_LQinv_k in grad_LQinv:
            grad_LQinv_k[np.triu_indices_from(grad_LQinv_k, 1)] = 0
        grad[self._lt_prec_name] = grad_LQinv
        return

    def get_prior_kwargs(self, prior_kwargs, parameters, **kwargs):
        var = kwargs['var']
        Qinv = getattr(parameters, self._inv_name)
        if not kwargs.get('from_mean', False):
            Qinv = np.array([np.eye(Qinv.shape[1])
                for _ in range(Qinv.shape[0])])
        df_Qinv = np.shape(Qinv)[-1] + 1.0 + var**-1
        scale_Qinv = Qinv/df_Qinv
        df_Qinv = np.array([df_Qinv+0 for k in range(Qinv.shape[0])])

        prior_kwargs[self._scale_name] = scale_Qinv
        prior_kwargs[self._df_name] = df_Qinv
        return

    def get_default_kwargs(self, default_kwargs, **kwargs):
        n = kwargs[self.dim_names[0]]
        num_states = kwargs[self.dim_names[1]]
        var = kwargs['var']
        Qinv = np.array([np.eye(n) for _ in range(num_states)])
        df_Qinv = np.shape(Qinv)[-1] + 1.0 + var**-1
        scale_Qinv = Qinv/df_Qinv
        df_Qinv = np.array([df_Qinv+0 for k in range(Qinv.shape[0])])

        default_kwargs[self._scale_name] = scale_Qinv
        default_kwargs[self._df_name] = df_Qinv
        return

class CovariancesPrecondHelper(PrecondHelper):
    def __init__(self, name='Q', dim_names=None):
        self.name = name
        self._inv_name = '{0}inv'.format(name)
        self._lt_prec_name = 'L{0}inv'.format(name)
        self.dim_names = ['n', 'num_states'] if dim_names is None else dim_names
        return

    def precondition(self, preconditioner,
            precond_grad, grad, parameters, **kwargs):
        Qinv = getattr(parameters, self._inv_name)
        num_states = Qinv.shape[0]
        for k in range(num_states):
            grad[self._lt_prec_name][k][np.triu_indices_from(Qinv[k], 1)] = 0
        precond_grad[self._lt_prec_name] = np.array([
            np.dot(0.5*Qinv[k], grad[self._lt_prec_name][k])
            for k in range(num_states)
            ])
        return

    def precondition_noise(self, preconditioner,
            noise, parameters, **kwargs):
        LQinv = getattr(parameters, self._lt_prec_name)
        noise[self._lt_prec_name] = np.array([
            np.dot(np.sqrt(0.5)*LQinv[k],
                np.random.normal(loc=0, size=(LQinv.shape[1], LQinv.shape[1]))
                )
            for k in range(LQinv.shape[0])
        ])
        for noise_k in noise[self._lt_prec_name]:
            noise_k[np.triu_indices_from(noise_k, 1)] = 0
        return

    def correction_term(self, preconditioner, correction, parameters, **kwargs):
        LQinv = getattr(parameters, self._lt_prec_name)
        correction[self._lt_prec_name] = 0.5 * (LQinv.shape[1]+1) * LQinv
        return



# Single Shared Covariance
### TODO

if __name__ == "__main__":
    # Demo of Parameters
    class CovParameters(BaseParameters):
        """ Cov Parameters """
        _param_helper_list = [
                CovarianceParamHelper(name='Q', dim_names=['n'])
                ]
        for param_helper in _param_helper_list:
            properties = param_helper.get_properties()
            for name, prop in properties.items():
                vars()[name] = prop

        def __str__(self):
            my_str = "CovParameters:"
            my_str += "\nQ:\n" + str(self.Q)
            return my_str

    class CovPrior(BasePrior):
        """ Cov Prior """
        _Parameters = CovParameters
        _prior_helper_list = [
            CovariancePriorHelper(name='Q', dim_names=['n'])
                ]


