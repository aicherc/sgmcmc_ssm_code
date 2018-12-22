import numpy as np
import scipy.stats
import logging
from .._utils import (
        matrix_normal_logpdf,
        pos_def_mat_inv,
        )
logger = logging.getLogger(name=__name__)

# Transition Matrix for regression covariates (m by d) matrices
class DMixin(object):
    # Mixin for D[0], ..., D[num_states-1] variables
    def _set_dim(self, **kwargs):
        if 'D' in kwargs:
            num_states, m, d = np.shape(kwargs['D'])
        else:
            raise ValueError("D not provided")
        self._set_check_dim(num_states=num_states, m=m, d=d)
        super()._set_dim(**kwargs)
        return

    def _set_var_dict(self, **kwargs):
        if 'D' in kwargs:
            self.var_dict['D'] = np.array(kwargs['D']).astype(float)
        else:
            raise ValueError("D not provided")

        super()._set_var_dict(**kwargs)
        return

    @classmethod
    def _from_dict_to_vector(cls, vector_list, var_dict, **kwargs):
        vector_list.append(var_dict['D'].flatten())
        return super()._from_dict_to_vector(vector_list, var_dict, **kwargs)

    @classmethod
    def _from_vector_to_dict(cls, var_dict, vector, **kwargs):
        num_states, m, d = kwargs['num_states'], kwargs['m'], kwargs['d']
        D = np.reshape(vector[0:num_states*m*d], (num_states, m, d))
        var_dict['D'] = D
        var_dict = super()._from_vector_to_dict(
                var_dict, vector[num_states*m*d:], **kwargs)
        return var_dict

    @property
    def D(self):
        D = self.var_dict['D']
        return D
    @D.setter
    def D(self, D):
        self.var_dict['D'] = D
        return

    @property
    def num_states(self):
        return self.dim['num_states']
    @property
    def m(self):
        return self.dim['m']
    @property
    def d(self):
        return self.dim['d']

class DPrior(object):
    # Mixin for D variable
    def _set_hyperparams(self, **kwargs):
        if 'mean_D' in kwargs:
            num_states, m, d = np.shape(kwargs['mean_D'])
        else:
            raise ValueError("mean_D must be provided")
        if 'var_col_D' in kwargs:
            num_states2, d2 = np.shape(kwargs['var_col_D'])
        else:
            raise ValueError("var_col_D must be provided")

        if num_states != num_states2:
            raise ValueError("num_states don't match")
        if d != d2:
            raise ValueError("prior dimensions don't match")

        self._set_check_dim(num_states=num_states, m=m, d=d)
        self.hyperparams['mean_D'] = kwargs['mean_D']
        self.hyperparams['var_col_D'] = kwargs['var_col_D']
        super()._set_hyperparams(**kwargs)
        return

    def _sample_prior_var_dict(self, var_dict, **kwargs):
        # Requires Rinvs defined
        mean_D = self.hyperparams['mean_D']
        var_col_D = self.hyperparams['var_col_D']
        if "Rinvs" in kwargs:
            Rinvs = kwargs['Rinvs']
        elif "Rinv" in kwargs:
            Rinvs = np.array([kwargs['Rinv']
                for _ in range(self.dim['num_states'])])
        else:
            raise ValueError("Missing Covariance R")

        Ds = [None for k in range(self.dim['num_states'])]
        for k in range(len(Ds)):
            Ds[k] =  scipy.stats.matrix_normal(
                    mean=mean_D[k],
                    rowcov=np.linalg.inv(Rinvs[k]),
                    colcov=np.diag(var_col_D[k]),
                    ).rvs()
        var_dict['D'] = np.array(Ds)
        var_dict = super()._sample_prior_var_dict(var_dict, **kwargs)
        return var_dict

    def _sample_post_var_dict(self, var_dict, sufficient_stat, **kwargs):
        # Requires Rinvs defined
        mean_D = self.hyperparams['mean_D']
        var_col_D = self.hyperparams['var_col_D']
        if "Rinvs" in kwargs:
            Rinvs = kwargs['Rinvs']
        elif "Rinv" in kwargs:
            Rinvs = np.array([kwargs['Rinv']
                for _ in range(self.dim['num_states'])])
        else:
            raise ValueError("Missing Covariance")

        Ds = [None for k in range(self.dim['num_states'])]
        for k in range(0, self.dim['num_states']):
            S_prevprev = np.diag(var_col_D[k]**-1) + \
                    sufficient_stat['S_prevprev'][k]
            S_curprev = mean_D[k] * var_col_D[k]**-1  + \
                    sufficient_stat['S_curprev'][k]
            Ds[k] = scipy.stats.matrix_normal(
                    mean=np.linalg.solve(S_prevprev, S_curprev.T).T,
                    rowcov=np.linalg.inv(Rinvs[k]),
                    colcov=pos_def_mat_inv(S_prevprev),
                    ).rvs()
        var_dict['D'] = np.array(Ds)
        var_dict = super()._sample_post_var_dict(
                var_dict, sufficient_stat, **kwargs)
        return var_dict

    def _logprior(self, logprior, parameters, **kwargs):
        mean_D = self.hyperparams['mean_D']
        var_col_D = self.hyperparams['var_col_D']
        for D_k, mean_D_k, var_col_D_k, LRinv_k in zip(parameters.D,
                mean_D, var_col_D, parameters.LRinv):
            logprior += matrix_normal_logpdf(D_k,
                    mean=mean_D_k,
                    Lrowprec=LRinv_k,
                    Lcolprec=np.diag(var_col_D_k**-0.5),
                    )

        logprior = super()._logprior(logprior, parameters, **kwargs)
        return logprior

    def _grad_logprior(self, grad, parameters, **kwargs):
        mean_D = self.hyperparams['mean_D']
        var_col_D = self.hyperparams['var_col_D']
        Rinv = parameters.Rinv
        grad_D = np.array([
            -1.0*np.dot(
                np.dot(Rinv[k], parameters.D[k] - mean_D[k]),
                np.diag(var_col_D[k]**-1))
            for k in range(self.dim['num_states'])
            ])
        grad['D'] = grad_D
        grad = super()._grad_logprior(grad, parameters, **kwargs)
        return grad

    @classmethod
    def _get_default_kwargs(cls, default_kwargs, **kwargs):
        num_states = kwargs['num_states']
        m = kwargs['m']
        d = kwargs['d']
        var = kwargs['var']

        default_kwargs['mean_D'] = np.zeros((num_states, m, d))
        default_kwargs['var_col_D'] = np.array([
            np.ones(d)*var for _ in range(num_states)
            ])

        default_kwargs = super()._get_default_kwargs(default_kwargs, **kwargs)
        return default_kwargs

    @classmethod
    def _get_prior_kwargs(cls, prior_kwargs, parameters, **kwargs):
        var = kwargs['var']
        if kwargs.get('from_mean', False):
            mean_D = parameters.D.copy()
        else:
            mean_D = np.zeros_like(parameters.D)
        var_col_D = np.array([
            np.ones(parameters.d)*var for _ in range(parameters.num_states)
            ])

        prior_kwargs['mean_D'] = mean_D
        prior_kwargs['var_col_D'] = var_col_D
        prior_kwargs = super()._get_prior_kwargs(
                prior_kwargs, parameters, **kwargs)
        return prior_kwargs

    def _get_R_hyperparam_mean_var_col(self):
        return self.hyperparams['mean_D'], self.hyperparams['var_col_D']

class DPreconditioner(object):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def _precondition(self, precond_grad, grad, parameters, **kwargs):
        R = parameters.R
        precond_D = np.array([
            np.dot(R[k], grad['D'][k])
            for k in range(parameters.num_states)
            ])
        precond_grad['D'] = precond_D
        precond_grad = super()._precondition(precond_grad, grad,
                parameters, **kwargs)
        return precond_grad

    def _precondition_noise(self, noise, parameters, **kwargs):
        LRinv = parameters.LRinv
        precond_D = np.array([
                np.linalg.solve(LRinv[k].T,
                    np.random.normal(loc=0, size=(parameters.m, parameters.d)),
                )
            for k in range(parameters.num_states)
            ])
        noise['D'] = precond_D
        super()._precondition_noise(noise, parameters, **kwargs)
        return noise

    def _correction_term(self, correction, parameters, **kwargs):
        correction['D'] = np.zeros_like(parameters.D, dtype=float)
        super()._correction_term(correction, parameters, **kwargs)
        return correction


