import numpy as np
import scipy.stats
import logging
from .._utils import (
        matrix_normal_logpdf,
        pos_def_mat_inv,
        )
logger = logging.getLogger(name=__name__)

# Transition Matrix for regression covariates (m by n) matrices
class CMixin(object):
    # Mixin for C[0], ..., C[num_states-1] variables
    def _set_dim(self, **kwargs):
        if 'C' in kwargs:
            num_states, m, n = np.shape(kwargs['C'])
        else:
            raise ValueError("C not provided")
        self._set_check_dim(num_states=num_states, m=m, n=n)
        super()._set_dim(**kwargs)
        return

    def _set_var_dict(self, **kwargs):
        if 'C' in kwargs:
            self.var_dict['C'] = np.array(kwargs['C']).astype(float)
        else:
            raise ValueError("C not provided")

        super()._set_var_dict(**kwargs)
        return

    @classmethod
    def _from_dict_to_vector(cls, vector_list, var_dict, **kwargs):
        vector_list.append(var_dict['C'].flatten())
        return super()._from_dict_to_vector(vector_list, var_dict, **kwargs)

    @classmethod
    def _from_vector_to_dict(cls, var_dict, vector, **kwargs):
        num_states, m, n = kwargs['num_states'], kwargs['m'], kwargs['n']
        C = np.reshape(vector[0:num_states*m*n], (num_states, m, n))
        var_dict['C'] = C
        var_dict = super()._from_vector_to_dict(
                var_dict, vector[num_states*m*n:], **kwargs)
        return var_dict

    @property
    def C(self):
        C = self.var_dict['C']
        return C
    @C.setter
    def C(self, C):
        self.var_dict['C'] = C
        return

    @property
    def num_states(self):
        return self.dim['num_states']
    @property
    def m(self):
        return self.dim['m']
    @property
    def n(self):
        return self.dim['n']

class CPrior(object):
    # Mixin for C variable
    def _set_hyperparams(self, **kwargs):
        if 'mean_C' in kwargs:
            num_states, m, n = np.shape(kwargs['mean_C'])
        else:
            raise ValueError("mean_C must be provided")
        if 'var_col_C' in kwargs:
            num_states2, n2 = np.shape(kwargs['var_col_C'])
        else:
            raise ValueError("var_col_C must be provided")

        if num_states != num_states2:
            raise ValueError("num_states don't match")
        if n != n2:
            raise ValueError("prior dimensions don't match")

        self._set_check_dim(num_states=num_states, m=m, n=n)
        self.hyperparams['mean_C'] = kwargs['mean_C']
        self.hyperparams['var_col_C'] = kwargs['var_col_C']
        super()._set_hyperparams(**kwargs)
        return

    def _sample_prior_var_dict(self, var_dict, **kwargs):
        # Requires Rinvs defined
        mean_C = self.hyperparams['mean_C']
        var_col_C = self.hyperparams['var_col_C']
        if "Rinvs" in kwargs:
            Rinvs = kwargs['Rinvs']
        elif "Rinv" in kwargs:
            Rinvs = np.array([kwargs['Rinv']
                for _ in range(self.dim['num_states'])])
        else:
            raise ValueError("Missing Covariance R")

        Cs = [None for k in range(self.dim['num_states'])]
        for k in range(len(Cs)):
            Cs[k] =  scipy.stats.matrix_normal(
                    mean=mean_C[k],
                    rowcov=np.linalg.inv(Rinvs[k]),
                    colcov=np.diag(var_col_C[k]),
                    ).rvs()
        var_dict['C'] = np.array(Cs)
        var_dict = super()._sample_prior_var_dict(var_dict, **kwargs)
        return var_dict

    def _sample_post_var_dict(self, var_dict, sufficient_stat, **kwargs):
        # Requires Rinvs defined
        mean_C = self.hyperparams['mean_C']
        var_col_C = self.hyperparams['var_col_C']
        if "Rinvs" in kwargs:
            Rinvs = kwargs['Rinvs']
        elif "Rinv" in kwargs:
            Rinvs = np.array([kwargs['Rinv']
                for _ in range(self.dim['num_states'])])
        else:
            raise ValueError("Missing Covariance")

        Cs = [None for k in range(self.dim['num_states'])]
        for k in range(0, self.dim['num_states']):
            S_prevprev = np.diag(var_col_C[k]**-1) + \
                    sufficient_stat['S_prevprev'][k]
            S_curprev = var_col_C[k]**-1 * mean_C[k] + \
                    sufficient_stat['S_curprev'][k]
            Cs[k] = scipy.stats.matrix_normal(
                    mean=np.linalg.solve(S_prevprev, S_curprev.T).T,
                    rowcov=np.linalg.inv(Rinvs[k]),
                    colcov=pos_def_mat_inv(S_prevprev),
                    ).rvs()
        var_dict['C'] = np.array(Cs)
        var_dict = super()._sample_post_var_dict(
                var_dict, sufficient_stat, **kwargs)
        return var_dict

    def _logprior(self, logprior, parameters, **kwargs):
        mean_C = self.hyperparams['mean_C']
        var_col_C = self.hyperparams['var_col_C']
        for C_k, mean_C_k, var_col_C_k, LRinv_k in zip(parameters.C,
                mean_C, var_col_C, parameters.LRinv):
            logprior += matrix_normal_logpdf(C_k,
                    mean=mean_C_k,
                    Lrowprec=LRinv_k,
                    Lcolprec=np.diag(var_col_C_k**-0.5),
                    )

        logprior = super()._logprior(logprior, parameters, **kwargs)
        return logprior

    def _grad_logprior(self, grad, parameters, **kwargs):
        mean_C = self.hyperparams['mean_C']
        var_col_C = self.hyperparams['var_col_C']
        Rinv = parameters.Rinv
        grad_C = np.array([
            -1.0*np.dot(
                np.dot(Rinv[k], parameters.C[k] - mean_C[k]),
                np.diag(var_col_C[k]**-1))
            for k in range(self.dim['num_states'])
            ])
        grad['C'] = grad_C
        grad = super()._grad_logprior(grad, parameters, **kwargs)
        return grad

    @classmethod
    def _get_default_kwargs(cls, default_kwargs, **kwargs):
        num_states = kwargs['num_states']
        m = kwargs['m']
        n = kwargs['n']
        var = kwargs['var']

        default_kwargs['mean_C'] = np.zeros((num_states, m, n))
        default_kwargs['var_col_C'] = np.array([
            np.ones(n)*var for _ in range(num_states)
            ])

        default_kwargs = super()._get_default_kwargs(default_kwargs, **kwargs)
        return default_kwargs

    @classmethod
    def _get_prior_kwargs(cls, prior_kwargs, parameters, **kwargs):
        var = kwargs['var']
        if kwargs.get('from_mean', False):
            mean_C = parameters.C.copy()
        else:
            mean_C = np.zeros_like(parameters.C)
        var_col_C = np.array([
            np.ones(parameters.n)*var for _ in range(parameters.num_states)
            ])

        prior_kwargs['mean_C'] = mean_C
        prior_kwargs['var_col_C'] = var_col_C
        prior_kwargs = super()._get_prior_kwargs(
                prior_kwargs, parameters, **kwargs)
        return prior_kwargs

    def _get_R_hyperparam_mean_var_col(self):
        return self.hyperparams['mean_C'], self.hyperparams['var_col_C']

class CPreconditioner(object):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def _precondition(self, precond_grad, grad, parameters, **kwargs):
        R = parameters.R
        precond_C = np.array([
            np.dot(R[k], grad['C'][k])
            for k in range(parameters.num_states)
            ])
        precond_grad['C'] = precond_C
        precond_grad = super()._precondition(precond_grad, grad,
                parameters, **kwargs)
        return precond_grad

    def _precondition_noise(self, noise, parameters, **kwargs):
        LRinv = parameters.LRinv
        precond_C = np.array([
                np.linalg.solve(LRinv[k].T,
                    np.random.normal(loc=0, size=(parameters.m, parameters.n)),
                )
            for k in range(parameters.num_states)
            ])
        noise['C'] = precond_C
        super()._precondition_noise(noise, parameters, **kwargs)
        return noise

    def _correction_term(self, correction, parameters, **kwargs):
        correction['C'] = np.zeros_like(parameters.C, dtype=float)
        super()._correction_term(correction, parameters, **kwargs)
        return correction

class CSingleMixin(object):
    # Mixin for C variables
    def _set_dim(self, **kwargs):
        if 'C' in kwargs:
            m, n = np.shape(kwargs['C'])
        else:
            raise ValueError("C not provided")
        self._set_check_dim(m=m, n=n)
        super()._set_dim(**kwargs)
        return

    def _set_var_dict(self, **kwargs):
        if 'C' in kwargs:
            self.var_dict['C'] = np.array(kwargs['C']).astype(float)
        else:
            raise ValueError("C not provided")

        super()._set_var_dict(**kwargs)
        return

    @classmethod
    def _from_dict_to_vector(cls, vector_list, var_dict, **kwargs):
        vector_list.append(var_dict['C'].flatten())
        return super()._from_dict_to_vector(vector_list, var_dict, **kwargs)

    @classmethod
    def _from_vector_to_dict(cls, var_dict, vector, **kwargs):
        m, n = kwargs['m'], kwargs['n']
        C = np.reshape(vector[0:m*n], (m, n))
        var_dict['C'] = C
        var_dict = super()._from_vector_to_dict(
                var_dict, vector[m*n:], **kwargs)
        return var_dict

    @property
    def C(self):
        C = self.var_dict['C']
        return C
    @C.setter
    def C(self, C):
        self.var_dict['C'] = C
        return

    @property
    def m(self):
        return self.dim['m']
    @property
    def n(self):
        return self.dim['n']

class CSinglePrior(object):
    # Mixin for C variable
    def _set_hyperparams(self, **kwargs):
        if 'mean_C' in kwargs:
            m, n = np.shape(kwargs['mean_C'])
        else:
            raise ValueError("mean_C must be provided")
        if 'var_col_C' in kwargs:
            n2 = np.shape(kwargs['var_col_C'])[0]
        else:
            raise ValueError("var_col_C must be provided")

        if n != n2:
            raise ValueError("prior dimensions don't match")

        self._set_check_dim(m=m, n=n)
        self.hyperparams['mean_C'] = kwargs['mean_C']
        self.hyperparams['var_col_C'] = kwargs['var_col_C']
        super()._set_hyperparams(**kwargs)
        return

    def _sample_prior_var_dict(self, var_dict, **kwargs):
        # Requires Rinvs defined
        mean_C = self.hyperparams['mean_C']
        var_col_C = self.hyperparams['var_col_C']
        if "Rinv" in kwargs:
            Rinv = kwargs['Rinv']
        else:
            raise ValueError("Missing Covariance R")

        var_dict['C'] =  scipy.stats.matrix_normal(
                mean=mean_C,
                rowcov=np.linalg.inv(Rinv),
                colcov=np.diag(var_col_C),
                ).rvs()
        var_dict = super()._sample_prior_var_dict(var_dict, **kwargs)
        return var_dict

    def _sample_post_var_dict(self, var_dict, sufficient_stat, **kwargs):
        # Requires Rinvs defined
        mean_C = self.hyperparams['mean_C']
        var_col_C = self.hyperparams['var_col_C']
        if "Rinv" in kwargs:
            Rinv = kwargs['Rinv']
        else:
            raise ValueError("Missing Covariance")

        S_prevprev = np.diag(var_col_C**-1) + \
                sufficient_stat['S_prevprev']
        S_curprev = mean_C * var_col_C**-1  + \
                sufficient_stat['S_curprev']
        var_dict['C'] = scipy.stats.matrix_normal(
                mean=np.linalg.solve(S_prevprev, S_curprev.T).T,
                rowcov=np.linalg.inv(Rinv),
                colcov=pos_def_mat_inv(S_prevprev),
                ).rvs()
        var_dict = super()._sample_post_var_dict(
                var_dict, sufficient_stat, **kwargs)
        return var_dict

    def _logprior(self, logprior, parameters, **kwargs):
        mean_C = self.hyperparams['mean_C']
        var_col_C = self.hyperparams['var_col_C']
        logprior += matrix_normal_logpdf(parameters.C,
                mean=mean_C,
                Lrowprec=parameters.LRinv,
                Lcolprec=np.diag(var_col_C**-0.5),
                )

        logprior = super()._logprior(logprior, parameters, **kwargs)
        return logprior

    def _grad_logprior(self, grad, parameters, **kwargs):
        mean_C = self.hyperparams['mean_C']
        var_col_C = self.hyperparams['var_col_C']
        Rinv = parameters.Rinv
        grad['C'] = -1.0 * np.dot(Rinv, parameters.C - mean_C) * var_col_C**-1
        grad = super()._grad_logprior(grad, parameters, **kwargs)
        return grad

    @classmethod
    def _get_default_kwargs(cls, default_kwargs, **kwargs):
        m = kwargs['m']
        n = kwargs['n']
        var = kwargs['var']

        default_kwargs['mean_C'] = np.zeros((m, n))
        default_kwargs['var_col_C'] = np.ones(n)*var

        default_kwargs = super()._get_default_kwargs(default_kwargs, **kwargs)
        return default_kwargs

    @classmethod
    def _get_prior_kwargs(cls, prior_kwargs, parameters, **kwargs):
        var = kwargs['var']
        if kwargs.get('from_mean', False):
            mean_C = parameters.C.copy()
        else:
            mean_C = np.zeros_like(parameters.C)
        var_col_C = np.ones(parameters.n)*var

        prior_kwargs['mean_C'] = mean_C
        prior_kwargs['var_col_C'] = var_col_C
        prior_kwargs = super()._get_prior_kwargs(
                prior_kwargs, parameters, **kwargs)
        return prior_kwargs

    def _get_R_hyperparam(self):
        mean = self.hyperparams['mean_C']
        mean_prec = self.hyperparams['var_col_C']**-1 * mean
        prec = np.diag(self.hyperparams['var_col_C']**-1)
        return mean, mean_prec, prec

class CSinglePreconditioner(object):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def _precondition(self, precond_grad, grad, parameters, **kwargs):
        R = parameters.R
        precond_grad['C'] = np.dot(R, grad['C'])
        precond_grad = super()._precondition(precond_grad, grad,
                parameters, **kwargs)
        return precond_grad

    def _precondition_noise(self, noise, parameters, **kwargs):
        LRinv = parameters.LRinv
        noise['C'] = np.linalg.solve(LRinv.T,
                    np.random.normal(loc=0, size=(parameters.m, parameters.n)),
                    )
        super()._precondition_noise(noise, parameters, **kwargs)
        return noise

    def _correction_term(self, correction, parameters, **kwargs):
        correction['C'] = np.zeros_like(parameters.C, dtype=float)
        super()._correction_term(correction, parameters, **kwargs)
        return correction


