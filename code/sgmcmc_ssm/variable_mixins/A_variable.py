import numpy as np
import scipy.stats
import logging
from .._utils import (
        matrix_normal_logpdf,
        pos_def_mat_inv,
        varp_stability_projection,
        )
logger = logging.getLogger(name=__name__)

# Transition Matrix for regression covariates (m by n) matrices
class AMixin(object):
    # Mixin for A[0], ..., A[num_states-1] variables
    def _set_dim(self, **kwargs):
        if 'A' in kwargs:
            num_states, n, n2 = np.shape(kwargs['A'])
        else:
            raise ValueError("A not provided")
        if n != n2:
            raise ValueError("A must be square matrices")
        self._set_check_dim(num_states=num_states, n=n)
        super()._set_dim(**kwargs)
        return

    def _set_var_dict(self, **kwargs):
        if 'A' in kwargs:
            self.var_dict['A'] = np.array(kwargs['A']).astype(float)
        else:
            raise ValueError("A not provided")

        super()._set_var_dict(**kwargs)
        return

    def _project_parameters(self, **kwargs):
        if kwargs.get('thresh_A', True):
            A = self.A
            for k, A_k in enumerate(A):
                A_k = varp_stability_projection(A_k,
                        eigenvalue_cutoff=kwargs.get('A_eigenvalue_cutoff',
                            0.9999),
                        logger=logger)
                A[k] = A_k
            self.A = A
        return super()._project_parameters(**kwargs)


    @classmethod
    def _from_dict_to_vector(cls, vector_list, var_dict, **kwargs):
        vector_list.append(var_dict['A'].flatten())
        return super()._from_dict_to_vector(vector_list, var_dict, **kwargs)

    @classmethod
    def _from_vector_to_dict(cls, var_dict, vector, **kwargs):
        num_states, m, n = kwargs['num_states'], kwargs['m'], kwargs['n']
        A = np.reshape(vector[0:num_states*m*n], (num_states, m, n))
        var_dict['A'] = A
        var_dict = super()._from_vector_to_dict(
                var_dict, vector[num_states*m*n:], **kwargs)
        return var_dict

    @property
    def A(self):
        A = self.var_dict['A']
        return A
    @A.setter
    def A(self, A):
        self.var_dict['A'] = A
        return

    @property
    def num_states(self):
        return self.dim['num_states']
    @property
    def n(self):
        return self.dim['n']

class APrior(object):
    # Mixin for A variable
    def _set_hyperparams(self, **kwargs):
        if 'mean_A' in kwargs:
            num_states, n, n2 = np.shape(kwargs['mean_A'])
        else:
            raise ValueError("mean_A must be provided")
        if 'var_col_A' in kwargs:
            num_states2, n3 = np.shape(kwargs['var_col_A'])
        else:
            raise ValueError("var_col_A must be provided")

        if num_states != num_states2:
            raise ValueError("num_states don't match")
        if n != n2:
            raise ValueError("mean_A must be square")
        if n != n3:
            raise ValueError("prior dimensions don't match")

        self._set_check_dim(num_states=num_states, n=n)
        self.hyperparams['mean_A'] = kwargs['mean_A']
        self.hyperparams['var_col_A'] = kwargs['var_col_A']
        super()._set_hyperparams(**kwargs)
        return

    def _sample_prior_var_dict(self, var_dict, **kwargs):
        # Requires Qinvs defined
        mean_A = self.hyperparams['mean_A']
        var_col_A = self.hyperparams['var_col_A']
        if "Qinvs" in kwargs:
            Qinvs = kwargs['Qinvs']
        elif "Qinv" in kwargs:
            Qinvs = np.array([kwargs['Qinv']
                for _ in range(self.dim['num_states'])])
        else:
            raise ValueError("Missing Covariance Q")

        As = [None for k in range(self.dim['num_states'])]
        for k in range(len(As)):
            As[k] =  scipy.stats.matrix_normal(
                    mean=mean_A[k],
                    rowcov=np.linalg.inv(Qinvs[k]),
                    colcov=np.diag(var_col_A[k]),
                    ).rvs()
        var_dict['A'] = np.array(As)
        var_dict = super()._sample_prior_var_dict(var_dict, **kwargs)
        return var_dict

    def _sample_post_var_dict(self, var_dict, sufficient_stat, **kwargs):
        # Requires Qinvs defined
        mean_A = self.hyperparams['mean_A']
        var_col_A = self.hyperparams['var_col_A']
        if "Qinvs" in kwargs:
            Qinvs = kwargs['Qinvs']
        elif "Qinv" in kwargs:
            Qinvs = np.array([kwargs['Qinv']
                for _ in range(self.dim['num_states'])])
        else:
            raise ValueError("Missing Covariance")

        As = [None for k in range(self.dim['num_states'])]
        for k in range(0, self.dim['num_states']):
            S_prevprev = np.diag(var_col_A[k]**-1) + \
                    sufficient_stat['Sx_prevprev'][k]
            S_curprev = var_col_A[k]**-1 * mean_A[k] + \
                    sufficient_stat['Sx_curprev'][k]
            As[k] = scipy.stats.matrix_normal(
                    mean=np.linalg.solve(S_prevprev, S_curprev.T).T,
                    rowcov=np.linalg.inv(Qinvs[k]),
                    colcov=pos_def_mat_inv(S_prevprev),
                    ).rvs()
        var_dict['A'] = np.array(As)
        var_dict = super()._sample_post_var_dict(
                var_dict, sufficient_stat, **kwargs)
        return var_dict

    def _logprior(self, logprior, parameters, **kwargs):
        mean_A = self.hyperparams['mean_A']
        var_col_A = self.hyperparams['var_col_A']
        for A_k, mean_A_k, var_col_A_k, LQinv_k in zip(parameters.A,
                mean_A, var_col_A, parameters.LQinv):
            logprior += matrix_normal_logpdf(A_k,
                    mean=mean_A_k,
                    Lrowprec=LQinv_k,
                    Lcolprec=np.diag(var_col_A_k**0.5),
                    )

        logprior = super()._logprior(logprior, parameters, **kwargs)
        return logprior

    def _grad_logprior(self, grad, parameters, **kwargs):
        mean_A = self.hyperparams['mean_A']
        var_col_A = self.hyperparams['var_col_A']
        Qinv = parameters.Qinv
        grad_A = np.array([
            -1.0*np.dot(
                np.dot(Qinv[k], parameters.A[k] - mean_A[k]),
                np.diag(var_col_A[k]**-1))
            for k in range(self.dim['num_states'])
            ])
        grad['A'] = grad_A
        grad = super()._grad_logprior(grad, parameters, **kwargs)
        return grad

    @classmethod
    def _get_default_kwargs(cls, default_kwargs, **kwargs):
        num_states = kwargs['num_states']
        n = kwargs['n']
        var = kwargs['var']

        default_kwargs['mean_A'] = np.zeros((num_states, n, n))
        default_kwargs['var_col_A'] = np.array([
            np.ones(n)*var for _ in range(num_states)
            ])

        default_kwargs = super()._get_default_kwargs(default_kwargs, **kwargs)
        return default_kwargs

    @classmethod
    def _get_prior_kwargs(cls, prior_kwargs, parameters, **kwargs):
        var = kwargs['var']
        if kwargs.get('from_mean', False):
            mean_A = parameters.A.copy()
        else:
            mean_A = np.zeros_like(parameters.A)
        var_col_A = np.array([
            np.ones(parameters.n)*var for _ in range(parameters.num_states)
            ])

        prior_kwargs['mean_A'] = mean_A
        prior_kwargs['var_col_A'] = var_col_A
        prior_kwargs = super()._get_prior_kwargs(
                prior_kwargs, parameters, **kwargs)
        return prior_kwargs

    def _get_Q_hyperparam_mean_var_col(self):
        return self.hyperparams['mean_A'], self.hyperparams['var_col_A']

class APreconditioner(object):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def _precondition(self, precond_grad, grad, parameters, **kwargs):
        Q = parameters.Q
        precond_A = np.array([
            np.dot(Q[k], grad['A'][k])
            for k in range(parameters.num_states)
            ])
        precond_grad['A'] = precond_A
        precond_grad = super()._precondition(precond_grad, grad,
                parameters, **kwargs)
        return precond_grad

    def _precondition_noise(self, noise, parameters, **kwargs):
        LQinv = parameters.LQinv
        precond_A = np.array([
                np.linalg.solve(LQinv[k].T,
                    np.random.normal(loc=0, size=(parameters.n, parameters.n)),
                )
            for k in range(parameters.num_states)
            ])
        noise['A'] = precond_A
        super()._precondition_noise(noise, parameters, **kwargs)
        return noise

    def _correction_term(self, correction, parameters, **kwargs):
        correction['A'] = np.zeros_like(parameters.A, dtype=float)
        super()._correction_term(correction, parameters, **kwargs)
        return correction

class ASingleMixin(object):
    # Mixin for A variables
    def _set_dim(self, **kwargs):
        if 'A' in kwargs:
            n, n2 = np.shape(kwargs['A'])
        else:
            raise ValueError("A not provided")
        if n != n2:
            raise ValueError("A must be square matrices")
        self._set_check_dim(n=n)
        super()._set_dim(**kwargs)
        return

    def _set_var_dict(self, **kwargs):
        if 'A' in kwargs:
            self.var_dict['A'] = np.array(kwargs['A']).astype(float)
        else:
            raise ValueError("A not provided")

        super()._set_var_dict(**kwargs)
        return

    def _project_parameters(self, **kwargs):
        if kwargs.get('thresh_A', True):
            A = self.A
            A = varp_stability_projection(A,
                    eigenvalue_cutoff=kwargs.get('A_eigenvalue_cutoff', 0.9999),
                    logger=logger)
            self.A = A
        return super()._project_parameters(**kwargs)

    @classmethod
    def _from_dict_to_vector(cls, vector_list, var_dict, **kwargs):
        vector_list.append(var_dict['A'].flatten())
        return super()._from_dict_to_vector(vector_list, var_dict, **kwargs)

    @classmethod
    def _from_vector_to_dict(cls, var_dict, vector, **kwargs):
        m, n = kwargs['m'], kwargs['n']
        A = np.reshape(vector[0:m*n], (m, n))
        var_dict['A'] = A
        var_dict = super()._from_vector_to_dict(
                var_dict, vector[m*n:], **kwargs)
        return var_dict

    @property
    def A(self):
        A = self.var_dict['A']
        return A
    @A.setter
    def A(self, A):
        self.var_dict['A'] = A
        return

    @property
    def m(self):
        return self.dim['m']
    @property
    def n(self):
        return self.dim['n']

class ASinglePrior(object):
    # Mixin for A variable
    def _set_hyperparams(self, **kwargs):
        if 'mean_A' in kwargs:
            n, n2 = np.shape(kwargs['mean_A'])
        else:
            raise ValueError("mean_A must be provided")
        if 'var_col_A' in kwargs:
            n3 = np.shape(kwargs['var_col_A'])[0]
        else:
            raise ValueError("var_col_A must be provided")

        if n != n2:
            raise ValueError("mean_A must be square")
        if n != n3:
            raise ValueError("prior dimensions don't match")

        self._set_check_dim(n=n)
        self.hyperparams['mean_A'] = kwargs['mean_A']
        self.hyperparams['var_col_A'] = kwargs['var_col_A']
        super()._set_hyperparams(**kwargs)
        return

    def _sample_prior_var_dict(self, var_dict, **kwargs):
        # Requires Qinvs defined
        mean_A = self.hyperparams['mean_A']
        var_col_A = self.hyperparams['var_col_A']
        if "Qinv" in kwargs:
            Qinv = kwargs['Qinv']
        else:
            raise ValueError("Missing Covariance Q")

        var_dict['A'] =  scipy.stats.matrix_normal(
                mean=mean_A,
                rowcov=np.linalg.inv(Qinv),
                colcov=np.diag(var_col_A),
                ).rvs()
        var_dict = super()._sample_prior_var_dict(var_dict, **kwargs)
        return var_dict

    def _sample_post_var_dict(self, var_dict, sufficient_stat, **kwargs):
        # Requires Qinvs defined
        mean_A = self.hyperparams['mean_A']
        var_col_A = self.hyperparams['var_col_A']
        if "Qinv" in kwargs:
            Qinv = kwargs['Qinv']
        else:
            raise ValueError("Missing Covariance")

        S_prevprev = np.diag(var_col_A**-1) + \
                sufficient_stat['Sx_prevprev']
        S_curprev = mean_A * var_col_A**-1  + \
                sufficient_stat['Sx_curprev']
        var_dict['A'] = scipy.stats.matrix_normal(
                mean=np.linalg.solve(S_prevprev, S_curprev.T).T,
                rowcov=np.linalg.inv(Qinv),
                colcov=pos_def_mat_inv(S_prevprev),
                ).rvs()
        var_dict = super()._sample_post_var_dict(
                var_dict, sufficient_stat, **kwargs)
        return var_dict

    def _logprior(self, logprior, parameters, **kwargs):
        mean_A = self.hyperparams['mean_A']
        var_col_A = self.hyperparams['var_col_A']
        logprior += matrix_normal_logpdf(parameters.A,
                mean=mean_A,
                Lrowprec=parameters.LQinv,
                Lcolprec=np.diag(var_col_A**0.5),
                )

        logprior = super()._logprior(logprior, parameters, **kwargs)
        return logprior

    def _grad_logprior(self, grad, parameters, **kwargs):
        mean_A = self.hyperparams['mean_A']
        var_col_A = self.hyperparams['var_col_A']
        Qinv = parameters.Qinv
        grad['A'] = -1.0 * np.dot(Qinv, parameters.A - mean_A) * var_col_A**-1
        grad = super()._grad_logprior(grad, parameters, **kwargs)
        return grad

    @classmethod
    def _get_default_kwargs(cls, default_kwargs, **kwargs):
        n = kwargs['n']
        var = kwargs['var']

        default_kwargs['mean_A'] = np.zeros((n, n))
        default_kwargs['var_col_A'] = np.ones(n)*var

        default_kwargs = super()._get_default_kwargs(default_kwargs, **kwargs)
        return default_kwargs

    @classmethod
    def _get_prior_kwargs(cls, prior_kwargs, parameters, **kwargs):
        var = kwargs['var']
        if kwargs.get('from_mean', False):
            mean_A = parameters.A.copy()
        else:
            mean_A = np.zeros_like(parameters.A)
        var_col_A = np.ones(parameters.n)*var

        prior_kwargs['mean_A'] = mean_A
        prior_kwargs['var_col_A'] = var_col_A
        prior_kwargs = super()._get_prior_kwargs(
                prior_kwargs, parameters, **kwargs)
        return prior_kwargs

    def _get_Q_hyperparam(self):
        mean = self.hyperparams['mean_A']
        mean_prec = self.hyperparams['var_col_A']**-1 * mean
        prec = np.diag(self.hyperparams['var_col_A']**-1)
        return mean, mean_prec, prec

class ASinglePreconditioner(object):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def _precondition(self, precond_grad, grad, parameters, **kwargs):
        Q = parameters.Q
        precond_grad['A'] = np.dot(Q, grad['A'])
        precond_grad = super()._precondition(precond_grad, grad,
                parameters, **kwargs)
        return precond_grad

    def _precondition_noise(self, noise, parameters, **kwargs):
        LQinv = parameters.LQinv
        noise['A'] = np.linalg.solve(LQinv.T,
                    np.random.normal(loc=0, size=(parameters.n, parameters.n)),
                    )
        super()._precondition_noise(noise, parameters, **kwargs)
        return noise

    def _correction_term(self, correction, parameters, **kwargs):
        correction['A'] = np.zeros_like(parameters.A, dtype=float)
        super()._correction_term(correction, parameters, **kwargs)
        return correction


