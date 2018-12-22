import numpy as np
import scipy.stats
import logging
from .._utils import array_wishart_rvs, pos_def_mat_inv
logger = logging.getLogger(name=__name__)

class QSingleMixin(object):
    # Mixin for Q variable
    def _set_dim(self, **kwargs):
        if 'LQinv' in kwargs:
            n, n2 = np.shape(kwargs['LQinv'])
        else:
            raise ValueError("LQinv not provided")
        if n != n2:
            raise ValueError("LQinv must be square, not {0}".format(
                (n, n2)))
        self._set_check_dim(n=n)
        super()._set_dim(**kwargs)
        return

    def _set_var_dict(self, **kwargs):
        if 'LQinv' in kwargs:
            LQinv = np.array(kwargs['LQinv']).astype(float)
            LQinv[np.triu_indices_from(LQinv, 1)] = 0.0 # zero out
            self.var_dict['LQinv'] = LQinv
        else:
            raise ValueError("LQinv not provided")
        super()._set_var_dict(**kwargs)
        return

    @classmethod
    def _from_dict_to_vector(cls, vector_list, var_dict, **kwargs):
        LQinv = var_dict['LQinv']
        vector_list.append(LQinv[np.tril_indices_from(LQinv)])
        return super()._from_dict_to_vector(vector_list, var_dict, **kwargs)

    @classmethod
    def _from_vector_to_dict(cls, var_dict, vector, **kwargs):
        n = kwargs['n']
        LQinv = np.zeros((n, n))
        LQinv[np.tril_indices(n)] = vector[0:(n+1)*n//2]
        var_dict['LQinv'] = LQinv
        var_dict = super()._from_vector_to_dict(
                var_dict, vector[(n+1)*n//2:], **kwargs)
        return var_dict

    def _project_parameters(self, **kwargs):
        self.LQinv[np.triu_indices_from(self.LQinv, 1)] = 0

        if kwargs.get('thresh_LQinv', True):
           # Threshold diag(LQinv) to be positive
            LQinv = self.LQinv
            if np.any(np.diag(LQinv) < 0.0):
                logger.info(
                    "Reflecting LQinv: {0} < 0.0".format(LQinv)
                    )
                LQinv[:] = np.linalg.cholesky(
                    np.dot(LQinv, LQinv.T) + np.eye(self.n)*1e-9)
            self.LQinv = LQinv
        if kwargs.get('diag_Q', False):
           # Threshold LQinv to be diagonal
           self.LQinv = np.diag(np.diag(self.LQinv))
        return super()._project_parameters(**kwargs)

    @property
    def LQinv(self):
        LQinv = self.var_dict['LQinv']
        return LQinv
    @LQinv.setter
    def LQinv(self, LQinv):
        self.var_dict['LQinv'] = LQinv
        return
    @property
    def Qinv(self):
        LQinv = self.LQinv
        # 1e-9*np.eye is Fudge Factor for Stability
        Qinv = LQinv.dot(LQinv.T) + 1e-9*np.eye(self.dim['n'])
        return Qinv
    @property
    def Q(self):
        if self.n == 1:
            Q = self.Qinv**-1
        else:
            Q = pos_def_mat_inv(self.Qinv)
        return Q
    @property
    def n(self):
        return self.dim['n']

class QMixin(object):
    # Mixin for Q[0], ..., Q[num_states-1] variables
    def _set_dim(self, **kwargs):
        if 'LQinv' in kwargs:
            num_states, n, n2 = np.shape(kwargs['LQinv'])
        else:
            raise ValueError("LQinv not provided")

        if n != n2:
            raise ValueError("LQinv must be square, not {0}".format(
                (n, n2)))
        self._set_check_dim(num_states=num_states, n=n)
        super()._set_dim(**kwargs)
        return

    def _set_var_dict(self, **kwargs):
        if 'LQinv' in kwargs:
            LQinv = np.array(kwargs['LQinv']).astype(float)
            for LQinv_k in LQinv:
                LQinv_k[np.triu_indices_from(LQinv_k, 1)] = 0
            self.var_dict['LQinv'] = LQinv
        else:
            raise ValueError("LQinv not provided")
        super()._set_var_dict(**kwargs)
        return

    @classmethod
    def _from_dict_to_vector(cls, vector_list, var_dict, **kwargs):
        LQinv = var_dict['LQinv']
        vector_list.extend([LQinv_k[np.tril_indices_from(LQinv_k)]
            for LQinv_k in LQinv])
        return super()._from_dict_to_vector(vector_list, var_dict, **kwargs)

    @classmethod
    def _from_vector_to_dict(cls, var_dict, vector, **kwargs):
        num_states, n = kwargs['num_states'], kwargs['n']
        LQinv_vec = vector[:num_states*(n+1)*n//2]
        LQinv = np.zeros((num_states, n, n))
        for k in range(num_states):
            LQinv[k][np.tril_indices(n)] = \
                    LQinv_vec[k*(n+1)*n//2:(k+1)*(n+1)*n//2]
        var_dict['LQinv'] = LQinv
        var_dict = super()._from_vector_to_dict(
                var_dict, vector[num_states*(n+1)*n//2:], **kwargs)
        return var_dict

    def _project_parameters(self, **kwargs):
        for LQinv_k in self.LQinv:
            LQinv_k[np.triu_indices_from(LQinv_k, 1)] = 0

        if kwargs.get('thresh_LQinv', True):
           # Threshold diag(LQinv) to be positive
            LQinv = self.LQinv
            for LQinv_k in LQinv:
                if np.any(np.diag(LQinv_k) < 0.0):
                    logger.info(
                        "Reflecting LQinv: {0} < 0.0".format(LQinv_k)
                        )
                    LQinv_k[:] = np.linalg.cholesky(
                            np.dot(LQinv_k, LQinv_k.T) + np.eye(self.n)*1e-9)
            self.LQinv = LQinv
        if kwargs.get('diag_Q', False):
           # Threshold LQinv to be diagonal
           self.LQinv = np.array([
                np.diag(np.diag(LQinv_k)) for LQinv_k in self.LQinv
                ])
        return super()._project_parameters(**kwargs)

    @property
    def LQinv(self):
        LQinv = self.var_dict['LQinv']
        return LQinv
    @LQinv.setter
    def LQinv(self, LQinv):
        self.var_dict['LQinv'] = LQinv
        return
    @property
    def Qinv(self):
        LQinv = self.LQinv
        # 1e-9*np.eye is Fudge Factor for Stability
        Qinv = np.array([LQinv_k.dot(LQinv_k.T) + 1e-9*np.eye(self.dim['n'])
                for LQinv_k in LQinv])
        return Qinv
    @property
    def Q(self):
        if self.n == 1:
            Q = np.array([Qinv_k**-1 for Qinv_k in self.Qinv])
        else:
            Q = np.array([pos_def_mat_inv(Qinv_k) for Qinv_k in self.Qinv])
        return Q
    @property
    def n(self):
        return self.dim['n']
    @property
    def num_states(self):
        return self.dim['num_states']

class QSinglePrior(object):
    # Mixin for Q Single variable
    def _set_hyperparams(self, **kwargs):
        if 'scale_Qinv' in kwargs:
            n, n2 = np.shape(kwargs['scale_Qinv'])
        else:
            raise ValueError("scale_Qinv must be provided")
        if 'df_Qinv' not in kwargs:
            raise ValueError("df_Qinv must be provided")

        if n != n2:
            raise ValueError("scale_Qinv has wrong shape")

        self._set_check_dim(n=n)

        self.hyperparams['scale_Qinv'] = kwargs['scale_Qinv']
        self.hyperparams['df_Qinv'] = kwargs['df_Qinv']
        super()._set_hyperparams(**kwargs)
        return

    def _sample_prior_var_dict(self, var_dict, **kwargs):
        # Requires Qinvs defined
        scale_Qinv = self.hyperparams['scale_Qinv']
        df_Qinv = self.hyperparams['df_Qinv']

        Qinv = array_wishart_rvs(df=df_Qinv, scale=scale_Qinv)
        LQinv = np.linalg.cholesky(Qinv)

        var_dict['LQinv'] = LQinv
        var_dict = super()._sample_prior_var_dict(var_dict, Qinv=Qinv,
                **kwargs)
        return var_dict

    def _sample_post_var_dict(self, var_dict, sufficient_stat, **kwargs):
        mean, mean_prec, prec = self._get_Q_hyperparam()
        scale_Qinv = self.hyperparams['scale_Qinv']
        df_Qinv = self.hyperparams['df_Qinv']

        if len(np.shape(prec)) == 1:
            S_prevprev = \
                prec + sufficient_stat['Sx_prevprev']
            S_curprev = \
                mean_prec + sufficient_stat['Sx_curprev']
            S_curcur =  \
                np.outer(mean, mean_prec) + sufficient_stat['Sx_curcur']
            S_schur = S_curcur - np.outer(S_curprev, S_curprev)/S_prevprev
            df_Q = df_Qinv + sufficient_stat['Sx_count']
            scale_Qinv = \
                np.linalg.inv(np.linalg.inv(scale_Qinv) + S_schur)
            Qinv = array_wishart_rvs(df=df_Q, scale=scale_Qinv)
        else:
            S_prevprev = \
                np.diag(prec) + sufficient_stat['Sx_prevprev']
            S_curprev = \
                mean_prec + sufficient_stat['Sx_curprev']
            S_curcur =  \
                np.matmul(mean, mean_prec) + sufficient_stat['Sx_curcur']
            S_schur = S_curcur - np.matmul(S_curprev,
                    np.linalg.solve(S_prevprev, S_curprev.T))
            df_Q = df_Qinv + sufficient_stat['Sx_count']
            scale_Qinv = \
                np.linalg.inv(np.linalg.inv(scale_Qinv) + S_schur)
            Qinv = array_wishart_rvs(df=df_Q, scale=scale_Qinv)

        LQinv = np.linalg.cholesky(Qinv)

        var_dict['LQinv'] = LQinv
        var_dict = super()._sample_post_var_dict(
                var_dict, sufficient_stat, Qinv=Qinv, **kwargs)
        return var_dict

    def _logprior(self, logprior, parameters, **kwargs):
        scale_Qinv = self.hyperparams['scale_Qinv']
        df_Qinv = self.hyperparams['df_Qinv']

        logprior += scipy.stats.wishart.logpdf(parameters.Qinv,
            df=df_Qinv, scale=scale_Qinv)
        logprior = super()._logprior(logprior, parameters, **kwargs)
        return logprior

    def _grad_logprior(self, grad, parameters, **kwargs):
        scale_Qinv = self.hyperparams['scale_Qinv']
        df_Qinv = self.hyperparams['df_Qinv']

        LQinv = parameters.LQinv
        grad_LQinv = \
            (df_Qinv - self.dim['n'] - 1) * np.linalg.inv(LQinv.T) - \
            np.linalg.solve(scale_Qinv, LQinv)

        grad['LQinv'] = grad_LQinv
        grad = super()._grad_logprior(grad, parameters, **kwargs)
        return grad

    @classmethod
    def _get_default_kwargs(cls, default_kwargs, **kwargs):
        n = kwargs['n']
        var = kwargs['var']

        Qinv = np.eye(n)
        df_Qinv = np.shape(Qinv)[-1] + 1.0 + var**-1
        scale_Qinv = Qinv/df_Qinv

        default_kwargs['scale_Qinv'] = scale_Qinv
        default_kwargs['df_Qinv'] = df_Qinv

        default_kwargs = super()._get_default_kwargs(default_kwargs, **kwargs)
        return default_kwargs

    @classmethod
    def _get_prior_kwargs(cls, prior_kwargs, parameters, **kwargs):
        var = kwargs['var']
        if kwargs.get('from_mean', False):
            Qinv = parameters.Qinv
        else:
            Qinv = np.eye(parameters.n)
        df_Qinv = np.shape(Qinv)[-1] + 1.0 + var**-1
        scale_Qinv = Qinv/df_Qinv

        prior_kwargs['scale_Qinv'] = scale_Qinv
        prior_kwargs['df_Qinv'] = df_Qinv

        prior_kwargs = super()._get_prior_kwargs(
                prior_kwargs, parameters, **kwargs)
        return prior_kwargs

class QPrior(object):
    # Mixin for Q variable
    def _set_hyperparams(self, **kwargs):
        if 'scale_Qinv' in kwargs:
            num_states, n, n2 = np.shape(kwargs['scale_Qinv'])
        else:
            raise ValueError("scale_Qinv must be provided")
        if 'df_Qinv' in kwargs:
            num_states2 = np.shape(kwargs['df_Qinv'])[0]
        else:
            raise ValueError("df_Qinv must be provided")

        if n != n2:
            raise ValueError("scale_Qinv has wrong shape")

        if num_states != num_states2:
            raise ValueError("scale_Qinv and df_Qinv don't match")

        if "num_states" in self.dim:
            if num_states != self.dim['num_states']:
                raise ValueError("num_states do not match existing dims")
        else:
            self.dim['num_states'] = num_states
        if "n" in self.dim:
            if m != self.dim['n']:
                raise ValueError("n do not match existing dims")
        else:
            self.dim['n'] = n

        self.hyperparams['scale_Qinv'] = kwargs['scale_Qinv']
        self.hyperparams['df_Qinv'] = kwargs['df_Qinv']
        super()._set_hyperparams(**kwargs)
        return

    def _sample_prior_var_dict(self, var_dict, **kwargs):
        # Requires Qinvs defined
        scale_Qinv = self.hyperparams['scale_Qinv']
        df_Qinv = self.hyperparams['df_Qinv']

        Qinvs = [
                array_wishart_rvs(df=df_Qinv_k, scale=scale_Qinv_k)
                for df_Qinv_k, scale_Qinv_k in zip(df_Qinv, scale_Qinv)
                ]
        LQinv = np.array([np.linalg.cholesky(Qinv_k) for Qinv_k in Qinvs])

        var_dict['LQinv'] = LQinv
        var_dict = super()._sample_prior_var_dict(var_dict, Qinvs=Qinvs,
                **kwargs)
        return var_dict

    def _sample_post_var_dict(self, var_dict, sufficient_stat, **kwargs):
        mean, var_col = self._get_Q_hyperparam_mean_var_col()
        scale_Qinv = self.hyperparams['scale_Qinv']
        df_Qinv = self.hyperparams['df_Qinv']

        Qinvs = [None for _ in range(self.dim['num_states'])]
        if len(np.shape(var_col)) == 1:
            for k in range(0, self.dim['num_states']):
                S_prevprev = \
                    var_col[k]**-1 + sufficient_stat['Sx_prevprev'][k]
                S_curprev = \
                    var_col[k]**-1 * mean[k] + sufficient_stat['Sx_curprev'][k]
                S_curcur =  \
                    np.outer(mean[k], var_col[k]**-1 * mean[k]) + \
                sufficient_stat['Sx_curcur'][k]
                S_schur = S_curcur - np.outer(S_curprev, S_curprev)/S_prevprev
                df_Q_k = df_Qinv[k] + sufficient_stat['Sx_count'][k]
                scale_Qinv_k = \
                    np.linalg.inv(np.linalg.inv(scale_Qinv[k]) + S_schur)
                Qinvs[k] = array_wishart_rvs(df=df_Q_k, scale=scale_Qinv_k)
        else:
            for k in range(0, self.dim['num_states']):
                S_prevprev = \
                    np.diag(var_col[k]**-1) + sufficient_stat['Sx_prevprev'][k]
                S_curprev = \
                    var_col[k]**-1 * mean[k] + sufficient_stat['Sx_curprev'][k]
                S_curcur =  \
                    np.matmul(mean[k], (var_col[k]**-1 * mean[k]).T) + \
                sufficient_stat['Sx_curcur'][k]
                S_schur = S_curcur - np.matmul(S_curprev,
                        np.linalg.solve(S_prevprev, S_curprev.T))
                df_Q_k = df_Qinv[k] + sufficient_stat['Sx_count'][k]
                scale_Qinv_k = \
                        np.linalg.inv(np.linalg.inv(scale_Qinv[k]) + S_schur)
                Qinvs[k] = array_wishart_rvs(df=df_Q_k, scale=scale_Qinv_k)

        Qinvs = np.array(Qinvs)
        LQinv = np.array([np.linalg.cholesky(Qinv_k) for Qinv_k in Qinvs])

        var_dict['LQinv'] = LQinv
        var_dict = super()._sample_post_var_dict(
                var_dict, sufficient_stat, Qinvs=Qinvs, **kwargs)
        return var_dict

    def _logprior(self, logprior, parameters, **kwargs):
        scale_Qinv = self.hyperparams['scale_Qinv']
        df_Qinv = self.hyperparams['df_Qinv']

        for Qinv_k, df_Qinv_k, scale_Qinv_k in zip(
                parameters.Qinv, df_Qinv, scale_Qinv):
            logprior += scipy.stats.wishart.logpdf(Qinv_k,
                df=df_Qinv_k, scale=scale_Qinv_k)
        logprior = super()._logprior(logprior, parameters, **kwargs)
        return logprior

    def _grad_logprior(self, grad, parameters, **kwargs):
        scale_Qinv = self.hyperparams['scale_Qinv']
        df_Qinv = self.hyperparams['df_Qinv']

        grad_LQinv = np.array([
            (df_Qinv_k - self.dim['m'] - 1) * np.linalg.inv(LQinv_k.T) - \
            np.linalg.solve(scale_Qinv_k, LQinv_k)
            for LQinv_k, df_Qinv_k, scale_Qinv_k in zip(
                parameters.LQinv, df_Qinv, scale_Qinv)
            ])

        grad['LQinv'] = grad_LQinv
        grad = super()._grad_logprior(grad, parameters, **kwargs)
        return grad

    @classmethod
    def _get_default_kwargs(cls, default_kwargs, **kwargs):
        num_states = kwargs['num_states']
        n = kwargs['n']
        var = kwargs['var']

        Qinv = np.array([np.eye(n) for _ in range(num_states)])
        df_Qinv = np.shape(Qinv)[-1] + 1.0 + var**-1
        scale_Qinv = Qinv/df_Qinv
        df_Qinv = np.array([df_Qinv+0 for k in range(num_states)])

        default_kwargs['scale_Qinv'] = scale_Qinv
        default_kwargs['df_Qinv'] = df_Qinv

        default_kwargs = super()._get_default_kwargs(default_kwargs, **kwargs)
        return default_kwargs

    @classmethod
    def _get_prior_kwargs(cls, prior_kwargs, parameters, **kwargs):
        var = kwargs['var']
        if kwargs.get('from_mean', False):
            Qinv = parameters.Qinv
        else:
            Qinv = np.array([
                np.eye(parameters.n) for _ in range(parameters.nun_states)
                ])
        df_Qinv = np.shape(Qinv)[-1] + 1.0 + var**-1
        scale_Qinv = Qinv/df_Qinv
        df_Qinv = np.array([df_Qinv+0 for k in range(paraneters.num_states)])

        prior_kwargs['scale_Qinv'] = scale_Qinv
        prior_kwargs['df_Qinv'] = df_Qinv

        prior_kwargs = super()._get_prior_kwargs(
                prior_kwargs, parameters, **kwargs)
        return prior_kwargs

class QSinglePreconditioner(object):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def _precondition(self, precond_grad, grad, parameters, **kwargs):
        Qinv = parameters.Qinv
        precond_grad['LQinv'] = np.dot(0.5*Qinv, grad['LQinv'])
        precond_grad = super()._precondition(precond_grad, grad,
                parameters, **kwargs)
        return precond_grad

    def _precondition_noise(self, noise, parameters, **kwargs):
        LQinv = parameters.LQinv
        precond_LQinv = np.dot(np.sqrt(0.5)*LQinv,
                np.random.normal(loc=0, size=(parameters.n, parameters.n)),
                )
        precond_LQinv[np.triu_indices_from(precond_LQinv, 1)] = 0
        noise['LQinv'] = precond_LQinv
        super()._precondition_noise(noise, parameters, **kwargs)
        return noise

    def _correction_term(self, correction, parameters, **kwargs):
        correction['LQinv'] = parameters.LQinv
        super()._correction_term(correction, parameters, **kwargs)
        return correction

class QPreconditioner(object):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def _precondition(self, precond_grad, grad, parameters, **kwargs):
        Qinv = parameters.Qinv
        precond_LQinv = np.array([
            np.dot(0.5*Qinv[k], grad['LQinv'][k])
            for k in range(parameters.num_states)
            ])

        precond_grad['LQinv'] = precond_LQinv
        precond_grad = super()._precondition(precond_grad, grad,
                parameters, **kwargs)
        return precond_grad

    def _precondition_noise(self, noise, parameters, **kwargs):
        LQinv = parameters.LQinv
        precond_LQinv = np.array([
            np.dot(np.sqrt(0.5)*LQinv[k],
                np.random.normal(loc=0, size=(parameters.m, parameters.m)),
                )
            for k in range(parameters.num_states)
            ])
        for precond_LQinv_k in precond_LQinv:
            precond_LQinv_k[np.triu_indices_from(precond_LQinv_k, 1)] = 0

        noise['LQinv'] = precond_LQinv
        super()._precondition_noise(noise, parameters, **kwargs)
        return noise

    def _correction_term(self, correction, parameters, **kwargs):
        correction['LQinv'] = parameters.LQinv
        super()._correction_term(correction, parameters, **kwargs)
        return correction


