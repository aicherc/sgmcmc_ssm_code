import numpy as np
import scipy.stats
import logging
from .._utils import array_wishart_rvs, pos_def_mat_inv
logger = logging.getLogger(name=__name__)

class RSingleMixin(object):
    # Mixin for R variable
    def _set_dim(self, **kwargs):
        if 'LRinv' in kwargs:
            m, m2 = np.shape(kwargs['LRinv'])
        else:
            raise ValueError("LRinv not provided")
        if m != m2:
            raise ValueError("LRinv must be square, not {0}".format(
                (m, m2)))
        self._set_check_dim(m=m)
        super()._set_dim(**kwargs)
        return

    def _set_var_dict(self, **kwargs):
        if 'LRinv' in kwargs:
            LRinv = np.array(kwargs['LRinv']).astype(float)
            LRinv[np.triu_indices_from(LRinv, 1)] = 0.0 # zero out
            self.var_dict['LRinv'] = LRinv
        else:
            raise ValueError("LRinv not provided")
        super()._set_var_dict(**kwargs)
        return

    @classmethod
    def _from_dict_to_vector(cls, vector_list, var_dict, **kwargs):
        LRinv = var_dict['LRinv']
        vector_list.append(LRinv[np.tril_indices_from(LRinv)])
        return super()._from_dict_to_vector(vector_list, var_dict, **kwargs)

    @classmethod
    def _from_vector_to_dict(cls, var_dict, vector, **kwargs):
        m = kwargs['m']
        LRinv = np.zeros((m, m))
        LRinv[np.tril_indices(m)] = vector[0:(m+1)*m//2]
        var_dict['LRinv'] = LRinv
        var_dict = super()._from_vector_to_dict(
                var_dict, vector[(m+1)*m//2:], **kwargs)
        return var_dict

    def _project_parameters(self, **kwargs):
        self.LRinv[np.triu_indices_from(self.LRinv, 1)] = 0

        if kwargs.get('thresh_LRinv', True):
           # Threshold diag(LRinv) to be positive
            LRinv = self.LRinv
            if np.any(np.diag(LRinv) < 0.0):
                logger.info(
                    "Reflecting LRinv: {0} < 0.0".format(LRinv)
                    )
                LRinv[:] = np.linalg.cholesky(
                    np.dot(LRinv, LRinv.T) + np.eye(self.m)*1e-9)
            self.LRinv = LRinv
        if kwargs.get('diag_R', False):
           # Threshold LRinv to be diagonal
           self.LRinv = np.diag(np.diag(self.LRinv))
        return super()._project_parameters(**kwargs)

    @property
    def LRinv(self):
        LRinv = self.var_dict['LRinv']
        return LRinv
    @LRinv.setter
    def LRinv(self, LRinv):
        self.var_dict['LRinv'] = LRinv
        return
    @property
    def Rinv(self):
        LRinv = self.LRinv
        # 1e-9*np.eye is Fudge Factor for Stability
        Rinv = LRinv.dot(LRinv.T) + 1e-9*np.eye(self.dim['m'])
        return Rinv
    @property
    def R(self):
        if self.m == 1:
            R = self.Rinv**-1
        else:
            R = pos_def_mat_inv(self.Rinv)
        return R
    @property
    def m(self):
        return self.dim['m']

class RMixin(object):
    # Mixin for R[0], ..., R[num_states-1] variables
    def _set_dim(self, **kwargs):
        if 'LRinv' in kwargs:
            num_states, m, m2 = np.shape(kwargs['LRinv'])
        else:
            raise ValueError("LRinv not provided")

        if m != m2:
            raise ValueError("LRinv must be square, not {0}".format(
                (m, m2)))
        self._set_check_dim(num_states=num_states, m=m)
        super()._set_dim(**kwargs)
        return

    def _set_var_dict(self, **kwargs):
        if 'LRinv' in kwargs:
            LRinv = np.array(kwargs['LRinv']).astype(float)
            for LRinv_k in LRinv:
                LRinv_k[np.triu_indices_from(LRinv_k, 1)] = 0
            self.var_dict['LRinv'] = LRinv
        else:
            raise ValueError("LRinv not provided")
        super()._set_var_dict(**kwargs)
        return

    @classmethod
    def _from_dict_to_vector(cls, vector_list, var_dict, **kwargs):
        LRinv = var_dict['LRinv']
        vector_list.extend([LRinv_k[np.tril_indices_from(LRinv_k)]
            for LRinv_k in LRinv])
        return super()._from_dict_to_vector(vector_list, var_dict, **kwargs)

    @classmethod
    def _from_vector_to_dict(cls, var_dict, vector, **kwargs):
        num_states, m = kwargs['num_states'], kwargs['m']
        LRinv_vec = vector[:num_states*(m+1)*m//2]
        LRinv = np.zeros((num_states, m, m))
        for k in range(num_states):
            LRinv[k][np.tril_indices(m)] = \
                    LRinv_vec[k*(m+1)*m//2:(k+1)*(m+1)*m//2]
        var_dict['LRinv'] = LRinv
        var_dict = super()._from_vector_to_dict(
                var_dict, vector[num_states*(m+1)*m//2:], **kwargs)
        return var_dict

    def _project_parameters(self, **kwargs):
        for LRinv_k in self.LRinv:
            LRinv_k[np.triu_indices_from(LRinv_k, 1)] = 0

        if kwargs.get('thresh_LRinv', True):
           # Threshold diag(LRinv) to be positive
            LRinv = self.LRinv
            for LRinv_k in LRinv:
                if np.any(np.diag(LRinv_k) < 0.0):
                    logger.info(
                        "Reflecting LRinv: {0} < 0.0".format(LRinv_k)
                        )
                    LRinv_k[:] = np.linalg.cholesky(
                            np.dot(LRinv_k, LRinv_k.T) + np.eye(self.m)*1e-9)
            self.LRinv = LRinv
        if kwargs.get('diag_R', False):
           # Threshold LRinv to be diagonal
           self.LRinv = np.array([
                np.diag(np.diag(LRinv_k)) for LRinv_k in self.LRinv
                ])
        return super()._project_parameters(**kwargs)

    @property
    def LRinv(self):
        LRinv = self.var_dict['LRinv']
        return LRinv
    @LRinv.setter
    def LRinv(self, LRinv):
        self.var_dict['LRinv'] = LRinv
        return
    @property
    def Rinv(self):
        LRinv = self.LRinv
        # 1e-9*np.eye is Fudge Factor for Stability
        Rinv = np.array([LRinv_k.dot(LRinv_k.T) + 1e-9*np.eye(self.dim['m'])
                for LRinv_k in LRinv])
        return Rinv
    @property
    def R(self):
        if self.m == 1:
            R = np.array([Rinv_k**-1 for Rinv_k in self.Rinv])
        else:
            R = np.array([pos_def_mat_inv(Rinv_k) for Rinv_k in self.Rinv])
        return R
    @property
    def m(self):
        return self.dim['m']
    @property
    def num_states(self):
        return self.dim['num_states']

class RSinglePrior(object):
    # Mixin for R Single variable
    def _set_hyperparams(self, **kwargs):
        if 'scale_Rinv' in kwargs:
            m, m2 = np.shape(kwargs['scale_Rinv'])
        else:
            raise ValueError("scale_Rinv must be provided")
        if 'df_Rinv' not in kwargs:
            raise ValueError("df_Rinv must be provided")

        if m != m2:
            raise ValueError("scale_Rinv has wrong shape")

        self._set_check_dim(m=m)

        self.hyperparams['scale_Rinv'] = kwargs['scale_Rinv']
        self.hyperparams['df_Rinv'] = kwargs['df_Rinv']
        super()._set_hyperparams(**kwargs)
        return

    def _sample_prior_var_dict(self, var_dict, **kwargs):
        # Requires Rinvs defined
        scale_Rinv = self.hyperparams['scale_Rinv']
        df_Rinv = self.hyperparams['df_Rinv']

        Rinv = array_wishart_rvs(df=df_Rinv, scale=scale_Rinv)
        LRinv = np.linalg.cholesky(Rinv)

        var_dict['LRinv'] = LRinv
        var_dict = super()._sample_prior_var_dict(var_dict, Rinv=Rinv,
                **kwargs)
        return var_dict

    def _sample_post_var_dict(self, var_dict, sufficient_stat, **kwargs):
        mean, mean_prec, prec = self._get_R_hyperparam()
        scale_Rinv = self.hyperparams['scale_Rinv']
        df_Rinv = self.hyperparams['df_Rinv']

        if len(np.shape(prec)) == 1:
            S_prevprev = \
                prec + sufficient_stat['S_prevprev']
            S_curprev = \
                mean_prec + sufficient_stat['S_curprev']
            S_curcur =  \
                np.outer(mean, mean_prec) + sufficient_stat['S_curcur']
            S_schur = S_curcur - np.outer(S_curprev, S_curprev)/S_prevprev
            df_R = df_Rinv + sufficient_stat['S_count']
            scale_Rinv = \
                np.linalg.inv(np.linalg.inv(scale_Rinv) + S_schur)
            Rinv = array_wishart_rvs(df=df_R, scale=scale_Rinv)
        else:
            S_prevprev = \
                np.diag(prec) + sufficient_stat['S_prevprev']
            S_curprev = \
                mean_prec + sufficient_stat['S_curprev']
            S_curcur =  \
                np.matmul(mean, mean_prec) + sufficient_stat['S_curcur']
            S_schur = S_curcur - np.matmul(S_curprev,
                    np.linalg.solve(S_prevprev, S_curprev.T))
            df_R = df_Rinv + sufficient_stat['S_count']
            scale_Rinv = \
                np.linalg.inv(np.linalg.inv(scale_Rinv) + S_schur)
            Rinv = array_wishart_rvs(df=df_R, scale=scale_Rinv)

        LRinv = np.linalg.cholesky(Rinv)

        var_dict['LRinv'] = LRinv
        var_dict = super()._sample_post_var_dict(
                var_dict, sufficient_stat, Rinv=Rinv, **kwargs)
        return var_dict

    def _logprior(self, logprior, parameters, **kwargs):
        scale_Rinv = self.hyperparams['scale_Rinv']
        df_Rinv = self.hyperparams['df_Rinv']

        logprior += scipy.stats.wishart.logpdf(parameters.Rinv,
            df=df_Rinv, scale=scale_Rinv)
        logprior = super()._logprior(logprior, parameters, **kwargs)
        return logprior

    def _grad_logprior(self, grad, parameters, **kwargs):
        scale_Rinv = self.hyperparams['scale_Rinv']
        df_Rinv = self.hyperparams['df_Rinv']

        LRinv = parameters.LRinv
        grad_LRinv = \
            (df_Rinv - self.dim['m'] - 1) * np.linalg.inv(LRinv.T) - \
            np.linalg.solve(scale_Rinv, LRinv)

        grad['LRinv'] = grad_LRinv
        grad = super()._grad_logprior(grad, parameters, **kwargs)
        return grad

    @classmethod
    def _get_default_kwargs(cls, default_kwargs, **kwargs):
        m = kwargs['m']
        var = kwargs['var']

        Rinv = np.eye(m)
        df_Rinv = np.shape(Rinv)[-1] + 1.0 + var**-1
        scale_Rinv = Rinv/df_Rinv

        default_kwargs['scale_Rinv'] = scale_Rinv
        default_kwargs['df_Rinv'] = df_Rinv

        default_kwargs = super()._get_default_kwargs(default_kwargs, **kwargs)
        return default_kwargs

    @classmethod
    def _get_prior_kwargs(cls, prior_kwargs, parameters, **kwargs):
        var = kwargs['var']
        if kwargs.get('from_mean', False):
            Rinv = parameters.Rinv
        else:
            Rinv = np.eye(parameters.m)
        df_Rinv = np.shape(Rinv)[-1] + 1.0 + var**-1
        scale_Rinv = Rinv/df_Rinv

        prior_kwargs['scale_Rinv'] = scale_Rinv
        prior_kwargs['df_Rinv'] = df_Rinv

        prior_kwargs = super()._get_prior_kwargs(
                prior_kwargs, parameters, **kwargs)
        return prior_kwargs

class RPrior(object):
    # Mixin for R variable
    def _set_hyperparams(self, **kwargs):
        if 'scale_Rinv' in kwargs:
            num_states, m, m2 = np.shape(kwargs['scale_Rinv'])
        else:
            raise ValueError("scale_Rinv must be provided")
        if 'df_Rinv' in kwargs:
            num_states2 = np.shape(kwargs['df_Rinv'])[0]
        else:
            raise ValueError("df_Rinv must be provided")

        if m != m2:
            raise ValueError("scale_Rinv has wrong shape")

        if num_states != num_states2:
            raise ValueError("scale_Rinv and df_Rinv don't match")

        if "num_states" in self.dim:
            if num_states != self.dim['num_states']:
                raise ValueError("num_states do not match existing dims")
        else:
            self.dim['num_states'] = num_states
        if "m" in self.dim:
            if m != self.dim['m']:
                raise ValueError("m do not match existing dims")
        else:
            self.dim['m'] = m

        self.hyperparams['scale_Rinv'] = kwargs['scale_Rinv']
        self.hyperparams['df_Rinv'] = kwargs['df_Rinv']
        super()._set_hyperparams(**kwargs)
        return

    def _sample_prior_var_dict(self, var_dict, **kwargs):
        # Requires Rinvs defined
        scale_Rinv = self.hyperparams['scale_Rinv']
        df_Rinv = self.hyperparams['df_Rinv']

        Rinvs = [
                array_wishart_rvs(df=df_Rinv_k, scale=scale_Rinv_k)
                for df_Rinv_k, scale_Rinv_k in zip(df_Rinv, scale_Rinv)
                ]
        LRinv = np.array([np.linalg.cholesky(Rinv_k) for Rinv_k in Rinvs])

        var_dict['LRinv'] = LRinv
        var_dict = super()._sample_prior_var_dict(var_dict, Rinvs=Rinvs,
                **kwargs)
        return var_dict

    def _sample_post_var_dict(self, var_dict, sufficient_stat, **kwargs):
        mean, var_col = self._get_R_hyperparam_mean_var_col()
        scale_Rinv = self.hyperparams['scale_Rinv']
        df_Rinv = self.hyperparams['df_Rinv']

        Rinvs = [None for _ in range(self.dim['num_states'])]
        if len(np.shape(var_col)) == 1:
            for k in range(0, self.dim['num_states']):
                S_prevprev = \
                    var_col[k]**-1 + sufficient_stat['S_prevprev'][k]
                S_curprev = \
                     var_col[k]**-1 * mean[k] + sufficient_stat['S_curprev'][k]
                S_curcur =  \
                    np.outer(mean[k], var_col[k]**-1 * mean[k]) + \
                sufficient_stat['S_curcur'][k]
                S_schur = S_curcur - np.outer(S_curprev, S_curprev)/S_prevprev
                df_R_k = df_Rinv[k] + sufficient_stat['S_count'][k]
                scale_Rinv_k = \
                    np.linalg.inv(np.linalg.inv(scale_Rinv[k]) + S_schur)
                Rinvs[k] = array_wishart_rvs(df=df_R_k, scale=scale_Rinv_k)
        else:
            for k in range(0, self.dim['num_states']):
                S_prevprev = \
                    np.diag(var_col[k]**-1) + sufficient_stat['S_prevprev'][k]
                S_curprev = \
                    var_col[k]**-1 * mean[k] + sufficient_stat['S_curprev'][k]
                S_curcur =  \
                    np.matmul(mean[k], (var_col[k]**-1 * mean[k]).T) + \
                sufficient_stat['S_curcur'][k]
                S_schur = S_curcur - np.matmul(S_curprev,
                        np.linalg.solve(S_prevprev, S_curprev.T))
                df_R_k = df_Rinv[k] + sufficient_stat['S_count'][k]
                scale_Rinv_k = \
                        np.linalg.inv(np.linalg.inv(scale_Rinv[k]) + S_schur)
                Rinvs[k] = array_wishart_rvs(df=df_R_k, scale=scale_Rinv_k)

        Rinvs = np.array(Rinvs)
        LRinv = np.array([np.linalg.cholesky(Rinv_k) for Rinv_k in Rinvs])

        var_dict['LRinv'] = LRinv
        var_dict = super()._sample_post_var_dict(
                var_dict, sufficient_stat, Rinvs=Rinvs, **kwargs)
        return var_dict

    def _logprior(self, logprior, parameters, **kwargs):
        scale_Rinv = self.hyperparams['scale_Rinv']
        df_Rinv = self.hyperparams['df_Rinv']

        for Rinv_k, df_Rinv_k, scale_Rinv_k in zip(
                parameters.Rinv, df_Rinv, scale_Rinv):
            logprior += scipy.stats.wishart.logpdf(Rinv_k,
                df=df_Rinv_k, scale=scale_Rinv_k)
        logprior = super()._logprior(logprior, parameters, **kwargs)
        return logprior

    def _grad_logprior(self, grad, parameters, **kwargs):
        scale_Rinv = self.hyperparams['scale_Rinv']
        df_Rinv = self.hyperparams['df_Rinv']

        grad_LRinv = np.array([
            (df_Rinv_k - self.dim['m'] - 1) * np.linalg.inv(LRinv_k.T) - \
            np.linalg.solve(scale_Rinv_k, LRinv_k)
            for LRinv_k, df_Rinv_k, scale_Rinv_k in zip(
                parameters.LRinv, df_Rinv, scale_Rinv)
            ])

        grad['LRinv'] = grad_LRinv
        grad = super()._grad_logprior(grad, parameters, **kwargs)
        return grad

    @classmethod
    def _get_default_kwargs(cls, default_kwargs, **kwargs):
        num_states = kwargs['num_states']
        m = kwargs['m']
        var = kwargs['var']

        Rinv = np.array([np.eye(m) for _ in range(num_states)])
        df_Rinv = np.shape(Rinv)[-1] + 1.0 + var**-1
        scale_Rinv = Rinv/df_Rinv
        df_Rinv = np.array([df_Rinv+0 for k in range(num_states)])

        default_kwargs['scale_Rinv'] = scale_Rinv
        default_kwargs['df_Rinv'] = df_Rinv

        default_kwargs = super()._get_default_kwargs(default_kwargs, **kwargs)
        return default_kwargs

    @classmethod
    def _get_prior_kwargs(cls, prior_kwargs, parameters, **kwargs):
        var = kwargs['var']
        if kwargs.get('from_mean', False):
            Rinv = parameters.Rinv
        else:
            Rinv = np.array([
                np.eye(parameters.m) for _ in range(parameters.num_states)
                ])
        df_Rinv = np.shape(Rinv)[-1] + 1.0 + var**-1
        scale_Rinv = Rinv/df_Rinv
        df_Rinv = np.array([df_Rinv+0 for k in range(parameters.num_states)])

        prior_kwargs['scale_Rinv'] = scale_Rinv
        prior_kwargs['df_Rinv'] = df_Rinv

        prior_kwargs = super()._get_prior_kwargs(
                prior_kwargs, parameters, **kwargs)
        return prior_kwargs

class RSinglePreconditioner(object):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def _precondition(self, precond_grad, grad, parameters, **kwargs):
        Rinv = parameters.Rinv
        grad['LRinv'][np.triu_indices_from(grad['LRinv'], 1)] = 0
        precond_grad['LRinv'] = np.dot(0.5*Rinv, grad['LRinv'])
        precond_grad = super()._precondition(precond_grad, grad,
                parameters, **kwargs)
        return precond_grad

    def _precondition_noise(self, noise, parameters, **kwargs):
        LRinv = parameters.LRinv
        precond_LRinv = np.dot(np.sqrt(0.5)*LRinv,
                np.random.normal(loc=0, size=(parameters.m, parameters.m)),
                )
        precond_LRinv[np.triu_indices_from(precond_LRinv, 1)] = 0
        noise['LRinv'] = precond_LRinv
        super()._precondition_noise(noise, parameters, **kwargs)
        return noise

    def _correction_term(self, correction, parameters, **kwargs):
        correction['LRinv'] = 0.5*(parameters.m + 1) * parameters.LRinv
        super()._correction_term(correction, parameters, **kwargs)
        return correction

class RPreconditioner(object):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def _precondition(self, precond_grad, grad, parameters, **kwargs):
        Rinv = parameters.Rinv
        for k in range(parameters.num_states):
            grad['LRinv'][k][np.triu_indices_from(grad['LRinv'][k], 1)] = 0
        precond_LRinv = np.array([
            np.dot(0.5*Rinv[k], grad['LRinv'][k])
            for k in range(parameters.num_states)
            ])

        precond_grad['LRinv'] = precond_LRinv
        precond_grad = super()._precondition(precond_grad, grad,
                parameters, **kwargs)
        return precond_grad

    def _precondition_noise(self, noise, parameters, **kwargs):
        LRinv = parameters.LRinv
        precond_LRinv = np.array([
            np.dot(np.sqrt(0.5)*LRinv[k],
                np.random.normal(loc=0, size=(parameters.m, parameters.m)),
                )
            for k in range(parameters.num_states)
            ])
        for precond_LRinv_k in precond_LRinv:
            precond_LRinv_k[np.triu_indices_from(precond_LRinv_k, 1)] = 0

        noise['LRinv'] = precond_LRinv
        super()._precondition_noise(noise, parameters, **kwargs)
        return noise

    def _correction_term(self, correction, parameters, **kwargs):
        correction['LRinv'] = 0.5*(parameters.m + 1) * parameters.LRinv
        super()._correction_term(correction, parameters, **kwargs)
        return correction

