import numpy as np
import logging
from .._utils import normal_logpdf
logger = logging.getLogger(name=__name__)


class MuMixin(object):
    # Mixin for Mu[0], ..., Mu[num_states-1] variables
    def _set_dim(self, **kwargs):
        if 'mu' in kwargs:
            num_states, m = np.shape(kwargs['mu'])
        else:
            raise ValueError("mu not provided")

        if "num_states" in self.dim:
            if num_states != self.dim['num_states']:
                raise ValueError("mu.shape[0] does not match existing dims")
        else:
            self.dim['num_states'] = num_states
        if "m" in self.dim:
            if m != self.dim['m']:
                raise ValueError("mu.shape[1] does not match existing dims")
        else:
            self.dim['m'] = m
        super()._set_dim(**kwargs)
        return

    def _set_var_dict(self, **kwargs):
        if 'mu' in kwargs:
            self.var_dict['mu'] = np.array(kwargs['mu']).astype(float)
        else:
            raise ValueError("mu not provided")

        super()._set_var_dict(**kwargs)
        return

    @classmethod
    def _from_dict_to_vector(cls, vector_list, var_dict, **kwargs):
        vector_list.append(var_dict['mu'].flatten())
        return super()._from_dict_to_vector(vector_list, var_dict, **kwargs)

    @classmethod
    def _from_vector_to_dict(cls, var_dict, vector, **kwargs):
        num_states, m = kwargs['num_states'], kwargs['m']
        mu = np.reshape(vector[0:num_states*m], (num_states, m))
        var_dict['mu'] = mu
        var_dict = super()._from_vector_to_dict(
                var_dict, vector[num_states*m:], **kwargs)
        return var_dict

    @property
    def mu(self):
        mu = self.var_dict['mu']
        return mu
    @mu.setter
    def mu(self, mu):
        self.var_dict['mu'] = mu
        return

    @property
    def num_states(self):
        return self.dim['num_states']
    @property
    def m(self):
        return self.dim['m']


class MuPrior(object):
    # Mixin for Mu variable
    def _set_hyperparams(self, **kwargs):
        if 'mean_mu' in kwargs:
            num_states, m = np.shape(kwargs['mean_mu'])
        else:
            raise ValueError("mean_mu must be provided")
        if 'var_col_mu' in kwargs:
            num_states2 = np.shape(kwargs['var_col_mu'])[0]
        else:
            raise ValueError("mean_mu must be provided")

        if num_states != num_states2:
            raise ValueError("mean_mu + var_col_mu don't match")

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

        self.hyperparams['mean_mu'] = kwargs['mean_mu']
        self.hyperparams['var_col_mu'] = kwargs['var_col_mu']
        super()._set_hyperparams(**kwargs)
        return

    def _sample_prior_var_dict(self, var_dict, **kwargs):
        # Requires Rinvs defined
        mean_mu = self.hyperparams['mean_mu']
        var_col_mu = self.hyperparams['var_col_mu']
        if "Rinvs" in kwargs:
            Rinvs = kwargs['Rinvs']
        elif "Rinv" in kwargs:
            Rinvs = np.array([kwargs['Rinv']
                for _ in range(self.dim['num_states'])])
        else:
            raise ValueError("Missing Covariance")

        mus = [None for k in range(self.dim['num_states'])]
        for k in range(len(mus)):
            mu_k = np.random.multivariate_normal(
                    mean=mean_mu[k],
                    cov=var_col_mu[k]*np.linalg.inv(Rinvs[k]),
                    )

            mus[k] = mu_k
        var_dict['mu'] = np.array(mus)
        var_dict = super()._sample_prior_var_dict(var_dict, **kwargs)
        return var_dict

    def _sample_post_var_dict(self, var_dict, sufficient_stat, **kwargs):
        # Requires Rinvs defined
        mean_mu = self.hyperparams['mean_mu']
        var_col_mu = self.hyperparams['var_col_mu']
        if "Rinvs" in kwargs:
            Rinvs = kwargs['Rinvs']
        elif "Rinv" in kwargs:
            Rinvs = np.array([kwargs['Rinv']
                for _ in range(self.dim['num_states'])])
        else:
            raise ValueError("Missing Covariance")

        mus = [None for k in range(self.dim['num_states'])]
        for k in range(0, self.dim['num_states']):
            S_prevprev = var_col_mu[k]**-1 + sufficient_stat['S_prevprev'][k]
            S_curprev = \
                var_col_mu[k]**-1*mean_mu[k] + sufficient_stat['S_curprev'][k]
            post_mean_mu_k = S_curprev/S_prevprev
            mu_k = np.random.multivariate_normal(
                    mean=post_mean_mu_k,
                    cov=np.linalg.inv(Rinvs[k])/S_prevprev,
                    )

            mus[k] = mu_k
        var_dict['mu'] = np.array(mus)
        var_dict = super()._sample_post_var_dict(
                var_dict, sufficient_stat, **kwargs)
        return var_dict

    def _logprior(self, logprior, parameters, **kwargs):
        mean_mu = self.hyperparams['mean_mu']
        var_col_mu = self.hyperparams['var_col_mu']
        for mu_k, mean_mu_k, var_col_mu_k, LRinv_k in zip(parameters.mu,
                mean_mu, var_col_mu, parameters.LRinv):
            logprior += normal_logpdf(mu_k,
                    mean=mean_mu_k,
                    Lprec=var_col_mu_k**-0.5 * LRinv_k,
                    )

        logprior = super()._logprior(logprior, parameters, **kwargs)
        return logprior

    def _grad_logprior(self, grad, parameters, **kwargs):
        mean_mu = self.hyperparams['mean_mu']
        var_col_mu = self.hyperparams['var_col_mu']
        Rinv = parameters.Rinv
        grad_mu = np.array([
            -1.0*np.dot(var_col_mu[k]**-1 * Rinv[k],
                    parameters.mu[k] - mean_mu[k])
            for k in range(self.dim['num_states'])
            ])
        grad['mu'] = grad_mu
        grad = super()._grad_logprior(grad, parameters, **kwargs)
        return grad

    @classmethod
    def _get_default_kwargs(cls, default_kwargs, **kwargs):
        num_states = kwargs['num_states']
        m = kwargs['m']
        var = kwargs['var']

        default_kwargs['mean_mu'] = np.zeros((num_states, m), dtype=float)
        default_kwargs['var_col_mu'] = np.array([
            var for _ in range(num_states)
            ], dtype=float)

        default_kwargs = super()._get_default_kwargs(default_kwargs, **kwargs)
        return default_kwargs

    @classmethod
    def _get_prior_kwargs(cls, prior_kwargs, parameters, **kwargs):
        var = kwargs['var']
        if kwargs.get('from_mean', False):
            mean_mu = parameters.mu
        else:
            mean_mu = np.zeros_like(parameters.mu, dtype=float)
        var_col_mu = np.array([
            var*1.0 for _ in range(parameters.num_states)
            ], dtype=float)

        prior_kwargs['mean_mu'] = mean_mu
        prior_kwargs['var_col_mu'] = var_col_mu
        prior_kwargs = super()._get_prior_kwargs(
                prior_kwargs, parameters, **kwargs)
        return prior_kwargs

    def _get_R_hyperparam_mean_var_col(self):
        return self.hyperparams['mean_mu'], self.hyperparams['var_col_mu']


class MuPreconditioner(object):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def _precondition(self, precond_grad, grad, parameters, **kwargs):
        R = parameters.R
        precond_mu = np.array([
            np.dot(R[k], grad['mu'][k])
            for k in range(parameters.num_states)
            ])
        precond_grad['mu'] = precond_mu
        precond_grad = super()._precondition(precond_grad, grad,
                parameters, **kwargs)
        return precond_grad

    def _precondition_noise(self, noise, parameters, **kwargs):
        LRinv = parameters.LRinv
        precond_mu = np.array([
            np.linalg.solve(LRinv[k].T,
                np.random.normal(loc=0, size=(parameters.m)),
                )
            for k in range(parameters.num_states)
            ])
        noise['mu'] = precond_mu
        super()._precondition_noise(noise, parameters, **kwargs)
        return noise

    def _correction_term(self, correction, parameters, **kwargs):
        correction['mu'] = np.zeros_like(parameters.mu, dtype=float)
        super()._correction_term(correction, parameters, **kwargs)
        return correction

