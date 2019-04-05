import numpy as np
import scipy.stats
import logging
from .._utils import (
        matrix_normal_logpdf,
        pos_def_mat_inv,
        )
from scipy.special import expit, logit
logger = logging.getLogger(name=__name__)

# Seasonal Regression Variable (for Logistic Regression)
class GARCHMixin(object):
    # alpha, beta, gamma <-> mu, phi, lambduh
    @staticmethod
    def convert_alpha_beta_gamma(alpha, beta, gamma):
        """ Convert alpha, beta, gamma to log_mu, logit_phi, logit_lambduh

        mu = alpha / (1- beta - gamma)
        phi = beta + gamma
        lambda = beta / (beta + gamma)
        """
        if alpha <= 0 or beta <= 0 or gamma <= 0:
            raise ValueError("Cannot have alpha, beta, or gamma <= 0")
        if beta + gamma >=1:
            raise ValueError("Cannot have beta + gamma >- 1")
        log_mu = np.log(alpha/(1 - beta - gamma))
        logit_phi = logit(beta + gamma)
        logit_lambduh = logit(beta/(beta + gamma))
        return log_mu, logit_phi, logit_lambduh

    def _set_var_dict(self, **kwargs):
        if 'log_mu' in kwargs:
            self.var_dict['log_mu'] = np.array(kwargs['log_mu']).astype(float)
        else:
            raise ValueError("logit_mu not provided")
        if 'logit_phi' in kwargs:
            self.var_dict['logit_phi'] = \
                    np.array(kwargs['logit_phi']).astype(float)
        else:
            raise ValueError("logit_phi not provided")
        if 'logit_lambduh' in kwargs:
            self.var_dict['logit_lambduh'] = \
                np.array(kwargs['logit_lambduh']).astype(float)
        else:
            raise ValueError("logit_lambduh not provided")
        super()._set_var_dict(**kwargs)
        return

    def _project_parameters(self, **kwargs):
        if kwargs.get('thresh_phi', False):
            logit_phi = self.logit_phi
            if logit_phi > 2.95:
                #logger.info("Thresholding phi: {0} > 0.95".format(self.phi))
                self.logit_phi = logit_phi*2.95/logit_phi
        return super()._project_parameters(**kwargs)


    @classmethod
    def _from_dict_to_vector(cls, vector_list, var_dict, **kwargs):
        vector_list.append(var_dict['log_mu'].flatten())
        vector_list.append(var_dict['logit_phi'].flatten())
        vector_list.append(var_dict['logit_lambduh'].flatten())
        return super()._from_dict_to_vector(vector_list, var_dict, **kwargs)

    @classmethod
    def _from_vector_to_dict(cls, var_dict, vector, **kwargs):
        var_dict['log_mu'] = vector[0]
        var_dict['logit_phi'] = vector[1]
        var_dict['logit_lambduh'] = vector[2]
        var_dict = super()._from_vector_to_dict(
                var_dict, vector[2:], **kwargs)
        return var_dict

    @property
    def log_mu(self):
        log_mu = self.var_dict['log_mu']
        return log_mu
    @log_mu.setter
    def log_mu(self, log_mu):
        self.var_dict['log_mu'] = log_mu
        return
    @property
    def mu(self):
        mu = np.exp(self.var_dict['log_mu'])
        return mu

    @property
    def logit_phi(self):
        logit_phi = self.var_dict['logit_phi']
        return logit_phi
    @logit_phi.setter
    def logit_phi(self, logit_phi):
        self.var_dict['logit_phi'] = logit_phi
        return
    @property
    def phi(self):
        phi = expit(self.var_dict['logit_phi'])
        return phi

    @property
    def logit_lambduh(self):
        logit_lambduh = self.var_dict['logit_lambduh']
        return logit_lambduh
    @logit_lambduh.setter
    def logit_lambduh(self, logit_lambduh):
        self.var_dict['logit_lambduh'] = logit_lambduh
        return
    @property
    def lambduh(self):
        lambduh = expit(self.var_dict['logit_lambduh'])
        return lambduh

    @property
    def alpha(self):
        alpha = self.mu * (1-self.phi)
        return alpha
    @property
    def beta(self):
        beta = self.phi * self.lambduh
        return beta
    @property
    def gamma(self):
        gamma = self.phi * (1-self.lambduh)
        return gamma

class GARCHMixinPrior(object):
    # beta distribution for (phi + 1)/2, (lambduh + 1)/2,
    # inv-gamma distribution for mu
    def _set_hyperparams(self, **kwargs):
        hyperparams_list = [
                'scale_mu', 'shape_mu',
                'alpha_phi', 'beta_phi',
                'alpha_lambduh', 'beta_lambduh',
                ]
        for hyperparam in hyperparams_list:
            if hyperparam not in kwargs:
                raise ValueError("{} must be provided".format(hyperparam))
            self.hyperparams[hyperparam] = kwargs[hyperparam]
        super()._set_hyperparams(**kwargs)
        return

    def _sample_prior_var_dict(self, var_dict, **kwargs):
        mu = scipy.stats.invgamma(
                a=self.hyperparams['shape_mu'],
                scale=self.hyperparams['scale_mu']
                ).rvs()
        var_dict['log_mu'] = np.log(mu)
        phi = scipy.stats.beta(
                a=self.hyperparams['alpha_phi'],
                b=self.hyperparams['beta_phi'],
                ).rvs()
        var_dict['logit_phi'] = logit(phi)
        lambduh = scipy.stats.beta(
                a=self.hyperparams['alpha_lambduh'],
                b=self.hyperparams['beta_lambduh'],
                ).rvs()
        var_dict['logit_lambduh'] = logit(lambduh)
        var_dict = super()._sample_prior_var_dict(var_dict, **kwargs)
        return var_dict

    def _sample_post_var_dict(self, var_dict, sufficient_stat, **kwargs):
        raise NotImplementedError()

    def _logprior(self, logprior, parameters, **kwargs):
        logprior += scipy.stats.invgamma(
                a=self.hyperparams['shape_mu'],
                scale=self.hyperparams['scale_mu']
                ).logpdf(parameters.mu)
        logprior += scipy.stats.beta(
                a=self.hyperparams['alpha_phi'],
                b=self.hyperparams['beta_phi']
                ).logpdf((1+parameters.phi)/2.0)
        logprior += scipy.stats.beta(
                a=self.hyperparams['alpha_lambduh'],
                b=self.hyperparams['beta_lambduh']
                ).logpdf((1+parameters.lambduh)/2.0)
        logprior = super()._logprior(logprior, parameters, **kwargs)
        return logprior

    def _grad_logprior(self, grad, parameters, **kwargs):

        grad['log_mu'] = - self.hyperparams['shape_mu'] - 1 + \
                self.hyperparams['scale_mu'] / parameters.mu

        grad['logit_phi'] = (
                (self.hyperparams['alpha_phi'] - 1) / (1 + parameters.phi) -
                (self.hyperparams['beta_phi'] - 1) / (1 - parameters.phi)
                ) * parameters.phi * (1-parameters.phi)

        grad['logit_lambduh'] = (
                (self.hyperparams['alpha_lambduh'] - 1) / (1 + parameters.lambduh) -
                (self.hyperparams['beta_lambduh'] - 1) / (1 - parameters.lambduh)
                ) * parameters.lambduh * (1-parameters.lambduh)

        grad = super()._grad_logprior(grad, parameters, **kwargs)
        return grad

    @classmethod
    def _get_default_kwargs(cls, default_kwargs, **kwargs):
        var = kwargs['var']
        if var > 1:
            var = 1
        default_kwargs['scale_mu'] = var + 2
        default_kwargs['shape_mu'] = default_kwargs['scale_mu'] + 1
        default_kwargs['alpha_phi'] = 1 + 19*var**-1
        default_kwargs['beta_phi'] = default_kwargs['alpha_phi'] / 9
        default_kwargs['alpha_lambduh'] = 1 + 19*var**-1
        default_kwargs['beta_lambduh'] = default_kwargs['alpha_lambduh'] / 9
        default_kwargs = super()._get_default_kwargs(default_kwargs, **kwargs)
        return default_kwargs

    @classmethod
    def _get_prior_kwargs(cls, prior_kwargs, parameters, **kwargs):
        var = kwargs['var']
        if var > 1:
            var = 1
        prior_kwargs['scale_mu'] = var + 2
        prior_kwargs['shape_mu'] = prior_kwargs['scale_mu'] + 1
        prior_kwargs['alpha_phi'] = 10 + var
        prior_kwargs['beta_phi'] = prior_kwargs['alpha_phi'] / 9
        prior_kwargs['alpha_lambduh'] = 10 + var
        prior_kwargs['beta_lambduh'] = prior_kwargs['alpha_lambduh'] / 9
        prior_kwargs = super()._get_prior_kwargs(
                prior_kwargs, parameters, **kwargs)
        return prior_kwargs


