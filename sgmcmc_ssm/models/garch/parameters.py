import numpy as np
import logging
logger = logging.getLogger(name=__name__)

from ...base_parameters import (
        BaseParameters, BasePrior, BasePreconditioner,
        )
from ...variables import (
        CovarianceParamHelper, CovariancePriorHelper,
        )
from ...variables.garch_var import (
        GARCHParamHelper, GARCHPriorHelper,
        )

from ..._utils import var_stationary_precision
from scipy.special import logit


class GARCHParameters(BaseParameters):
    """ GARCH Parameters """
    _param_helper_list = [
            GARCHParamHelper(),
            CovarianceParamHelper(name='R', dim_names=['m']),
            ]
    for param_helper in _param_helper_list:
        properties = param_helper.get_properties()
        for name, prop in properties.items():
            vars()[name] = prop

    def __str__(self):
        my_str = "GARCHParameters:"
        my_str +="\nalpha:{0}, beta:{1}, gamma:{2}, tau:{3}\n".format(
                np.around(np.asscalar(self.alpha), 6),
                np.around(np.asscalar(self.beta), 6),
                np.around(np.asscalar(self.gamma), 6),
                np.around(np.asscalar(self.tau), 6))
        return my_str

    @property
    def tau(self):
        if self.m == 1:
            tau = self.var_dict['LRinv'] ** -1
        else:
            tau = np.linalg.inv(self.var_dict['LRinv'].T)
        return tau

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

class GARCHPrior(BasePrior):
    """ GARCH Prior
    See individual Prior Mixins for details
    """
    _Parameters = GARCHParameters
    _prior_helper_list = [
            GARCHPriorHelper(),
            CovariancePriorHelper(name='R', dim_names=['m']),
            ]

def generate_garch_data(T, parameters, initial_message = None,
        tqdm=None):
    """ Helper function for generating GARCH time series

    Args:
        T (int): length of series
        parameters (GARCHParameters): parameters
        initial_message (ndarray): prior for u_{-1}

    Returns:
        data (dict): dictionary containing:
            observations (ndarray): T by m
            latent_vars (ndarray): T by n
            parameters (GARCHParameters)
            init_message (ndarray)
    """
    n = 1
    m = 1
    alpha = parameters.alpha
    beta = parameters.beta
    gamma = parameters.gamma
    R = parameters.R

    if initial_message is None:
        init_precision =  np.atleast_2d((1 - beta - gamma)/alpha)
        initial_message = {
                'log_constant': 0.0,
                'mean_precision': np.zeros(n),
                'precision': init_precision,
                }

    latent_vars = np.zeros((T, n), dtype=float)
    sigma2s = np.zeros((T), dtype=float)
    obs_vars = np.zeros((T, m), dtype=float)
    latent_prev = np.random.multivariate_normal(
            mean=np.linalg.solve(initial_message['precision'],
                initial_message['mean_precision']),
            cov=np.linalg.inv(initial_message['precision']),
            )

    pbar = range(T)
    if tqdm is not None:
        pbar = tqdm(pbar)
        pbar.set_description("generating data")
    sigma2_prev = 0
    for t in pbar:
        sigma2s[t] = alpha + beta*latent_prev**2 + gamma*sigma2_prev
        latent_vars[t] = np.random.multivariate_normal(
                mean=np.zeros(1),
                cov=np.array([[sigma2s[t]]]),
                )
        obs_vars[t] = np.random.multivariate_normal(
                mean=latent_vars[t],
                cov=R,
                )
        latent_prev = latent_vars[t]
        sigma2_prev = sigma2s[t]

    data = dict(
            observations=obs_vars,
            latent_vars=latent_vars,
            sigma2s=sigma2s,
            parameters=parameters,
            initial_message=initial_message,
            )
    return data


