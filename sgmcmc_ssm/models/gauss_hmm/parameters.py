import numpy as np
import logging
logger = logging.getLogger(name=__name__)

from ...base_parameters import (
        BaseParameters, BasePrior, BasePreconditioner,
        )
from ...variables import (
        TransitionMatrixParamHelper, TransitionMatrixPriorHelper,
        TransitionMatrixPrecondHelper,
        VectorsParamHelper, VectorsPriorHelper,
        VectorsPrecondHelper,
        CovariancesParamHelper, CovariancesPriorHelper,
        CovariancesPrecondHelper,
        )

class GaussHMMParameters(BaseParameters):
    """ Gaussian HMM Parameters """
    _param_helper_list = [
            TransitionMatrixParamHelper(name='pi', dim_names=['num_states', 'pi_type']),
            VectorsParamHelper(name='mu', dim_names=['m', 'num_states']),
            CovariancesParamHelper(name='R', dim_names=['m', 'num_states']),
            ]
    for param_helper in _param_helper_list:
        properties = param_helper.get_properties()
        for name, prop in properties.items():
            vars()[name] = prop

    def __str__(self):
        my_str = "GaussHMMParameters:"
        my_str += "\npi:\n" + str(self.pi)
        my_str += "\npi_type: `" + str(self.pi_type) + "`"
        my_str += "\nmu:\n" + str(self.mu)
        my_str += "\nR:\n" + str(self.R)
        return my_str

class GaussHMMPrior(BasePrior):
    """ Gaussian HMM Prior
    See individual Prior Mixins for details
    """
    _Parameters = GaussHMMParameters
    _prior_helper_list = [
            CovariancesPriorHelper(name='R', dim_names=['m', 'num_states'], matrix_name='mu'),
            TransitionMatrixPriorHelper(name='pi', dim_names=['num_states']),
            VectorsPriorHelper(name='mu', dim_names=['m', 'num_states'],
                var_row_name='R'),
            ]

class GaussHMMPreconditioner(BasePreconditioner):
    """ Gaussian HMM Preconditioner
    See individual Precondition Mixins for details
    """
    _precond_helper_list = [
            TransitionMatrixPrecondHelper(name='pi', dim_names=['num_states']),
            VectorsPrecondHelper(name='mu', dim_names=['m', 'num_states'], var_row_name='R'),
            CovariancesPrecondHelper(name='R', dim_names=['m', 'num_states']),
            ]

def generate_gausshmm_data(T, parameters, initial_message = None,
        tqdm=None):
    """ Helper function for generating Gaussian HMM time series

    Args:
        T (int): length of series
        parameters (GAUSSHMMParameters): parameters
        initial_message (ndarray): prior for u_{-1}

    Returns:
        data (dict): dictionary containing:
            observations (ndarray): T by m
            latent_vars (ndarray): T takes values in {1,...,num_states}
            parameters (GaussHMMParameters)
            init_message (ndarray)
    """
    from ..._utils import random_categorical
    k, m = np.shape(parameters.mu)
    mu = parameters.mu
    R = parameters.R
    Pi = parameters.pi

    if initial_message is None:
        initial_message = {
                'prob_vector': np.ones(k)/k,
                'log_constant': 0.0,
                }

    latent_vars = np.zeros((T), dtype=int)
    obs_vars = np.zeros((T, m))
    latent_prev = random_categorical(initial_message['prob_vector'])
    pbar = range(T)
    if tqdm is not None:
        pbar = tqdm(pbar)
        pbar.set_description("generating data")
    for t in pbar:
        latent_vars[t] = random_categorical(Pi[latent_prev])
        mu_k = mu[latent_vars[t]]
        R_k = R[latent_vars[t]]
        obs_vars[t] = np.random.multivariate_normal(mean=mu_k, cov = R_k)
        latent_prev = latent_vars[t]

    data = dict(
            observations=obs_vars,
            latent_vars=latent_vars,
            parameters=parameters,
            initial_message=initial_message,
            )
    return data


