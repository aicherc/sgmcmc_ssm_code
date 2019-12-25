import numpy as np
import logging
logger = logging.getLogger(name=__name__)

from ...base_parameters import (
        BaseParameters, BasePrior, BasePreconditioner,
        )
from ...variables import (
        SquareMatrixParamHelper, SquareMatrixPriorHelper,
        SquareMatrixPrecondHelper,
        RectMatrixParamHelper, RectMatrixPriorHelper,
        RectMatrixPrecondHelper,
        CovarianceParamHelper, CovariancePriorHelper,
        CovariancePrecondHelper,
        )
from ..._utils import var_stationary_precision


class SVMParameters(BaseParameters):
    """ SVM Parameters """
    _param_helper_list = [
            SquareMatrixParamHelper(name='A', dim_names=['n']),
            CovarianceParamHelper(name='Q', dim_names=['n']),
            CovarianceParamHelper(name='R', dim_names=['m']),
            ]
    for param_helper in _param_helper_list:
        properties = param_helper.get_properties()
        for name, prop in properties.items():
            vars()[name] = prop

    def __str__(self):
        my_str = "SVMParameters:"
        if self.n == 1:
            my_str +="\nA:{0}, Q:{1}, R:{2}\n".format(
                    self.A[0,0], self.Q[0,0], self.R[0,0])
        else:
            my_str += "\nA:\n" + str(self.A)
            my_str += "\nQ:\n" + str(self.Q)
            my_str += "\nR:\n" + str(self.R)
        return my_str

    @property
    def phi(self):
        phi = self.var_dict['A']
        return phi

    @property
    def sigma(self):
        if self.n == 1:
            sigma = self.var_dict['LQinv'] ** -1
        else:
            sigma = np.linalg.inv(self.var_dict['LQinv'].T)
        return sigma

    @property
    def tau(self):
        if self.m == 1:
            tau = self.var_dict['LRinv'] ** -1
        else:
            tau = np.linalg.inv(self.var_dict['LRinv'].T)
        return tau

class SVMPrior(BasePrior):
    """ SVM Prior
    See individual Prior Mixins for details
    """
    _Parameters = SVMParameters
    _prior_helper_list = [
            CovariancePriorHelper(name='Q', dim_names=['n'], matrix_name='A'),
            CovariancePriorHelper(name='R', dim_names=['m']),
            SquareMatrixPriorHelper(name='A', dim_names=['n'],
                var_row_name='Q'),
            ]

def generate_svm_data(T, parameters, initial_message = None,
        tqdm=None):
    """ Helper function for generating SVM time series

    Args:
        T (int): length of series
        parameters (SVMParameters): parameters
        initial_message (ndarray): prior for u_{-1}

    Returns:
        data (dict): dictionary containing:
            observations (ndarray): T by m
            latent_vars (ndarray): T by n
            parameters (SVMParameters)
            init_message (ndarray)
    """
    n, _ = np.shape(parameters.A)
    m, _ = np.shape(parameters.R)
    A = parameters.A
    Q = parameters.Q
    R = parameters.R

    if initial_message is None:
        init_precision = var_stationary_precision(
                parameters.Qinv, parameters.A, 10)
        initial_message = {
                'log_constant': 0.0,
                'mean_precision': np.zeros(n),
                'precision': init_precision,
                }

    latent_vars = np.zeros((T, n), dtype=float)
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
    for t in pbar:
        latent_vars[t] = np.random.multivariate_normal(
                mean=np.dot(A, latent_prev),
                cov=Q,
                )
        obs_vars[t] = np.random.multivariate_normal(
                mean=np.zeros(1),
                cov=np.exp(latent_vars[t])*R,
                )
        latent_prev = latent_vars[t]

    data = dict(
            observations=obs_vars,
            latent_vars=latent_vars,
            parameters=parameters,
            initial_message=initial_message,
            )
    return data


