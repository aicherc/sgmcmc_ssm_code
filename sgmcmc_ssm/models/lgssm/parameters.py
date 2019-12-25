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

class LGSSMParameters(BaseParameters):
    """ LGSSM Parameters """
    _param_helper_list = [
            SquareMatrixParamHelper(name='A', dim_names=['n']),
            RectMatrixParamHelper(name='C', dim_names=['m', 'n']),
            CovarianceParamHelper(name='Q', dim_names=['n']),
            CovarianceParamHelper(name='R', dim_names=['m']),
            ]
    for param_helper in _param_helper_list:
        properties = param_helper.get_properties()
        for name, prop in properties.items():
            vars()[name] = prop

    def __str__(self):
        my_str = "LGSSMParameters:"
        my_str += "\nA:\n" + str(self.A)
        my_str += "\nC:\n" + str(self.C)
        my_str += "\nQ:\n" + str(self.Q)
        my_str += "\nR:\n" + str(self.R)
        return my_str

    def project_parameters(self, **kwargs):
        if 'C' not in kwargs:
            kwargs['C'] = dict(fixed_eye=True)
        return super().project_parameters(**kwargs)

class LGSSMPrior(BasePrior):
    """ LGSSM Prior
    See individual Prior Mixins for details
    """
    _Parameters = LGSSMParameters
    _prior_helper_list = [
            CovariancePriorHelper(name='Q', dim_names=['n'], matrix_name='A'),
            CovariancePriorHelper(name='R', dim_names=['m'], matrix_name='C'),
            SquareMatrixPriorHelper(name='A', dim_names=['n'],
                var_row_name='Q'),
            RectMatrixPriorHelper(name='C', dim_names=['m', 'n'],
                var_row_name='R'),
            ]

class LGSSMPreconditioner(BasePreconditioner):
    """ LGSSM Preconditioner
    See individual Precondition Mixins for details
    """
    _precond_helper_list = [
            SquareMatrixPrecondHelper(name='A', dim_names=['n'], var_row_name='Q'),
            RectMatrixPrecondHelper(name='C', dim_names=['m', 'n'], var_row_name='R'),
            CovariancePrecondHelper(name='Q', dim_names=['n']),
            CovariancePrecondHelper(name='R', dim_names=['m']),
            ]

def generate_lgssm_data(T, parameters, initial_message = None,
        tqdm=None):
    """ Helper function for generating LGSSM time series

    Args:
        T (int): length of series
        parameters (LGSSMParameters): parameters
        initial_message (ndarray): prior for u_{-1}

    Returns:
        data (dict): dictionary containing:
            observations (ndarray): T by m
            latent_vars (ndarray): T by n
            parameters (LGSSMParameters)
            init_message (ndarray)
    """
    m, n = np.shape(parameters.C)
    A = parameters.A
    C = parameters.C
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
                mean=np.dot(C, latent_vars[t]),
                cov=R,
                )
        latent_prev = latent_vars[t]

    data = dict(
            observations=obs_vars,
            latent_vars=latent_vars,
            parameters=parameters,
            initial_message=initial_message,
            )
    return data


