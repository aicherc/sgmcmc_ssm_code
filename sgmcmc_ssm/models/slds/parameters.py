import numpy as np
import logging
logger = logging.getLogger(name=__name__)

from ...base_parameters import (
        BaseParameters, BasePrior, BasePreconditioner,
        )
from ...variables import (
        TransitionMatrixParamHelper, TransitionMatrixPriorHelper,
        TransitionMatrixPrecondHelper,
        SquareMatricesParamHelper, SquareMatricesPriorHelper,
        SquareMatricesPrecondHelper,
        RectMatrixParamHelper, RectMatrixPriorHelper,
        RectMatrixPrecondHelper,
        CovariancesParamHelper, CovariancesPriorHelper,
        CovariancesPrecondHelper,
        CovarianceParamHelper, CovariancePriorHelper,
        CovariancePrecondHelper,
        )
from ..._utils import (
        random_categorical,
        var_stationary_precision,
        )


class SLDSParameters(BaseParameters):
    """ SLDS Parameters """
    _param_helper_list = [
            TransitionMatrixParamHelper(name='pi', dim_names=['num_states', 'pi_type']),
            SquareMatricesParamHelper(name='A', dim_names=['n', 'num_states']),
            CovariancesParamHelper(name='Q', dim_names=['n', 'num_states']),
            RectMatrixParamHelper(name='C', dim_names=['m', 'n']),
            CovarianceParamHelper(name='R', dim_names=['m']),
            ]
    for param_helper in _param_helper_list:
        properties = param_helper.get_properties()
        for name, prop in properties.items():
            vars()[name] = prop

    def __str__(self):
        my_str = "SLDSParameters:"
        my_str += "\npi:\n" + str(self.pi)
        my_str += "\npi_type: `" + str(self.pi_type) + "`"
        my_str += "\nA:\n" + str(self.A)
        my_str += "\nC:\n" + str(self.C)
        my_str += "\nQ:\n" + str(self.Q)
        my_str += "\nR:\n" + str(self.R)
        return my_str

    def project_parameters(self, **kwargs):
        if 'C' not in kwargs:
            kwargs['C'] = dict(fixed_eye=True)
        return super().project_parameters(**kwargs)

class SLDSPrior(BasePrior):
    """ SLDS Prior
    See individual Prior Mixins for details
    """
    _Parameters = SLDSParameters
    _prior_helper_list = [
            CovariancePriorHelper(name='R', dim_names=['m'],
                matrix_name='C'),
            CovariancesPriorHelper(name='Q', dim_names=['n', 'num_states'],
                matrix_name='A'),
            TransitionMatrixPriorHelper(name='pi', dim_names=['num_states']),
            SquareMatricesPriorHelper(name='A', dim_names=['n', 'num_states'],
                var_row_name='Q'),
            RectMatrixPriorHelper(name='C', dim_names=['m', 'n'],
                var_row_name='R'),
            ]

class SLDSPreconditioner(BasePreconditioner):
    """ SLDS Preconditioner
    See individual Precondition Mixins for details
    """
    _precond_helper_list = [
            TransitionMatrixPrecondHelper(name='pi', dim_names=['num_states']),
            SquareMatricesPrecondHelper(name='A', dim_names=['n', 'num_states'],
                var_row_name='Q'),
            CovariancesPrecondHelper(name='Q', dim_names=['n', 'num_states']),
            RectMatrixPrecondHelper(name='C', dim_names=['m', 'n'],
                var_row_name='R'),
            CovariancePrecondHelper(name='R', dim_names=['m']),
            ]

def generate_slds_data(T, parameters, initial_message = None,
        tqdm=None):
    """ Helper function for generating SLDS time series

    Args:
        T (int): length of series
        parameters (LGSSMParameters): parameters
        initial_message (ndarray): prior for u_{-1}

    Returns:
        data (dict): dictionary containing:
            observations (ndarray): T by m
            latent_vars (dict):
                'x': continuous ndarray T by n
                'z': discrete ndarray values in {0,...,num_states-1}
            parameters (LGSSMParameters)
            init_message (ndarray)
    """
    num_states = np.shape(parameters.pi)[0]
    m, n = np.shape(parameters.C)
    Pi = parameters.pi
    A = parameters.A
    C = parameters.C
    Q = parameters.Q
    R = parameters.R

    if initial_message is None:
        init_precision = np.mean([var_stationary_precision(Qinv_k, A_k, 10)
            for Qinv_k, A_k in zip(parameters.Qinv, parameters.A)],
            axis=0)
        initial_message = {
                'x': {
                    'log_constant': 0.0,
                    'mean_precision': np.zeros(n),
                    'precision': init_precision,
                        },
                'z': {
                    'log_constant': 0.0,
                    'prob_vector': np.ones(num_states)/num_states,
                    },
                }

    latent_vars = {
            'x': np.zeros((T, n)),
            'z': np.zeros((T), dtype=int),
            }
    obs_vars = np.zeros((T, m))
    latent_prev_z = random_categorical(initial_message['z']['prob_vector'])
    latent_prev_x = np.random.multivariate_normal(
            mean=np.linalg.solve(initial_message['x']['precision'],
                initial_message['x']['mean_precision']),
            cov=np.linalg.inv(initial_message['x']['precision']),
            )

    pbar = range(T)
    if tqdm is not None:
        pbar = tqdm(pbar)
        pbar.set_description("generating data")
    for t in pbar:
        latent_z = random_categorical(Pi[latent_prev_z])
        latent_x = np.random.multivariate_normal(
                mean=np.dot(A[latent_z], latent_prev_x),
                cov=Q[latent_z],
                )

        obs_vars[t] = np.random.multivariate_normal(
                mean=np.dot(C, latent_x),
                cov=R,
                )
        latent_vars['z'][t] = latent_z
        latent_vars['x'][t,:] = latent_x

        latent_prev_z = latent_z
        latent_prev_x = latent_x

    data = dict(
            observations=obs_vars,
            latent_vars=latent_vars,
            parameters=parameters,
            initial_message=initial_message,
            )
    return data


