import numpy as np
import logging
logger = logging.getLogger(name=__name__)

from ...base_parameters import (
        BaseParameters, BasePrior, BasePreconditioner,
        )
from ...variables import (
        TransitionMatrixParamHelper, TransitionMatrixPriorHelper,
        TransitionMatrixPrecondHelper,
        RectMatricesParamHelper, RectMatricesPriorHelper,
        RectMatricesPrecondHelper,
        CovariancesParamHelper, CovariancesPriorHelper,
        CovariancesPrecondHelper,
        )
from ..._utils import random_categorical

class ARPHMMParameters(BaseParameters):
    """ AR(p) HMM Parameters """
    _param_helper_list = [
            TransitionMatrixParamHelper(name='pi', dim_names=['num_states', 'pi_type']),
            RectMatricesParamHelper(name='D', dim_names=['m', 'd', 'num_states']),
            CovariancesParamHelper(name='R', dim_names=['m', 'num_states']),
            ]
    for param_helper in _param_helper_list:
        properties = param_helper.get_properties()
        for name, prop in properties.items():
            vars()[name] = prop

    def __str__(self):
        my_str = "ARPHMMParameters:"
        my_str += "\npi:\n" + str(self.pi)
        my_str += "\npi_type: `" + str(self.pi_type) + "`"
        my_str += "\nD:\n" + str(self.D)
        my_str += "\nR:\n" + str(self.R)
        return my_str

    @property
    def p(self):
        return (self.d//self.m)

    def project_parameters(self, **kwargs):
        if 'D' not in kwargs:
            kwargs['D'] = dict(thresh = True)
        return super().project_parameters(**kwargs)

class ARPHMMPrior(BasePrior):
    """ AR(p) HMM Prior
    See individual Prior Mixins for details
    """
    _Parameters = ARPHMMParameters
    _prior_helper_list = [
            CovariancesPriorHelper(name='R', dim_names=['m', 'num_states'],
                matrix_name='D'),
            TransitionMatrixPriorHelper(name='pi', dim_names=['num_states']),
            RectMatricesPriorHelper(name='D',
                dim_names=['m', 'd', 'num_states'],
                var_row_name='R'),
            ]

class ARPHMMPreconditioner(BasePreconditioner):
    """ AR(p) HMM Preconditioner
    See individual Precondition Mixins for details
    """
    _precond_helper_list = [
            TransitionMatrixPrecondHelper(name='pi', dim_names=['num_states']),
            RectMatricesPrecondHelper(name='D',
                dim_names=['m', 'd', 'num_states'],
                var_row_name='R'),
            CovariancesPrecondHelper(name='R', dim_names=['m', 'num_states']),
            ]

def generate_arphmm_data(T, parameters, initial_message = None,
        tqdm=None):
    """ Helper function for generating ARPHMM time series

    Args:
        T (int): length of series
        parameters (ARPHMMParameters): parameters
        initial_message (ndarray): prior for u_{-1}

    Returns:
        data (dict): dictionary containing:
            observations (ndarray): T by p+1 by m
            latent_vars (ndarray): T takes values in {1,...,num_states}
            parameters (ARPHMMParameters)
            init_message (ndarray)
    """
    num_states, m, mp = np.shape(parameters.D)
    p = mp//m
    D = parameters.D
    R = parameters.R
    Pi = parameters.pi

    if initial_message is None:
        initial_message = {
                'prob_vector': np.ones(num_states)/num_states,
                'log_constant': 0.0,
                'y_prev': np.zeros((p,m))
                }

    latent_vars = np.zeros((T+p), dtype=int)
    obs_vars = np.zeros((T+p, m))
    latent_prev = random_categorical(initial_message['prob_vector'])
    y_prev = initial_message.get('y_prev')
    pbar = range(T)
    if tqdm is not None:
        pbar = tqdm(pbar)
        pbar.set_description("generating data")
    for t in pbar:
        latent_vars[t] = random_categorical(Pi[latent_prev])
        D_k = D[latent_vars[t]]
        R_k = R[latent_vars[t]]
        obs_vars[t] = np.random.multivariate_normal(
                mean=np.dot(D_k, y_prev.flatten()),
                cov = R_k,
                )
        latent_prev = latent_vars[t]
        y_prev = np.vstack([obs_vars[t], y_prev[:-1,:]])

    observations = stack_y(obs_vars, p)
    latent_vars = latent_vars[p:]

    data = dict(
            observations=observations,
            latent_vars=latent_vars,
            parameters=parameters,
            initial_message=initial_message,
            )
    return data

def stack_y(y, p):
    """ Stack y
    Args:
        y (ndarray): T+p by m matrix
    Returns:
        y_stacked (ndarray): T by p+1 by m matrix of the form
            y_stacked[0] = [y[p], y[p-1], ..., y[0]]
            y_stacked[1] = [y[p+1], y[p], ..., y[1]]
            ...
            y_stacked[t] = [y[p+t], y[p+t-1], ..., y[t]]
            ...
            y_stacked[T] = [y[p+T], y[p+T-1], ..., y[T]]
    """
    if np.ndim(y) == 1:
        y = np.array([y]).T
    T, m = np.shape(y)
    y_lags = [np.pad(y, ((0, lag), (0,0)), mode='constant')[lag:, :]
        for lag in reversed(range(p+1))]
    y_stacked = np.swapaxes(np.dstack(y_lags), 1, 2)[:T-p]
    return y_stacked

