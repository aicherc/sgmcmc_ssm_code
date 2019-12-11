import numpy as np
import logging
logger = logging.getLogger(name=__name__)

from ..base_parameters import (
        BaseParameters, BasePrior, BasePreconditioner,
        )
from ..variables import (
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
from ..sgmcmc_sampler import (
        SGMCMCSampler,
        SGMCMCHelper,
        )
from .._utils import (
        random_categorical,
        var_stationary_precision,
        lower_tri_mat_inv,
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
        my_str += "\nA:\n" + str(self.A)
        my_str += "\nC:\n" + str(self.C)
        my_str += "\nQ:\n" + str(self.Q)
        my_str += "\nR:\n" + str(self.R)
        return my_str

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

class SLDSHelper(SGMCMCHelper):
    """ LGSSM Helper

        forward_message (dict) with keys
            x (dict):
                log_constant (double) log scaling const
                mean_precision (ndarray) mean precision
                precision (ndarray) precision
            z (dict):
                prob_vector (ndarray) dimension num_states
                log_constant (double) log scaling const
            x_prev (ndarray)
            z_prev (ndarray)

        backward_message (dict) with keys
            x (dict):
                log_constant (double) log scaling const
                mean_precision (ndarray) mean precision
                precision (ndarray) precision
            z (dict):
                likelihood_vector (ndarray) dimension num_states
                log_constant (double) log scaling const
            x_next (ndarray)
            z_next (ndarray)

    """
    def __init__(self, num_states, n, m,
            forward_message=None, backward_message=None,
            **kwargs):
        self.num_states = num_states
        self.n = n
        self.m = m

        if forward_message is None:
            forward_message = {
                    'x': {
                        'log_constant': 0.0,
                        'mean_precision': np.zeros(self.n),
                        'precision': np.eye(self.n)/10,
                            },
                    'z': {
                        'log_constant': 0.0,
                        'prob_vector': np.ones(self.num_states)/self.num_states,
                        },
                    }
        self.default_forward_message=forward_message

        if backward_message is None:
            backward_message =  {
                'x': {
                    'log_constant': 0.0,
                    'mean_precision': np.zeros(self.n),
                    'precision': np.zeros((self.n, self.n)),
                        },
                'z': {
                    'log_constant': np.log(self.num_states),
                    'likelihood_vector':
                        np.ones(self.num_states)/self.num_states,
                    },
                }
        self.default_backward_message=backward_message
        return

    def marginal_loglikelihood(self, observations, parameters,
            forward_message=None, backward_message=None, x=None, z=None,
            **kwargs):
        if forward_message is None:
            forward_message = self.default_forward_message
        if backward_message is None:
            backward_message = self.default_backward_message

        if (z is not None) and (x is not None):
            return self._complete_data_loglikelihood(
                    observations=observations,
                    x=x, z=z,
                    parameters=parameters,
                    forward_message=forward_message,
                    backward_message=backward_message,
                    **kwargs)
        elif (z is not None) and (x is None):
            return self._x_marginal_loglikelihood(
                    observations=observations,
                    z=z,
                    parameters=parameters,
                    forward_message=forward_message,
                    backward_message=backward_message,
                    **kwargs)
        elif (z is None) and (x is not None):
            return self._z_marginal_loglikelihood(
                    observations=observations,
                    x=x,
                    parameters=parameters,
                    forward_message=forward_message,
                    backward_message=backward_message,
                    **kwargs)
        else:
            raise ValueError("Cannot marginalize both x and z")

    def gradient_marginal_loglikelihood(self, observations, parameters,
            forward_message=None, backward_message=None,
            x=None, z=None, **kwargs):
        if (z is not None) and (x is not None):
            return self._gradient_complete_data_loglikelihood(
                    observations=observations,
                    x=x,
                    z=z,
                    parameters=parameters,
                    forward_message=forward_message,
                    backward_message=backward_message,
                    **kwargs)
        elif (z is not None) and (x is None):
            return self._x_gradient_marginal_loglikelihood(
                    observations=observations,
                    z=z,
                    parameters=parameters,
                    forward_message=forward_message,
                    backward_message=backward_message,
                    **kwargs)
        elif (z is None) and (x is not None):
            return self._z_gradient_marginal_loglikelihood(
                    observations=observations,
                    x=x,
                    parameters=parameters,
                    forward_message=forward_message,
                    backward_message=backward_message,
                    **kwargs)
        else:
            raise ValueError("Cannot marginalize both x and z")

    def _forward_messages(self, observations, parameters, forward_message,
            x=None, z=None, **kwargs):
        if z is not None:
            if x is not None:
                raise ValueError("Either x or z can be conditioned on")
            # Forward Messages conditioned on z
            return self._x_forward_messages(
                    observations=observations,
                    z=z,
                    parameters=parameters,
                    forward_message=forward_message,
                    **kwargs
                    )
        elif x is not None:
            # Forward Messages conditioned on z
            return self._z_forward_messages(
                    observations=observations,
                    x=x,
                    parameters=parameters,
                    forward_message=forward_message,
                    **kwargs
                    )
        else:
            raise ValueError("Requires x or z be passed to condition on")

    def _backward_messages(self, observations, parameters, backward_message, x=None, z=None, **kwargs):
        if z is not None:
            if x is not None:
                raise ValueError("Either x or z can be conditioned on")
            # Forward Messages conditioned on z
            return self._x_backward_messages(
                    observations=observations,
                    z=z,
                    parameters=parameters,
                    backward_message=backward_message,
                    **kwargs
                    )
        elif x is not None:
            # Forward Messages conditioned on z
            return self._z_backward_messages(
                    observations=observations,
                    x=x,
                    parameters=parameters,
                    backward_message=backward_message,
                    **kwargs
                    )
        else:
            raise ValueError("Requires x or z be passed to condition on")

    def _x_forward_messages(self, observations, z, parameters, forward_message,
            weights=None, tqdm=None, only_return_last=False):
        # Return list of forward messages Pr(x_{t} | y_{<=t}, z)
        # y is num_obs x m matrix
        num_obs = np.shape(observations)[0]
        if not only_return_last:
            forward_messages = [None]*(num_obs+1)
            forward_messages[0] = forward_message

        mean_precision = forward_message['x']['mean_precision']
        precision = forward_message['x']['precision']
        log_constant = forward_message['x']['log_constant']
        z_prev = forward_message.get('z_prev', None)

        Pi = parameters.pi
        A = parameters.A
        LQinv = parameters.LQinv
        Qinv = np.array([np.dot(LQinv_k, LQinv_k.T)
            for LQinv_k in LQinv])
        AtQinv = np.array([np.dot(A_k.T, Qinv_k)
            for (A_k, Qinv_k) in zip(A, Qinv)])
        AtQinvA = np.array([np.dot(AtQinv_k, A_k)
            for (A_k, AtQinv_k) in zip(A, AtQinv)])

        C = parameters.C
        LRinv = parameters.LRinv
        Rinv = np.dot(LRinv, LRinv.T)
        CtRinv = np.dot(C.T, Rinv)
        CtRinvC = np.dot(CtRinv, C)

        pbar = range(num_obs)
        if tqdm is not None:
            pbar = tqdm(pbar)
            pbar.set_description("forward messages")
        for t in pbar:
            y_cur = observations[t]
            z_cur = z[t]
            weight_t = 1.0 if weights is None else weights[t]

            # Calculate Predict Parameters
            J = np.linalg.solve(AtQinvA[z_cur] + precision, AtQinv[z_cur])
            pred_mean_precision = np.dot(J.T, mean_precision)
            pred_precision = Qinv[z_cur] - np.dot(AtQinv[z_cur].T, J)

            # Calculate Observation Parameters
            y_mean = np.dot(C,
                    np.linalg.solve(pred_precision, pred_mean_precision))
            y_precision = Rinv - np.dot(CtRinv.T,
                    np.linalg.solve(CtRinvC + pred_precision, CtRinv))
            log_constant += weight_t * (
                    -0.5 * np.dot(y_cur-y_mean,
                            np.dot(y_precision, y_cur-y_mean)) + \
                     0.5 * np.linalg.slogdet(y_precision)[1] + \
                    -0.5 * self.m * np.log(2*np.pi)
                    )

            if z_prev is not None:
                log_constant += weight_t * np.log(Pi[z_prev, z_cur])


            # Calculate Filtered Parameters
            new_mean_precision = pred_mean_precision + np.dot(CtRinv, y_cur)
            new_precision = pred_precision + CtRinvC

            # Save Messages
            mean_precision = new_mean_precision
            precision = new_precision
            z_prev = z_cur
            if not only_return_last:
                forward_messages[t+1] = dict(
                        x={
                            'mean_precision': mean_precision,
                            'precision': precision,
                            'log_constant': log_constant,
                            },
                        z_prev=z_prev,
                    )
        if only_return_last:
            last_message = dict(
                    x={
                        'mean_precision': mean_precision,
                        'precision': precision,
                        'log_constant': log_constant,
                        },
                    z_prev=z_prev,
                )
            return last_message
        else:
            return forward_messages

    def _x_backward_messages(self, observations, z, parameters, backward_message,
            weights=None, tqdm=None, only_return_last=False):
        # Return list of backward messages Pr(y_{>t} | x_t, z)
        # y is num_obs x n matrix
        num_obs = np.shape(observations)[0]
        if not only_return_last:
            backward_messages = [None]*(num_obs+1)
            backward_messages[-1] = backward_message

        mean_precision = backward_message['x']['mean_precision']
        precision = backward_message['x']['precision']
        log_constant = backward_message['x']['log_constant']
        z_next = backward_message.get('z_next', None)

        Pi = parameters.pi
        A = parameters.A
        LQinv = parameters.LQinv
        Qinv = np.array([np.dot(LQinv_k, LQinv_k.T)
            for LQinv_k in LQinv])
        AtQinv = np.array([np.dot(A_k.T, Qinv_k)
            for (A_k, Qinv_k) in zip(A, Qinv)])
        AtQinvA = np.array([np.dot(AtQinv_k, A_k)
            for (A_k, AtQinv_k) in zip(A, AtQinv)])

        C = parameters.C
        LRinv = parameters.LRinv
        Rinv = np.dot(LRinv, LRinv.T)
        CtRinv = np.dot(C.T, Rinv)
        CtRinvC = np.dot(CtRinv, C)

        pbar = reversed(range(num_obs))
        if tqdm is not None:
            pbar = tqdm(pbar)
            pbar.set_description("backward messages")
        for t in pbar:
            y_cur = observations[t]
            z_cur = z[t]
            weight_t = 1.0 if weights is None else weights[t]

            # Helper Values
            xi = Qinv[z_cur] + precision + CtRinvC
            L = np.linalg.solve(xi, AtQinv[z_cur].T)
            vi = mean_precision + np.dot(CtRinv, y_cur)

            # Calculate new parameters
            log_constant += weight_t * (
                    -0.5 * self.m * np.log(2.0*np.pi) + \
                    np.sum(np.log(np.diag(LRinv))) + \
                    np.sum(np.log(np.diag(LQinv[z_cur]))) + \
                    -0.5 * np.linalg.slogdet(xi)[1] + \
                    -0.5 * np.dot(y_cur, np.dot(Rinv, y_cur)) + \
                    0.5 * np.dot(vi, np.linalg.solve(xi, vi))
                    )
            if z_next is not None:
                log_constant += weight_t * np.log(Pi[z_cur, z_next])

            new_mean_precision = np.dot(L.T, vi)
            new_precision = AtQinvA[z_cur] - np.dot(AtQinv[z_cur], L)

            # Save Messages
            mean_precision = new_mean_precision
            precision = new_precision
            z_next = z_cur

            if not only_return_last:
                backward_messages[t] = dict(x={
                    'mean_precision': mean_precision,
                    'precision': precision,
                    'log_constant': log_constant,
                }, z_next=z_next)
        if only_return_last:
            last_message = dict(x={
                'mean_precision': mean_precision,
                'precision': precision,
                'log_constant': log_constant,
            }, z_next=z_next)
            return last_message
        else:
            return backward_messages

    def _x_marginal_loglikelihood(self, observations, z, parameters,
            forward_message=None, backward_message=None, weights=None,
            **kwargs):
        # Run forward pass + combine with backward pass
        # y is num_obs x m matrix

        # forward_pass is Pr(x_{T-1} | y_{<=T-1})
        forward_pass = self._forward_message(
                observations=observations,
                z=z,
                parameters=parameters,
                forward_message=forward_message,
                weights=weights,
                **kwargs)

        weight_T = 1.0 if weights is None else weights[-1]

        loglikelihood = x_marginal_loglikelihood_helper(
                forward_pass, backward_message,
                weight = weight_T,
                )
        z_next = backward_message.get('z_next')
        z_prev = forward_pass.get('z_prev')
        if (z_next is not None) and (z_prev is not None):
            loglikelihood = loglikelihood + weight_T * np.log(
                    parameters.pi[z_prev, z_next])

        return loglikelihood

    def _x_gradient_marginal_loglikelihood(self, observations, z, parameters,
            forward_message=None, backward_message=None, weights=None,
            tqdm=None):
        Pi, expanded_pi = parameters.pi, parameters.expanded_pi
        A, LQinv, C, LRinv = \
                parameters.A, parameters.LQinv, parameters.C, parameters.LRinv

        # Forward Pass
        # forward_messages = [Pr(x_{t} | z, y_{-inf:t}), y{t}] for t=-1,...,T-1
        forward_messages = self.forward_pass(
                observations=observations,
                z=z,
                parameters=parameters,
                forward_message=forward_message,
                include_init_message=True)

        # Backward Pass
        # backward_messages = [Pr(y_{t+1:inf} | z,x_{t}), y{t}] for t=-1,...,T-1
        backward_messages = self.backward_pass(
                observations=observations,
                z=z,
                parameters=parameters,
                backward_message=backward_message,
                include_init_message=True)

        # Gradients
        grad = {var: np.zeros_like(value)
                for var, value in parameters.as_dict().items()}

        # Helper Constants
        Rinv = np.dot(LRinv, LRinv.T)
        RinvC = np.dot(Rinv, C)
        CtRinvC = np.dot(C.T, RinvC)
        LRinv_diaginv = np.diag(np.diag(LRinv)**-1)

        Qinv = np.array([np.dot(LQinv_k, LQinv_k.T)
            for LQinv_k in LQinv])
        QinvA = np.array([np.dot(Qinv_k, A_k)
            for (A_k, Qinv_k) in zip(A, Qinv)])
        AtQinvA = np.array([np.dot(A_k.T, QinvA_k)
            for (A_k, QinvA_k) in zip(A, QinvA)])
        LQinv_diaginv = np.array([np.diag(np.diag(LQinv_k)**-1)
            for LQinv_k in LQinv])

        # Emission Gradients
        p_bar = zip(forward_messages[1:], backward_messages[1:], observations)
        if tqdm is not None:
            pbar = tqdm(pbar)
            pbar.set_description("emission gradient loglike")
        for t, (forward_t, backward_t, y_t) in enumerate(p_bar):
            weight_t = 1.0 if weights is None else weights[t]

            # Pr(x_t | y)
            c_mean_precision = \
                    forward_t['x']['mean_precision'] + \
                    backward_t['x']['mean_precision']
            c_precision = \
                    forward_t['x']['precision'] + backward_t['x']['precision']

            x_mean = np.linalg.solve(c_precision, c_mean_precision)
            xxt_mean = np.linalg.inv(c_precision) + np.outer(x_mean, x_mean)

            # Gradient of C
            grad['C'] += weight_t * (np.outer(np.dot(Rinv, y_t), x_mean) + \
                    -1.0 * np.dot(RinvC, xxt_mean))

            # Gradient of LRinv
            #raise NotImplementedError("SHOULD CHECK THE MATH FOR LRINV")
            Cxyt = np.outer(np.dot(C, x_mean), y_t)
            CxxtCt = np.dot(C, np.dot(xxt_mean, C.T))
            grad['LRinv'] += weight_t * (LRinv_diaginv + \
                -1.0*np.dot(np.outer(y_t, y_t) - Cxyt - Cxyt.T + CxxtCt, LRinv))

        # Transition Gradients
        p_bar = zip(forward_messages[0:-1], backward_messages[1:], observations, z)
        if tqdm is not None:
            pbar = tqdm(pbar)
            pbar.set_description("transition gradient loglike")
        for t, (forward_t, backward_t, y_t, z_t) in enumerate(p_bar):
            weight_t = 1.0 if weights is None else weights[t]

            # Pr(x_t, x_t+1 | y)
            c_mean_precision = \
                np.concatenate([
                    forward_t['x']['mean_precision'],
                    backward_t['x']['mean_precision'] + np.dot(RinvC.T,y_t)
                    ])
            c_precision = \
                np.block([
                    [forward_t['x']['precision'] + AtQinvA[z_t],
                        -QinvA[z_t].T],
                    [-QinvA[z_t],
                        backward_t['x']['precision'] + CtRinvC + Qinv[z_t]]
                    ])

            c_mean = np.linalg.solve(c_precision, c_mean_precision)
            c_cov = np.linalg.inv(c_precision)

            xp_mean = c_mean[0:self.n]
            xn_mean = c_mean[self.n:]
            xpxpt_mean = c_cov[0:self.n, 0:self.n] + np.outer(xp_mean, xp_mean)
            xnxpt_mean = c_cov[self.n:, 0:self.n] + np.outer(xn_mean, xp_mean)
            xnxnt_mean = c_cov[self.n:, self.n:] + np.outer(xn_mean, xn_mean)

            # Gradient of A
            grad['A'][z_t] += weight_t * (np.dot(Qinv[z_t],
                    xnxpt_mean - np.dot(A[z_t],xpxpt_mean)))

            # Gradient of LQinv
            Axpxnt = np.dot(A[z_t], xnxpt_mean.T)
            AxpxptAt = np.dot(A[z_t], np.dot(xpxpt_mean, A[z_t].T))
            grad['LQinv'][z_t] += weight_t * (LQinv_diaginv[z_t] + \
                -1.0*np.dot(xnxnt_mean - Axpxnt - Axpxnt.T + AxpxptAt,
                        LQinv[z_t]))

        # Latent State Gradients
        z_prev = forward_message.get('z_prev') if forward_message is not None else None
        for t, z_t in enumerate(z):
            weight_t = 1.0 if weights is None else weights[t]
            if z_prev is not None:
                if parameters.pi_type == "logit":
                    logit_pi_grad_t = -Pi[z_prev] + 0.0
                    logit_pi_grad_t[z_t] += 1.0
                    grad['logit_pi'][z_prev] += weight_t * logit_pi_grad_t
                elif parameters.pi_type  == "expanded":
                    expanded_pi_grad_t = - Pi[z_prev] / expanded_pi[z_prev]
                    expanded_pi_grad_t[z_t] += 1.0 / expanded_pi[z_prev, z_t]
                    grad['expanded_pi'][z_prev] += weight_t * expanded_pi_grad_t
            z_prev = z_t

        return grad

    def _x_predictive_loglikelihood(self, observations, z, parameters, lag=10,
            forward_message=None, backward_message=None, **kwargs):
        if forward_message is None:
            forward_message = self.default_forward_message
        if backward_message is None:
            backward_message = self.default_backward_message

        # Calculate Filtered
        if lag == 0:
            forward_messages = self.forward_pass(
                    observations=observations,
                    z=z,
                    parameters=parameters,
                    forward_message=forward_message,
                    **kwargs)
        else:
            forward_messages = self.forward_pass(
                    observations=observations[0:-lag],
                    z=z[0:-lag],
                    parameters=parameters,
                    forward_message=forward_message,
                    **kwargs)
        loglike = 0.0
        A = parameters.A
        Q = parameters.Q
        C = parameters.C
        R = parameters.R
        for t in range(lag, np.shape(observations)[0]):
            y_cur = observations[t]
            z_cur = z[t]
            # Calculate Pr(x_t | y_{<=t-lag}, theta)
            mean_precision = forward_messages[t-lag]['x']['mean_precision']
            precision = forward_messages[t-lag]['x']['precision']
            mean = np.linalg.solve(precision, mean_precision)
            var = np.linalg.inv(precision)
            for l in range(lag):
                mean = np.dot(A[z_cur], mean)
                var = np.dot(A[z_cur], np.dot(var, A[z_cur].T)) + Q[z_cur]

            y_mean = np.dot(C, mean)
            y_var = np.dot(C, np.dot(var, C.T)) + R
            log_like_t = -0.5 * np.dot(y_cur - y_mean,
                    np.linalg.solve(y_var, y_cur - y_mean)) + \
                        -0.5 * np.linalg.slogdet(y_var)[1] + \
                        -0.5 * self.m * np.log(2*np.pi)
            loglike += log_like_t
        return loglike

    def _x_latent_var_sample(self, observations, z, parameters,
            forward_message=None, backward_message=None,
            distribution='smoothed', tqdm=None):
        """ Sample latent vars from observations

        Args:
            observations (ndarray): num_obs by n observations
            z (ndarray): num_obs latent states
            parameters (LGSSMParameters): parameters
            forward_message (dict): alpha message
            backward_message (dict): beta message
            distr (string): 'smoothed', 'filtered', 'predict'
                smoothed: sample X from Pr(X | Y, theta)
                filtered: sample X_t from Pr(X_t | Y_<=t, theta) iid for all t
                predictive: sample X_t from Pr(X_t | Y_<t, theta) iid for all t

        Returns
            x (ndarray): num_obs sampled latent values (in R^n)
        """
        if forward_message is None:
            forward_message = self.default_forward_message
        if backward_message is None:
            backward_message = self.default_backward_message

        A = parameters.A
        LQinv = parameters.LQinv
        Qinv = np.array([np.dot(LQinv_k, LQinv_k.T)
            for LQinv_k in LQinv])
        AtQinv = np.array([np.dot(A_k.T, Qinv_k)
            for (A_k, Qinv_k) in zip(A, Qinv)])
        AtQinvA = np.array([np.dot(AtQinv_k, A_k)
            for (A_k, AtQinv_k) in zip(A, AtQinv)])

        L = np.shape(observations)[0]

        if distribution == 'smoothed':
            # Forward Pass
            forward_messages = self.forward_pass(
                    observations=observations,
                    z=z,
                    parameters=parameters,
                    forward_message=forward_message,
                    include_init_message=False,
                    tqdm=tqdm
                    )

            # Backward Sampler
            x = np.zeros((L, self.n))
            x_cov = np.linalg.inv(forward_messages[-1]['x']['precision'])
            x_mean = np.dot(x_cov, forward_messages[-1]['x']['mean_precision'])
            x[-1, :] = np.random.multivariate_normal(mean=x_mean, cov=x_cov)

            pbar = reversed(range(L-1))
            if tqdm is not None:
                pbar = tqdm(pbar)
                pbar.set_description("backward smoothed sampling x")
            for t in pbar:
                x_next = x[t+1,:]
                z_next = z[t+1]
                x_cov = np.linalg.inv(forward_messages[t]['x']['precision'] +
                        AtQinvA[z_next])
                x_mean = np.dot(x_cov,
                        forward_messages[t]['x']['mean_precision'] +
                        np.dot(AtQinv[z_next], x_next))
                x[t,:] = np.random.multivariate_normal(x_mean, x_cov)
            return x

        elif distribution == 'filtered':
            # Forward Pass (not a valid probability density)
            forward_messages = self.forward_pass(
                    observations=observations,
                    z=z,
                    parameters=parameters,
                    forward_message=forward_message,
                    include_init_message=False,
                    tqdm=tqdm
                    )

            # Backward Sampler
            x = np.zeros((L, self.n))
            pbar = range(L)
            if tqdm is not None:
                pbar = tqdm(pbar)
                pbar.set_description("filtered sampling x")
            for t in pbar:
                x_cov = np.linalg.inv(forward_messages[t]['x']['precision'])
                x_mean = np.dot(x_cov,
                        forward_messages[t]['x']['mean_precision'])
                x[t,:] = np.random.multivariate_normal(x_mean, x_cov)
            return x

        elif distribution == 'predictive':
            # Forward Sampler (not a valid probability density)
            forward_messages = self.forward_pass(
                    observations=observations,
                    z=z,
                    parameters=parameters,
                    forward_message=forward_message,
                    include_init_message=True,
                    tqdm=tqdm
                    )

            # Backward Sampler
            x = np.zeros((L, self.n))
            Q = parameters.Q
            pbar = range(L)
            if tqdm is not None:
                pbar = tqdm(pbar)
                pbar.set_description("predictive sampling x")
            for t in pbar:
                z_cur = z[t]
                x_prev_cov = np.linalg.inv(
                        forward_messages[t]['x']['precision'])
                x_prev_mean = np.dot(x_prev_cov,
                        forward_messages[t]['x']['mean_precision'])
                x_cov = np.dot(A[z_cur],
                        np.dot(x_prev_cov, A[z_cur].T)) + Q[z_cur]
                x_mean = np.dot(A[z_cur], x_prev_mean)
                x[t,:] = np.random.multivariate_normal(x_mean, x_cov)
            return x
        else:
            raise ValueError("Invalid `distribution'; {0}".format(distribution))
        return

    def _z_forward_messages(self, observations, x, parameters, forward_message,
            weights=None, tqdm=None, only_return_last=False):
        # Return list of forward messages Pr(z_{t}, y_{<=t}, x)
        # y is num_obs x m matrix
        num_obs = np.shape(observations)[0]
        if not only_return_last:
            forward_messages = [None]*(num_obs+1)
            forward_messages[0] = forward_message

        prob_vector = forward_message['z']['prob_vector']
        log_constant = forward_message['z']['log_constant']
        x_prev = forward_message.get('x_prev', None)

        Pi = parameters.pi
        C = parameters.C
        LRinv = parameters.LRinv

        pbar = range(num_obs)
        if tqdm is not None:
            pbar = tqdm(pbar)
            pbar.set_description("forward messages")
        for t in pbar:
            y_cur = observations[t]
            x_cur = x[t]
            weight_t = 1.0 if weights is None else weights[t]

            # Log Pr(Y | X)
            LRinvTymCx = np.dot(LRinv.T, y_cur - np.dot(C, x_cur))
            log_constant += weight_t * (
                    -0.5 * self.m * np.log(2*np.pi) + \
                    -0.5*np.dot(LRinvTymCx, LRinvTymCx) + \
                    np.sum(np.log(np.diag(LRinv)))
                    )

            if x_prev is None:
                # Assume Non-informative prior for y_0
                prob_vector = np.dot(prob_vector, Pi)
            else:
                # Log Pr(X | X_prev)
                P_t, log_t = self._likelihoods(
                        x_cur, x_prev, parameters)
                prob_vector = np.dot(prob_vector, Pi)
                prob_vector = prob_vector * P_t
                log_constant += weight_t * (log_t + np.log(np.sum(prob_vector)))
                prob_vector = prob_vector/np.sum(prob_vector)

            # Save Messages
            x_prev = x_cur
            if not only_return_last:
                forward_messages[t+1] = dict(
                        z={
                            'prob_vector': prob_vector,
                            'log_constant': log_constant,
                            },
                        x_prev=x_prev,
                    )
        if only_return_last:
            last_message = dict(
                    z={
                        'prob_vector': prob_vector,
                        'log_constant': log_constant,
                        },
                    x_prev=x_prev,
                )
            return last_message
        else:
            return forward_messages

    def _z_backward_messages(self, observations, x, parameters,
            backward_message, weights=None, tqdm=None, only_return_last=False):
        # Return list of backward messages Pr(y_{>t} | x_t)
        # y is num_obs x n matrix
        num_obs = np.shape(observations)[0]
        if not only_return_last:
            backward_messages = [None]*(num_obs+1)
            backward_messages[-1] = backward_message

        prob_vector = backward_message['z']['likelihood_vector']
        log_constant = backward_message['z']['log_constant']
        x_next = backward_message.get('x_next', None)

        Pi = parameters.pi
        C = parameters.C
        LRinv = parameters.LRinv

        pbar = reversed(range(num_obs))
        if tqdm is not None:
            pbar = tqdm(pbar)
            pbar.set_description("backward messages")
        for t in pbar:
            y_cur = observations[t]
            x_cur = x[t]
            weight_t = 1.0 if weights is None else weights[t]

            # Log Pr(Y_cur | X_cur )
            LRinvTymCx = np.dot(LRinv.T, y_cur - np.dot(C, x_cur))
            log_constant += weight_t * (
                    -0.5 * self.m * np.log(2*np.pi) + \
                    -0.5*np.dot(LRinvTymCx, LRinvTymCx) + \
                    np.sum(np.log(np.diag(LRinv)))
                    )

            if x_next is None:
                prob_vector = np.dot(Pi, prob_vector)
                log_constant += weight_t * np.log(np.sum(prob_vector))
                prob_vector = prob_vector/np.sum(prob_vector)
            else:
                # Log Pr(X_next | X_cur)
                P_t, log_t = self._likelihoods(
                        x_next, x_cur, parameters)
                prob_vector = P_t * prob_vector
                prob_vector = np.dot(Pi, prob_vector)
                log_constant += weight_t * (log_t + \
                        np.log(np.sum(prob_vector)))
                prob_vector = prob_vector/np.sum(prob_vector)


            # Save Messages
            x_next = x_cur
            if not only_return_last:
                backward_messages[t] = dict(z={
                    'likelihood_vector': prob_vector,
                    'log_constant': log_constant,
                }, x_next=x_next)

        if only_return_last:
            last_message = dict(z={
                'likelihood_vector': prob_vector,
                'log_constant': log_constant,
            }, x_next=x_next)
            return last_message
        else:
            return backward_messages

    def _z_marginal_loglikelihood(self, observations, x, parameters,
            forward_message=None, backward_message=None, weights=None,
            **kwargs):
        # Run forward pass + combine with backward pass
        # y is num_obs x m matrix

        # forward_pass is Pr(z_{T-1} | x_{<=T-1}, y_{<=T-1})
        forward_pass = self._forward_message(
                observations=observations,
                x=x,
                parameters=parameters,
                forward_message=forward_message,
                weights=weights,
                **kwargs)

        Pi = parameters.pi
        x_prev = forward_pass.get('x_prev')
        x_cur = backward_message.get('x_next')
        prob_vector = forward_pass['z']['prob_vector']
        log_constant = forward_pass['z']['log_constant']
        prob_vector = np.dot(prob_vector, Pi)
        weight_T = 1.0 if weights is None else weights[-1]

        if (x_cur is not None) and (x_prev is not None):
            P_t, log_t = self._likelihoods(
                    x_cur, x_prev, parameters,
                )
            prob_vector = P_t * prob_vector
            log_constant += weight_T * log_t

        log_constant += weight_T * backward_message['z']['log_constant']
        likelihood = np.dot(prob_vector,
                backward_message['z']['likelihood_vector'])
        loglikelihood = weight_T * np.log(likelihood) + log_constant

        return loglikelihood

    def _z_gradient_marginal_loglikelihood(self, observations, x, parameters,
            forward_message=None, backward_message=None, weights=None,
            tqdm=None):
        Pi, expanded_pi = parameters.pi, parameters.expanded_pi
        A, LQinv, C, LRinv = \
                parameters.A, parameters.LQinv, parameters.C, parameters.LRinv
        Rinv = parameters.Rinv
        Q, Qinv = parameters.Q, parameters.Qinv
        LRinv_Tinv = lower_tri_mat_inv(LRinv).T

        # Forward Pass
        # forward_messages = [Pr(z_{t} | y_{-inf:t}, x_{-inf:t})]
        #   for t=-1:T-1
        forward_messages = self.forward_pass(
                observations=observations,
                x=x,
                parameters=parameters,
                forward_message=forward_message,
                include_init_message=True)

        # Backward Pass
        # backward_messages = [Pr(y_{t:inf}, x_{t:inf} | z_{t}), x_{t}]
        #   for t=0:T
        backward_messages = self.backward_pass(
                observations=observations,
                x=x,
                parameters=parameters,
                backward_message=backward_message,
                include_init_message=True)

        # Gradients
        grad = {var: np.zeros_like(value)
                for var, value in parameters.as_dict().items()}

        # Transition Gradients
        p_bar = zip(forward_messages[:-1], backward_messages[1:], observations, x)
        if tqdm is not None:
            pbar = tqdm(pbar)
            pbar.set_description("gradient loglike")
        for t, (forward_t, backward_t, y_t, x_t) in enumerate(p_bar):
            # r_t is Pr(z_{t-1} | y_{< t})
            # s_t is Pr(z_t | y_{< t})
            # q_t is Pr(y_{> t} | z_t)
            r_t = forward_t['z']['prob_vector']
            s_t = np.dot(r_t, Pi)
            q_t = backward_t['z']['likelihood_vector']
            weight_t = 1.0 if weights is None else weights[t]

            x_prev = forward_t.get('x_prev', None)
            x_cur = x_t
            if (x_prev is not None) and (x_cur is not None):
                P_t, _ = self._likelihoods(
                        x_cur, x_prev, parameters
                    )
            else:
                P_t = np.ones(self.num_states)

            # Marginal + Pairwise Marginal
            joint_post = np.diag(r_t).dot(Pi).dot(np.diag(P_t*q_t))
            joint_post = joint_post/np.sum(joint_post)
            marg_post = np.sum(joint_post, axis=0)

            # Grad for pi
            if parameters.pi_type == "logit":
                # Gradient of logit_pi
                grad['logit_pi'] += weight_t * (joint_post - \
                        np.diag(np.sum(joint_post, axis=1)).dot(Pi))
            elif parameters.pi_type == "expanded":
                grad['expanded_pi'] += weight_t * np.array([
                    (expanded_pi[k]**-1)*(
                        joint_post[k] - np.sum(joint_post[k])*Pi[k])
                    for k in range(self.num_states)
                    ])
            else:
                raise RuntimeError()

            # Grad for A and LQinv
            x_prev = forward_t.get('x_prev', None)
            if (x_prev is not None) and (x_cur is not None):
                for k, A_k, LQinv_k, Qinv_k, Q_k in zip(
                        range(self.num_states), A, LQinv, Qinv, Q):
                    diff_k = x_cur - A_k.dot(x_prev)
                    grad['A'][k] = (
                            np.outer(Qinv_k.dot(diff_k), x_prev)
                            ) * marg_post[k] * weight_t
                    grad['LQinv'][k] = (
                            (Q_k - np.outer(diff_k, diff_k)).dot(LQinv_k)
                            ) * marg_post[k] * weight_t


            # Grad for C and LRinv
            grad['C'] += weight_t * np.outer(np.dot(Rinv, y_t-np.dot(C,x_t)), x_t)
            grad['LRinv'] += weight_t * (LRinv_Tinv + \
                -1.0*np.dot(np.outer(y_t-np.dot(C,x_t), y_t-np.dot(C,x_t)),
                        LRinv))

        return grad

    def _z_predictive_loglikelihood(self, observations, x, parameters, lag=10,
            forward_message=None, backward_message=None, **kwargs):
        if forward_message is None:
            forward_message = self.default_forward_message
        if backward_message is None:
            backward_message = self.default_backward_message

        # Calculate Filtered
        forward_messages = self.forward_pass(
                observations=observations[0:-lag],
                parameters=parameters,
                x=x,
                forward_message=forward_message, **kwargs)
        loglike = 0.0
        Pi = parameters.pi
        for t in range(lag, np.shape(observations)[0]):
            # Calculate Pr(z_t | y_{<=t-lag}, theta)
            prob_vector = forward_messages[t-lag]['z']['prob_vector']
            for l in range(lag):
                prob_vector = np.dot(prob_vector, Pi)

            x_cur = x[t]
            x_prev = x[t-1]
            P_t, log_constant = self._likelihoods(x_cur, x_prev, parameters)
            likelihood = np.dot(prob_vector, P_t)
            loglike += np.log(likelihood) + log_constant
        return loglike

    def _z_latent_var_sample(self, observations, x, parameters,
            forward_message=None, backward_message=None,
            distribution='smoothed', tqdm=None):
        """ Sample latent vars from observations

        Args:
            observations (ndarray): num_obs by m observations
            x (ndarray): num_obs by n latent continuous variables
            parameters (LGSSMParameters): parameters
            forward_message (dict): alpha message
            backward_message (dict): beta message
            distr (string): 'smoothed', 'filtered', 'predict'
                smoothed: sample z from Pr(z | Y, theta)
                filtered: sample z_t from Pr(z_t | Y_<=t, theta) iid for all t
                predictive: sample z_t from Pr(z_t | Y_<t, theta) iid for all t

        Returns
            z (ndarray): num_obs sampled latent values (in 1,...,K)
        """
        if forward_message is None:
            forward_message = self.default_forward_message
        if backward_message is None:
            backward_message = self.default_backward_message

        Pi = parameters.pi
        L = np.shape(observations)[0]
        z = np.zeros(L, dtype=int)
        if np.shape(x)[0] != L:
            raise ValueError('observations and x have different shapes')

        if distribution == 'smoothed':
            # Backward Pass
            backward_messages = self.backward_pass(
                    observations=observations,
                    x=x,
                    parameters=parameters,
                    backward_message=backward_message,
                    tqdm=tqdm
                    )

            # Forward Sampler
            pbar = enumerate(backward_messages)
            if tqdm is not None:
                pbar = tqdm(pbar)
                pbar.set_description("forward smoothed sampling z")
            x_prev = forward_message.get('x_prev', None)
            for t, backward_t in pbar:
                x_cur = x[t]
                if t == 0:
                    post_t = np.dot(forward_message['z']['prob_vector'], Pi)
                else:
                    post_t = Pi[z[t-1]]
                if x_prev is not None:
                    P_t, _ = self._likelihoods(
                            x_cur, x_prev, parameters,
                        )
                    post_t = post_t * P_t * backward_t['z']['likelihood_vector']
                    post_t = post_t/np.sum(post_t)
                x_prev = x_cur
                z[t] = random_categorical(post_t)

        elif distribution == 'filtered':
            # Forward Sampler (not a valid probability density)
            x_prev = forward_message.get('x_prev', None)
            pbar = enumerate(x)
            if tqdm is not None:
                pbar = tqdm(pbar)
                pbar.set_description("forward filtered sampling z")
            for t, x_cur in pbar:
                if t == 0:
                    post_t = np.dot(forward_message['z']['prob_vector'], Pi)
                else:
                    post_t = Pi[z[t-1]]
                if x_prev is not None:
                    P_t, _ = self._likelihoods(
                            x_cur, x_prev, parameters,
                        )
                    post_t = post_t * P_t
                    post_t = post_t/np.sum(post_t)
                x_prev = x_cur
                z[t] = random_categorical(post_t)

        elif distribution == 'predictive':
            # Forward Sampler (not a valid probability density)
            x_prev = forward_message.get('x_prev', None)
            pbar = range(np.shape(x)[0])
            if tqdm is not None:
                pbar = tqdm(pbar)
                pbar.set_description("forward filtered sampling z")
            for t in pbar:
                if t == 0:
                    prob_vector = np.dot(forward_message['z']['prob_vector'],
                            Pi)
                else:
                    x_cur = x[t-1]
                    if x_prev is not None:
                        P_t, _ = self._likelihoods(
                                x_cur, x_prev, parameters,
                            )
                        prob_vector = prob_vector * P_t
                    prob_vector = np.dot(prob_vector, Pi)
                    x_prev = x_cur
                prob_vector = prob_vector/np.sum(prob_vector)
                z[t] = random_categorical(prob_vector)
        else:
            raise ValueError("Unrecognized distr {0}".format(distr))

        return z

    def _likelihoods(self, x_cur, x_prev, parameters):
        #if (x_prev is None) or (x_cur is None):
        #    return np.ones(self.num_states), 0.0
        logP_t = self._ar_loglikelihoods(
                x_cur=x_cur, x_prev=x_prev, parameters=parameters,
                )
        log_constant = np.max(logP_t)
        logP_t = logP_t - log_constant
        P_t = np.exp(logP_t)
        return P_t, log_constant

    def _ar_loglikelihoods(self, x_cur, x_prev, parameters):
        # y_cur should be p+1 by m,
        loglikelihoods = np.zeros(self.num_states, dtype=float)
        for k, (A_k, LQinv_k) in enumerate(
                zip(parameters.A, parameters.LQinv)):
            delta = x_cur - np.dot(A_k, x_prev)
            LQinvTdelta = np.dot(delta, LQinv_k)
            loglikelihoods[k] = \
                -0.5 * np.dot(LQinvTdelta, LQinvTdelta) + \
                -0.5 * self.m * np.log(2*np.pi) + \
                np.sum(np.log(np.diag(LQinv_k)))
        return loglikelihoods

    def _complete_data_loglikelihood(self, observations, x, z, parameters,
            forward_message=None, backward_message=None, weights=None,
            **kwargs):
        # y is num_obs x m matrix
        log_constant = 0.0
        Pi = parameters.pi
        A = parameters.A
        LQinv = parameters.LQinv
        C = parameters.C
        LRinv = parameters.LRinv

        z_prev = forward_message.get('z_prev')
        x_prev = forward_message.get('x_prev')
        for t, (y_t, x_t, z_t) in enumerate(zip(observations, x, z)):
            weight_t = 1.0 if weights is None else weights[t]

            # Pr(Z_t | Z_t-1)
            if z_prev is not None:
                log_constant += weight_t * np.log(Pi[z_prev, z_t])

            # Pr(X_t | X_t-1)
            if (z_prev is not None) and (x_prev is not None):
                diffLQinv = np.dot(x_t - np.dot(A[z_t],x_prev), LQinv[z_t])
                log_constant += weight_t * (
                        -0.5 * self.n * np.log(2*np.pi) + \
                        -0.5 * np.dot(diffLQinv, diffLQinv) + \
                        np.sum(np.log(np.diag(LQinv[z_t])))
                        )

            # Pr(Y_t | X_t)
            LRinvTymCx = np.dot(LRinv.T, y_t - np.dot(C, x_t))
            log_constant += weight_t * (
                    -0.5 * self.m * np.log(2*np.pi) + \
                    -0.5*np.dot(LRinvTymCx, LRinvTymCx) + \
                    np.sum(np.log(np.diag(LRinv)))
                    )

            z_prev = z_t
            x_prev = x_t

        return log_constant

    def _gradient_complete_data_loglikelihood(self, observations, x, z,
            parameters,
            forward_message=None, backward_message=None, weights=None,
            tqdm=None):
        if forward_message is None:
            forward_message = {}
        if backward_message is None:
            backward_message = {}
        Pi, expanded_pi = parameters.pi, parameters.expanded_pi
        A, LQinv = parameters.A, parameters.LQinv
        Qinv = parameters.Qinv
        C, LRinv = parameters.C, parameters.LRinv
        Rinv = parameters.Rinv

        LRinv_Tinv = lower_tri_mat_inv(LRinv).T
        LQinv_Tinv = np.array([lower_tri_mat_inv(LQinv[k]).T
            for k in range(parameters.num_states)])

        # Gradients
        grad = {var: np.zeros_like(value)
                for var, value in parameters.as_dict().items()}

        # Latent State Gradients
        z_prev = forward_message.get('z_prev')
        for t, z_t in enumerate(z):
            weight_t = 1.0 if weights is None else weights[t]
            if z_prev is not None:
                if parameters.pi_type == "logit":
                    logit_pi_grad_t = -Pi[z_prev] + 0.0
                    logit_pi_grad_t[z_t] += 1.0
                    grad['logit_pi'][z_prev] += weight_t * logit_pi_grad_t
                elif parameters.pi_type  == "expanded":
                    expanded_pi_grad_t = - Pi[z_prev] / expanded_pi[z_prev]
                    expanded_pi_grad_t[z_t] += 1.0 / expanded_pi[z_prev, z_t]
                    grad['expanded_pi'][z_prev] += weight_t * expanded_pi_grad_t
            z_prev = z_t

        # Transition Gradients
        x_prev = forward_message.get('x_prev')
        for t, (x_t, z_t) in enumerate(zip(x, z)):
            weight_t = 1.0 if weights is None else weights[t]
            if x_prev is not None:
                A_k = A[z_t]
                diff = x_t - np.dot(A_k, x_prev)
                grad['A'][z_t] += weight_t * np.outer(
                    np.dot(Qinv[z_t], diff), x_prev)
                grad['LQinv'][z_t] += weight_t * (LQinv_Tinv[z_t] + \
                    -1.0*np.dot(np.outer(diff, diff), LQinv[z_t]))
            x_prev = x_t

        # Emission Gradients
        for t, (x_t, y_t) in enumerate(zip(x, observations)):
            weight_t = 1.0 if weights is None else weights[t]
            diff = y_t - np.dot(C, x_t)
            grad['C'] += weight_t * np.outer(np.dot(Rinv, diff), x_t)
            grad['LRinv'] += weight_t * (LRinv_Tinv + \
                -1.0*np.dot(np.outer(diff, diff), LRinv))

        return grad

    def calc_gibbs_sufficient_statistic(self, observations, latent_vars,
            **kwargs):
        """ Gibbs Sample Sufficient Statistics
        Args:
            observations (ndarray): num_obs observations
            latent_vars (dict): latent vars

        Returns:
            sufficient_stat (dict)
        """
        y = observations
        x = latent_vars['x']
        z = latent_vars['z']

        # Sufficient Statistics for Pi
        z_pair_count = np.zeros((self.num_states, self.num_states))
        for t in range(1, np.size(z)):
            z_pair_count[z[t-1], z[t]] += 1.0

        # Sufficient Statistics for A and Q
        # From Emily Fox's Thesis Page 147
        transition_count = np.zeros(self.num_states)
        Sx_prevprev = np.zeros((self.num_states, self.n, self.n))
        Sx_curprev = np.zeros((self.num_states, self.n, self.n))
        Sx_curcur = np.zeros((self.num_states, self.n, self.n))

        for k in range(0, self.num_states):
            transition_count[k] = np.sum(z == k)
            # Construct Psi & Psi_prev Matrices
            if np.sum(z[1:] == k) == 0:
                # No Sufficient Statistics for No Observations
                continue
            PsiT = x[1:][z[1:]==k,:]
            PsiT_prev = x[:-1][z[1:]==k,:]

            # Sufficient Statistics for group k
            Sx_prevprev[k] = PsiT_prev.T.dot(PsiT_prev)
            Sx_curprev[k] = PsiT.T.dot(PsiT_prev)
            Sx_curcur[k] = PsiT.T.dot(PsiT)

        # Sufficient Statistics for C and R
        # From Emily Fox's Thesis Page 147
        PsiT = y
        PsiT_prev = x
        emission_count = len(PsiT)
        S_prevprev = PsiT_prev.T.dot(PsiT_prev)
        S_curprev = PsiT.T.dot(PsiT_prev)
        S_curcur = PsiT.T.dot(PsiT)

        # Return sufficient Statistics
        sufficient_stat = {}
        sufficient_stat['pi'] = dict(alpha = z_pair_count)
        sufficient_stat['A'] = dict(
                S_prevprev = Sx_prevprev,
                S_curprev = Sx_curprev,
                )
        sufficient_stat['Q'] = dict(
                S_count=transition_count,
                S_prevprev = Sx_prevprev,
                S_curprev = Sx_curprev,
                S_curcur=Sx_curcur,
                )
        sufficient_stat['C'] = dict(
                S_prevprev = S_prevprev,
                S_curprev = S_curprev,
                )
        sufficient_stat['R'] = dict(
                S_count=emission_count,
                S_prevprev = S_prevprev,
                S_curprev = S_curprev,
                S_curcur=S_curcur,
                )

        return sufficient_stat

def x_marginal_loglikelihood_helper(forward_message, backward_message,
        weight=1.0):
    # Calculate the marginal loglikelihood of forward + backward message
    f_mean_precision = forward_message['x']['mean_precision']
    f_precision = forward_message['x']['precision']
    c_mean_precision = f_mean_precision + backward_message['x']['mean_precision']
    c_precision = f_precision + backward_message['x']['precision']

    log_constant = forward_message['x']['log_constant'] + \
            (backward_message['x']['log_constant'] + \
            +0.5 * np.linalg.slogdet(f_precision)[1] + \
            -0.5 * np.linalg.slogdet(c_precision)[1] + \
            -0.5 * np.dot(f_mean_precision,
                    np.linalg.solve(f_precision, f_mean_precision)
                ) + \
            0.5 * np.dot(c_mean_precision,
                np.linalg.solve(c_precision, c_mean_precision)
                )
            ) * weight
    return log_constant

class SLDSSampler(SGMCMCSampler):
    def __init__(self, num_states, n, m, name="SLDSSampler", **kwargs):
        self.options = kwargs
        self.num_states = num_states
        self.n = n
        self.m = m
        self.name = name
        self.message_helper=SLDSHelper(
                num_states=self.num_states,
                n=self.n,
                m=self.m,
                )
        return

    def setup(self, observations, prior, parameters=None, forward_message=None):
        """ Initialize the sampler

        Args:
            observations (ndarray): T by m ndarray of time series values
            prior (SLDSPrior): prior
            forward_message (ndarray): prior probability for latent state
            parameters (SLDSParameters): initial parameters
                (optional, will sample from prior by default)

        """
        # Check Shape
        if np.shape(observations)[1] != self.m:
            raise ValueError("observations second dimension does not match m")

        self.observations = observations
        self.T = np.shape(self.observations)[0]

        self.prior = prior

        if parameters is None:
            self.parameters = self.prior.sample_prior()
        else:
            if not isinstance(parameters, SLDSParameters):
                raise ValueError("parameters is not a SLDSParameter")
            self.parameters = parameters


        if forward_message is None:
            forward_message = {
                    'x': {
                        'log_constant': 0.0,
                        'mean_precision': np.zeros(self.n),
                        'precision': np.eye(self.n)/10,
                            },
                    'z': {
                        'log_constant': 0.0,
                        'prob_vector': np.ones(self.num_states)/self.num_states,
                        },
                    }
        self.forward_message = forward_message
        self.backward_message =  {
                'x': {
                    'log_constant': 0.0,
                    'mean_precision': np.zeros(self.n),
                    'precision': np.zeros((self.n, self.n)),
                        },
                'z': {
                    'log_constant': np.log(self.num_states),
                    'likelihood_vector':
                        np.ones(self.num_states)/self.num_states,
                    },
                }

        return

    def init_parameters_from_x_and_z(self, x, z):
        """ Get initial parameters for the sampler

        Args:
            x (ndarray): latent var
            z (ndarray): latent var

        Return:
            init_parameters (SLDSParameters): init_parameters
        """
        # Check z is appropriate size
        if np.shape(z)[0] != self.T:
            raise ValueError("z must be length T = {0}".format(self.T))

        if not np.issubdtype(z.dtype, np.integer):
            raise ValueError("z must be integers, not {0}".format(z.dtype))

        if np.max(z) >= self.num_states or np.min(z) < 0:
            raise ValueError("z must be in (0, \ldots, {0}-1)".format(
                self.num_states))

        # Check x is appropriate size
        if np.shape(x)[0] != self.T or np.shape(x)[1] != self.n:
            raise ValueError("x must be size {0} not {1}".format(
                (self.T, self.n), np.shape(x)))

        # Init on Gibb Step
        init_parameters = self.message_helper.parameters_gibbs_sample(
                observations=self.observations,
                latent_vars=dict(x=x, z=z),
                forward_message=self.forward_message,
                backward_message=self.backward_message,
                prior=self.prior,
                )

        return init_parameters

    def init_parameters_from_k_means(self, x=None, lags=[0,1], kmeans=None, **kwargs):
        """ Get initial parameters for the sampler

        Use KMeans on data (treating observations as independent)
        Each point is concat(y[lag] for lag in lags)

        Args:
            x (ndarray): initialization of latent variables
                default is to use observations
            lags (list of indices): indices of lags to use for clustering
            kmeans (sklearn model): e.g. sklearn.cluster.KMeans
            **kwargs (dict): keyword args to pass to sklearn's kmean
                "n_init" : int (default = 10)
                "max_iter": int (default = 300)
                "n_jobs" : int (default = 1)
                See sklearn.cluster.KMeans for more


        Returns:
            init_parameters (SLDSParameters): init_parameters
        """
        from sklearn.cluster import KMeans, MiniBatchKMeans

        # Run KMeans
        if kmeans is None:
            if self.T <= 10**6:
                kmeans = KMeans(n_clusters = self.num_states, **kwargs)
            else:
                kmeans = MiniBatchKMeans(n_clusters = self.num_states, **kwargs)

        X = self.observations.reshape((self.T, -1))
        X_lagged = np.hstack([
            X[max(lags)-lag:X.shape[0]-lag] for lag in lags
        ])

        z = kmeans.fit_predict(X=X_lagged)
        if z.size < self.T:
            z = np.concatenate([np.zeros(self.T-z.size, dtype=int), z])
        if x is None:
            x = self.observations

        # Calculate Initial Param from KMeans init
        init_parameters = self.init_parameters_from_x_and_z(x=x, z=z)

        return init_parameters

    def init_sample_latent(self, init_method=None, init_burnin=0,
            parameters=None, observations=None, track_samples=True,
            z_init=None):
        """ Initialize latent variables

        Args:
            init_method (string)
                'copy' - use observations as continuous latent variables
                'filtered' - draw z_t, x_t conditional on z_<t, x_<t, y_<=t
                'filteredZ' - draw z_t conditional on z_<t, y_<=t
                'from_vector' - draw x conditional on given z
            init_burnin (int): additional Gibbs sampling steps
            z_init (ndarray): optional, for init_method == 'from_vector'

        Returns:
            latent_vars (dict):
                x (ndarray)
                z (ndarray)
        """
        if observations is None:
            observations = self.observations
        if parameters is None:
            parameters = self.parameters

        if init_method is None:
            # Set default init method
            if self.n <= self.m:
                init_method = 'copy'
            if self.n > self.m:
                init_method = 'filteredZ'

        # Init Methods
        if init_method == 'copy':
            if self.n > self.m:
                raise ValueError("Cannot use init_method = 'copy' since n > m")
            z = self.sample_z(x=observations[:, 0:self.n],
                    parameters=parameters,
                    observations=observations,
                    track_samples=track_samples,
                    )
            x = self.sample_x(z=z,
                    parameters=parameters,
                    observations=observations,
                    track_samples=track_samples,
                    )
        elif init_method == 'filtered':
            logger.warning("Executing <init_method == 'filtered'>")
            logger.warning("Strongly recommend <init_method == 'filteredZ>")
            x, z = self.message_helper.init_filter_naive(
                    y=observations,
                    parameters=parameters,
                    x_forward_message=self.forward_message['x'],
                    z_forward_message=self.forward_message['z'],
                    )

        elif init_method == 'filteredZ':
            z = self.message_helper.init_filter_z(
                    y=observations,
                    parameters=parameters,
                    x_forward_message=self.forward_message['x'],
                    z_forward_message=self.forward_message['z'],
                    )
            x = self.sample_x(z=z,
                    parameters=parameters,
                    observations=observations,
                    track_samples=track_samples,
                    )
        elif init_method == "from_vector":
            if np.max(z_init) >= self.num_states:
                raise ValueError("z_init contains more states than in model")
            z = z_init.copy()
            x = self.sample_x(z=z,
                    parameters=parameters,
                    observations=observations,
                    track_samples=track_samples,
                    )
        else:
            raise ValueError("Unrecognized init_method {0}".format(init_method))

        for step in range(init_burnin):
            z = self.sample_z(x=x,
                    parameters=parameters,
                    observations=observations,
                    track_samples=track_samples)
            x = self.sample_x(z=z,
                    parameters=parameters,
                    observations=observations,
                    track_samples=track_samples,
                    )

        return dict(x=x, z=z)

    def sample_z(self, x=None, parameters=None, observations=None, tqdm=None,
            track_samples=True):
        """ Sample Z (given X)"""
        if parameters is None:
            parameters = self.parameters
        if observations is None:
            observations = self.observations
        if x is None:
            x = self.x
        if np.shape(x)[0] != np.shape(observations)[0]:
            raise ValueError("x and observations are different lengths")
        if np.shape(x)[1] != self.n:
            raise ValueError("x must be T by n ndarray")

        z = self.message_helper._z_latent_var_sample(
                observations=observations,
                x=x,
                parameters=parameters,
                forward_message=self.forward_message,
                backward_message=self.backward_message,
                tqdm=tqdm,
                )
        if track_samples:
            self.z = z.copy()
        return z

    def sample_x(self, z=None, parameters=None, observations=None, tqdm=None,
            track_samples=True):
        """ Sample X (given Z)"""
        if parameters is None:
            parameters = self.parameters
        if observations is None:
            observations = self.observations
        if z is None:
            z = self.z
        if np.shape(z)[0] != np.shape(observations)[0]:
            raise ValueError("z and observations are different lengths")
        if z.dtype != int:
            raise ValueError("z must be ints")

        x = self.message_helper._x_latent_var_sample(
                observations=observations,
                z=z,
                parameters=parameters,
                forward_message=self.forward_message,
                backward_message=self.backward_message,
                tqdm=tqdm,
                )
        if track_samples:
            self.x = x.copy()
        return x

    def sample_latent(self, x=None, z=None, num_rep=1, **kwargs):
        """ Sample x, z for observations """
        # Setup z and x
        if x is None and z is None:
            x, z = self.x, self.z
        if z is None:
            z = self.sample_z(x=x, **kwargs)
        if x is None:
            x = self.sample_x(z=z, **kwargs)

        for rep in range(num_rep):
            z = self.sample_z(x=x, **kwargs)
            x = self.sample_x(z=z, **kwargs)
        return dict(x=x, z=z)

    def sample_gibbs(self, x=None, z=None, num_rep=1, **kwargs):
        """ One Step of Blocked Gibbs Sampler

        Returns:
            parameters (LGSSMParameters): sampled parameters after one step
        """
        latent_vars = self.sample_latent(x=x, z=z, num_rep=num_rep, **kwargs)
        new_parameters = self.message_helper.parameters_gibbs_sample(
                observations=self.observations,
                latent_vars=latent_vars,
                prior=self.prior,
                )
        self.parameters = new_parameters
        return self.parameters

    def noisy_loglikelihood(self, kind="complete",
            subsequence_length=-1,
            minibatch_size=1, buffer_length=10,
            latent_draws=1, latent_thinning=5,
            latent_burnin=5, latent_init=None,
            **kwargs):
        """ Approximation to loglikelihood (EM Lowerbound)

        Args:
            kind (string):
                "complete" - logPr(Y, | theta, X, Z) (default)
                "x_marginal" - logPr(Y | theta, X)
                "z_marginal" - logPr(Y | theta, Z)
            subsequence_length (int): length of subsequence used in evaluation
            minibatch_size (int): number of subsequences
            buffer_length (int): length of each subsequence buffer
            latent_draws (int): number of latent variable Monte Carlo draws in
                gradient approximation
            latent_thinning (int): number of steps between samples
            latent_burnin (int): number of burnin Gibb steps
            latent_init (string): latent variable initialization method
                See `self.init_sample_latent`

        """
        noisy_loglike = 0.0
        normalization_factor = 0.0
        for s in range(0, minibatch_size):
            # Get Subsequence and Buffer
            subsequence = self._random_subsequence_and_buffers(buffer_length,
                    subsequence_length)
            buffer_ = self.observations[subsequence['left_buffer_start']:
                    subsequence['right_buffer_end']]
            subsequence['buffer'] = buffer_

            if latent_init == "from_vector":
                z_init = kwargs['z_init'][subsequence['left_buffer_start']:
                        subsequence['right_buffer_end']]
            else:
                z_init = None

            # Run Blocked Gibbs on x_buffer, z_buffer
            # Init
            latent_buffer = self.init_sample_latent(
                    init_method=latent_init,
                    init_burnin=latent_burnin,
                    observations=buffer_,
                    track_samples=False,
                    z_init=z_init,
                    )

            for draw in range(latent_draws):
                if draw > 0 and latent_thinning > 0:
                    # Thinning
                    latent_buffer = self.sample_latent(
                            x=latent_buffer['x'], z=latent_buffer['z'],
                            num_rep=latent_thinning,
                            observations=buffer_,
                            track_samples=False,
                        )
                # Subsequence Objective Estimate
                noisy_loglike += self._subsequence_objective(
                        subsequence=subsequence,
                        x_buffer=latent_buffer['x'],
                        z_buffer=latent_buffer['z'],
                        kind=kind)

        # Average over Minibatch + Draws
        noisy_loglike *= 1.0/(minibatch_size*latent_draws)
        return noisy_loglike

    def _subsequence_objective(self, subsequence, x_buffer, z_buffer,
            kind="complete"):
        # Loglikelihood Approximation Calculator
        start = (subsequence['subsequence_start'] - \
                subsequence['left_buffer_start'])
        end = (subsequence['subsequence_end'] - \
                subsequence['left_buffer_start'])
        y = subsequence['buffer'][start:end]
        x = x_buffer[start:end]
        z = z_buffer[start:end]
        if kind == "complete":
            forward_message = {}
            if start > 0:
                forward_message['x_prev'] = x_buffer[start-1]
                forward_message['z_prev'] = z_buffer[start-1]
            loglikelihood = self.message_helper._complete_data_loglikelihood(
                    observations=y, x=x, z=z, parameters=self.parameters,
                    forward_message=forward_message,
                    backward_message=self.backward_message,
                    weights=subsequence['weights'],
                    )
        elif kind == "x_marginal":
            forward_message = (self
                    .message_helper
                    .forward_message(
                        observations=subsequence['buffer'][0:start],
                        x=x_buffer[0:start],
                        parameters=self.parameters,
                        forward_message=self.forward_message,
                    ))
            forward_message['log_constant'] = \
                    self.forward_message['z']['log_constant']
            loglikelihood = self.message_helper._z_marginal_loglikelihood(
                    observations=y, x=x, parameters=self.parameters,
                    forward_message=forward_message,
                    backward_message=self.backward_message,
                    weights=subsequence['weights'],
                    )
        elif kind == "z_marginal":
            forward_message = (self
                    .message_helper
                    .forward_message(
                        observations=subsequence['buffer'][0:start],
                        z=z_buffer[0:start],
                        parameters=self.parameters,
                        forward_message=self.forward_message,
                    ))
            forward_message['log_constant'] = \
                    self.forward_message['x']['log_constant']
            loglikelihood = self.message_helper._x_marginal_loglikelihood(
                    observations=y, z=z, parameters=self.parameters,
                    forward_message=forward_message,
                    backward_message=self.backward_message,
                    weights=subsequence['weights'],
                    )
        else:
            raise ValueError("Unrecognized kind = {0}".format(kind))

        return loglikelihood

    def noisy_gradient(self, kind="complete",
            subsequence_length=-1, minibatch_size=1, buffer_length=0,
            latent_draws=1, latent_thinning=5, latent_burnin=5,
            latent_init=None, preconditioner=None, is_scaled=True,
            **kwargs):
        """ Noisy Gradient Estimate

        Monte Carlo Estimate of gradient (using buffering)

            Runs Gibbs on buffered sequence

        Args:
            kind (string): type of gradient
                "complete" - grad logPr(Y, Xhat, Zhat | theta) (default)
                "x_marginal" - grad logPr(Y, Xhat | theta)
                "z_marginal" - grad logPr(Y, Zhat | theta)
            minibatch_size (int): number of subsequences
            buffer_length (int): length of each subsequence buffer
            latent_draws (int): number of latent variable Monte Carlo draws in
                gradient approximation
            latent_thinning (int): number of steps between samples
            latent_burnin (int): number of burnin Gibb steps
            latent_init (string): latent variable initialization method
                See `self.init_sample_latent`
            preconditioner (object): preconditioner for gradients
            use_analytic (boolean): use analytic gradient instead of autograd
            is_scaled (boolean): scale gradient by 1/T

        Returns:
            noisy_gradient (ndarray): gradient vector

        """
        noisy_grad_loglike = \
                self._noisy_grad_loglikelihood(
                        subsequence_length=subsequence_length,
                        minibatch_size=minibatch_size,
                        buffer_length=buffer_length,
                        kind=kind, latent_draws=latent_draws,
                        latent_thinning=latent_thinning,
                        latent_burnin=latent_burnin, latent_init=latent_init,
                        **kwargs)

        noisy_grad_prior = self.prior.grad_logprior(
                parameters=self.parameters)
        noisy_gradient = {var: noisy_grad_prior[var] + noisy_grad_loglike[var]
                for var in noisy_grad_prior}

        if preconditioner is None:
            if is_scaled:
                for var in noisy_gradient:
                    noisy_gradient[var] /= self.T
        else:
            scale = 1.0/self.T if is_scaled else 1.0
            noisy_gradient = preconditioner.precondition(noisy_gradient,
                    parameters=self.parameters,
                    scale=scale)

        return noisy_gradient

    def _noisy_grad_loglikelihood(self, subsequence_length=-1,
        minibatch_size=1, buffer_length=0, kind='complete',
        latent_draws=1, latent_thinning=5, latent_burnin=5, latent_init=None,
        **kwargs):
        # Noisy Gradient
        noisy_grad = {var: np.zeros_like(value)
                for var, value in self.parameters.as_dict().items()}

        for s in range(0, minibatch_size):
            # Get Subsequence and Buffer
            subsequence = self._random_subsequence_and_buffers(buffer_length,
                    subsequence_length=subsequence_length)
            buffer_ = self.observations[subsequence['left_buffer_start']:
                    subsequence['right_buffer_end']]
            subsequence['buffer'] = buffer_

            if latent_init == "from_vector":
                z_init = kwargs['z_init'][
                        subsequence['left_buffer_start']:\
                                subsequence['right_buffer_end']
                                ]
            else:
                z_init = None

            # Run Blocked Gibbs on x_buffer, z_buffer
            # Init
            latent_buffer = self.init_sample_latent(
                    init_method=latent_init,
                    init_burnin = latent_burnin,
                    observations=subsequence['buffer'],
                    track_samples=False,
                    z_init=z_init,
                    )

            for draw in range(latent_draws):
                if draw > 0 and latent_thinning > 0:
                    # Thinning
                    latent_buffer = self.sample_latent(
                            x=latent_buffer['x'],
                            z=latent_buffer['z'],
                            num_rep=latent_thinning,
                            observations=subsequence['buffer'],
                            track_samples=False,
                        )

                # Subsequence Gradient Estimate
                noisy_grad_add = self._subsequence_gradient(
                        subsequence=subsequence,
                        x_buffer=latent_buffer['x'],
                        z_buffer=latent_buffer['z'],
                        kind=kind,
                        )

        for var in noisy_grad:
            noisy_grad[var] *= 1.0 / (minibatch_size*latent_draws)
            if np.any(np.isnan(noisy_grad[var])):
                raise ValueError("NaNs in gradient of {0}".format(var))
            if np.linalg.norm(noisy_grad[var]) > 1e16:
                logger.warning("Norm of noisy_grad_loglike[{1} > 1e16: {0}".format(
                    noisy_grad_loglike, var))
        return noisy_grad

    def _subsequence_gradient(self, subsequence, x_buffer, z_buffer, kind):
        """ Forward + Backward Messages + Subsequence Gradient """
        start = (subsequence['subsequence_start'] - \
                subsequence['left_buffer_start'])
        end = (subsequence['subsequence_end'] - \
                subsequence['left_buffer_start'])
        y = subsequence['buffer'][start:end]
        x = x_buffer[start:end]
        z = z_buffer[start:end]

        if kind == "complete":
            # Naive: grad log Pr(y, x, z | theta)
            if start > 0:
                forward_message = {
                        'x_prev': x_buffer[start-1],
                        'z_prev': z_buffer[start-1],
                        }
            else:
                forward_message = {}
            if end < np.shape(subsequence['buffer'])[0]:
                backward_message = {
                        'x_next': x_buffer[end],
                        'z_next': z_buffer[end],
                        }
            else:
                backward_message = {}

            noisy_grad_loglike = (self
                    .message_helper
                    ._gradient_complete_data_loglikelihood(
                        observations=y,
                        x=x, z=z,
                        parameters=self.parameters,
                        forward_message=forward_message,
                        backward_message=backward_message,
                        weights=subsequence['weights'],
                    ))

        elif kind == "x_marginal":
            # X: grad log Pr(y, x | theta)
            forward_message = (self
                    .message_helper
                    .forward_message(
                        observations=subsequence['buffer'][0:start],
                        x=x_buffer[0:start],
                        parameters=self.parameters,
                        forward_message=self.forward_message,
                    ))
            if end < np.shape(subsequence['buffer'])[0]:
                backward_message = (self
                        .message_helper
                        .backward_message(
                            observations=subsequence['buffer'][end:],
                            x=x_buffer[end:],
                            parameters=self.parameters,
                            backward_message=self.backward_message,
                        ))
            else:
                backward_message = self.backward_message
            noisy_grad_loglike = (self
                    .message_helper
                    ._z_gradient_marginal_loglikelihood(
                        observations=y, x=x,
                        parameters=self.parameters,
                        forward_message=forward_message,
                        backward_message=backward_message,
                        weights=subsequence['weights'],
                    ))

        elif kind == "z_marginal":
            # Z: grad log Pr(y, z | theta)
            forward_message = (self
                    .message_helper
                    .forward_message(
                        observations=subsequence['buffer'][0:start],
                        z=z_buffer[0:start],
                        parameters=self.parameters,
                        forward_message=self.forward_message,
                    ))
            if end < np.shape(subsequence['buffer'])[0]:
                backward_message = (self
                        .message_helper
                        .backward_message(
                            observations=subsequence['buffer'][end:],
                            z=z_buffer[end:],
                            parameters=self.parameters,
                            backward_message=self.backward_message,
                        ))
            else:
                backward_message = self.backward_message
            gradient_kwargs = dict(
                        observations=y, z=z,
                        parameters=self.parameters,
                        forward_message=forward_message,
                        backward_message=backward_message,
                    )
            noisy_grad_loglike = (self
                    .message_helper
                    ._x_gradient_marginal_loglikelihood(
                        observations=y, z=z,
                        parameters=self.parameters,
                        forward_message=forward_message,
                        backward_message=backward_message,
                        weights=subsequence['weights'],
                        ))
        else:
            raise ValueError("Unrecognized kind = {0}".format(kind))

        return noisy_grad_loglike

