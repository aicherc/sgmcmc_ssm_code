import numpy as np
import logging
logger = logging.getLogger(name=__name__)

from ...sgmcmc_sampler import SGMCMCHelper
from .kernels import (
        LGSSMPriorKernel,
        LGSSMOptimalKernel,
        LGSSMHighDimOptimalKernel,
        )
from ...particle_filters.buffered_smoother import (
        buffered_pf_wrapper,
        average_statistic,
        )
from ..._utils import lower_tri_mat_inv

class LGSSMHelper(SGMCMCHelper):
    """ LGSSM Helper

        forward_message (dict) with keys
            log_constant (double) log scaling const
            mean_precision (ndarray) mean precision
            precision (ndarray) precision

        backward_message (dict) with keys
            log_constant (double) log scaling const
            mean_precision (ndarray) mean precision
            precision (ndarray) precision
    """
    def __init__(self, n, m, forward_message=None, backward_message=None,
            **kwargs):
        self.n = n
        self.m = m

        if forward_message is None:
             forward_message = {
                    'log_constant': 0.0,
                    'mean_precision': np.zeros(self.n),
                    'precision': np.eye(self.n)/10,
                    }
        self.default_forward_message=forward_message

        if backward_message is None:
            backward_message = {
                'log_constant': 0.0,
                'mean_precision': np.zeros(self.n),
                'precision': np.zeros((self.n, self.n)),
                }
        self.default_backward_message=backward_message
        return

    ## Message Passing
    def _forward_messages(self, observations, parameters, forward_message,
            weights=None, tqdm=None, only_return_last=False):
        # Return list of forward messages Pr(x_{t} | y_{<=t})
        # y is num_obs x m matrix
        num_obs = np.shape(observations)[0]
        if not only_return_last:
            forward_messages = [None]*(num_obs+1)
            forward_messages[0] = forward_message

        mean_precision = forward_message['mean_precision']
        precision = forward_message['precision']
        log_constant = forward_message['log_constant']

        A = parameters.A
        LQinv = parameters.LQinv
        Qinv = np.dot(LQinv, LQinv.T)
        AtQinv = np.dot(A.T, Qinv)
        AtQinvA = np.dot(AtQinv, A)
        C = parameters.C
        LRinv = parameters.LRinv
        Rinv = np.dot(LRinv, LRinv.T)
        CtRinv = np.dot(C.T, Rinv)
        CtRinvC = np.dot(CtRinv, C)

        pbar = range(num_obs)
        if tqdm is not None:
            pbar = tqdm(pbar, leave=False)
            pbar.set_description("forward messages")
        for t in pbar:
            y_cur = observations[t]
            weight_t = 1.0 if weights is None else weights[t]

            # Calculate Predict Parameters
            J = np.linalg.solve(AtQinvA + precision, AtQinv)
            pred_mean_precision = np.dot(J.T, mean_precision)
            pred_precision = Qinv - np.dot(AtQinv.T, J)

            # Calculate Observation Parameters
            y_mean = np.dot(C,
                    np.linalg.solve(pred_precision, pred_mean_precision))
            y_precision = Rinv - np.dot(CtRinv.T,
                    np.linalg.solve(CtRinvC + pred_precision, CtRinv))
            log_c = (-0.5 * np.dot(y_cur-y_mean,
                            np.dot(y_precision, y_cur-y_mean)) + \
                     0.5 * np.linalg.slogdet(y_precision)[1] + \
                    -0.5 * self.m * np.log(2*np.pi))
            log_constant += log_c * weight_t

            # Calculate Filtered Parameters
            new_mean_precision = pred_mean_precision + np.dot(CtRinv, y_cur)
            new_precision = pred_precision + CtRinvC

            # Save Messages
            mean_precision = new_mean_precision
            precision = new_precision
            if not only_return_last:
                forward_messages[t+1] = {
                    'mean_precision': mean_precision,
                    'precision': precision,
                    'log_constant': log_constant,
                }
        if only_return_last:
            last_message = {
                    'mean_precision': mean_precision,
                    'precision': precision,
                    'log_constant': log_constant,
                }
            return last_message
        else:
            return forward_messages

    def _backward_messages(self, observations, parameters, backward_message,
            weights=None, tqdm=None, only_return_last=False):
        # Return list of backward messages Pr(y_{>t} | x_t)
        # y is num_obs x n matrix
        num_obs = np.shape(observations)[0]
        if not only_return_last:
            backward_messages = [None]*(num_obs+1)
            backward_messages[-1] = backward_message

        mean_precision = backward_message['mean_precision']
        precision = backward_message['precision']
        log_constant = backward_message['log_constant']

        A = parameters.A
        LQinv = parameters.LQinv
        Qinv = np.dot(LQinv, LQinv.T)
        AtQinv = np.dot(A.T, Qinv)
        AtQinvA = np.dot(AtQinv, A)
        C = parameters.C
        LRinv = parameters.LRinv
        Rinv = np.dot(LRinv, LRinv.T)
        CtRinv = np.dot(C.T, Rinv)
        CtRinvC = np.dot(CtRinv, C)

        pbar = reversed(range(num_obs))
        if tqdm is not None:
            pbar = tqdm(pbar, total=num_obs, leave=False)
            pbar.set_description("backward messages")
        for t in pbar:
            y_cur = observations[t]
            weight_t = 1.0 if weights is None else weights[t]

            # Helper Values
            xi = Qinv + precision + CtRinvC
            L = np.linalg.solve(xi, AtQinv.T)
            vi = mean_precision + np.dot(CtRinv, y_cur)

            # Calculate new parameters
            log_c = (-0.5 * self.m * np.log(2.0*np.pi) + \
                    np.sum(np.log(np.diag(LRinv))) + \
                    np.sum(np.log(np.diag(LQinv))) + \
                    -0.5 * np.linalg.slogdet(xi)[1] + \
                    -0.5 * np.dot(y_cur, np.dot(Rinv, y_cur)) + \
                    0.5 * np.dot(vi, np.linalg.solve(xi, vi)))

            log_constant += log_c * weight_t

            new_mean_precision = np.dot(L.T, vi)
            new_precision = AtQinvA - np.dot(AtQinv, L)

            # Save Messages
            mean_precision = new_mean_precision
            precision = new_precision

            if not only_return_last:
                backward_messages[t] = {
                    'mean_precision': mean_precision,
                    'precision': precision,
                    'log_constant': log_constant,
                }
        if only_return_last:
            last_message = {
                    'mean_precision': mean_precision,
                    'precision': precision,
                    'log_constant': log_constant,
                }
            return last_message
        else:
            return backward_messages

    ## Loglikelihood Functions
    def marginal_loglikelihood(self, observations, parameters,
            forward_message=None, backward_message=None, weights=None,
            tqdm=None, **kwargs):
        # Run forward pass + combine with backward pass
        # y is num_obs x n matrix
        if forward_message is None:
            forward_message = self.default_forward_message
        if backward_message is None:
            backward_message = self.default_backward_message

        # forward_pass is Pr(x_{T-1} | y_{<=T-1})
        forward_message = self._forward_message(
                observations=observations,
                parameters=parameters,
                forward_message=forward_message,
                weights=weights,
                tqdm=tqdm,
                **kwargs)

        # Calculate the marginal loglikelihood of forward + backward message
        f_mean_precision = forward_message['mean_precision']
        f_precision = forward_message['precision']
        c_mean_precision = f_mean_precision + backward_message['mean_precision']
        c_precision = f_precision + backward_message['precision']
        weight = 1.0 if weights is None else weights[-1]

        loglikelihood = forward_message['log_constant'] + \
                (backward_message['log_constant'] + \
                +0.5 * np.linalg.slogdet(f_precision)[1] + \
                -0.5 * np.linalg.slogdet(c_precision)[1] + \
                -0.5 * np.dot(f_mean_precision,
                        np.linalg.solve(f_precision, f_mean_precision)
                    ) + \
                0.5 * np.dot(c_mean_precision,
                    np.linalg.solve(c_precision, c_mean_precision)
                    )
                ) * weight

        return loglikelihood

    def complete_data_loglikelihood(self, observations, latent_vars, parameters,
            forward_message=None, weights=None, **kwargs):
        if forward_message is None:
            forward_message = self.default_forward_message

        log_constant = 0.0
        A = parameters.A
        LQinv = parameters.LQinv
        C = parameters.C
        LRinv = parameters.LRinv

        x_prev = forward_message.get('x_prev')
        for t, (y_t, x_t) in enumerate(zip(observations, latent_vars)):
            weight_t = 1.0 if weights is None else weights[t]

            # Pr(X_t | X_t-1)
            if (x_prev is not None):
                diffLQinv = np.dot(x_t - np.dot(A,x_prev), LQinv)
                log_c = (-0.5 * self.n * np.log(2*np.pi) + \
                        -0.5 * np.dot(diffLQinv, diffLQinv) + \
                        np.sum(np.log(np.diag(LQinv))))
                log_constant += log_c * weight_t

            # Pr(Y_t | X_t)
            LRinvTymCx = np.dot(LRinv.T, y_t - np.dot(C, x_t))
            log_c = (-0.5 * self.m * np.log(2*np.pi) + \
                    -0.5*np.dot(LRinvTymCx, LRinvTymCx) + \
                    np.sum(np.log(np.diag(LRinv))))
            log_constant += log_c * weight_t
            x_prev = x_t

        return log_constant

    def predictive_loglikelihood(self, observations, parameters, lag=1,
            forward_message=None, backward_message=None, tqdm=None, **kwargs):
        if forward_message is None:
            forward_message = self.default_forward_message
        if backward_message is None:
            backward_message = self.default_backward_message

        # Calculate Filtered
        if lag == 0:
            forward_messages = self.forward_pass(observations,
                    parameters, forward_message, tqdm=tqdm, **kwargs)
        else:
            forward_messages = self.forward_pass(observations[0:-lag],
                    parameters, forward_message, tqdm=tqdm, **kwargs)
        loglike = 0.0
        A = parameters.A
        Q = parameters.Q
        C = parameters.C
        R = parameters.R
        pbar = range(lag, np.shape(observations)[0])
        if tqdm is not None:
            pbar = tqdm(pbar, leave=False)
            pbar.set_description('predictive_loglikelihood')
        for t in pbar:
            # Calculate Pr(x_t | y_{<=t-lag}, theta)
            mean_precision = forward_messages[t-lag]['mean_precision']
            precision = forward_messages[t-lag]['precision']
            mean = np.linalg.solve(precision, mean_precision)
            var = np.linalg.inv(precision)
            for l in range(lag):
                mean = np.dot(A, mean)
                var = np.dot(A, np.dot(var, A.T)) + Q

            y_mean = np.dot(C, mean)
            y_var = np.dot(C, np.dot(var, C.T)) + R
            y_cur = observations[t]
            log_like_t = -0.5 * np.dot(y_cur - y_mean,
                    np.linalg.solve(y_var, y_cur - y_mean)) + \
                        -0.5 * np.linalg.slogdet(y_var)[1] + \
                        -0.5 * self.m * np.log(2*np.pi)
            loglike += log_like_t
        return loglike

    ## Gradient Functions
    def gradient_marginal_loglikelihood(self, observations, parameters,
            forward_message=None, backward_message=None, weights=None,
            include_init=True, tqdm=None):
        A, LQinv, C, LRinv = \
                parameters.A, parameters.LQinv, parameters.C, parameters.LRinv

        # Forward Pass
        # forward_messages = [Pr(x_{t} | y_{-inf:t}), y{t}] for t = -1,...,T-1
        forward_messages = self.forward_pass(observations,
                parameters, forward_message,
                include_init_message=True)

        # Backward Pass
        # backward_messages = [Pr(y_{t+1:inf} | x_{t}), Y_{t}] for t =-1,...,T-1
        backward_messages = self.backward_pass(observations,
                parameters, backward_message,
                include_init_message=True)

        # Gradients
        A_grad = np.zeros_like(A)
        C_grad = np.zeros_like(C)
        LQinv_grad = np.zeros_like(LQinv)
        LRinv_grad = np.zeros_like(LRinv)

        # Helper Constants
        Rinv = np.dot(LRinv, LRinv.T)
        RinvC = np.dot(Rinv, C)
        CtRinvC = np.dot(C.T, RinvC)
        LRinv_diaginv = np.diag(np.diag(LRinv)**-1)

        Qinv = np.dot(LQinv, LQinv.T)
        QinvA = np.dot(Qinv, A)
        AtQinvA = np.dot(A.T, QinvA)
        LQinv_diaginv = np.diag(np.diag(LQinv)**-1)

        # Emission Gradients
        p_bar = zip(forward_messages[1:], backward_messages[1:], observations)
        if tqdm is not None:
            p_bar = tqdm(p_bar, total=np.shape(observations)[0], leave=False)
            p_bar.set_description("gradient loglike")
        for t, (forward_t, backward_t, y_t) in enumerate(p_bar):
            weight_t = 1.0 if weights is None else weights[t]

            # Pr(x_t | y)
            c_mean_precision = \
                    forward_t['mean_precision'] + backward_t['mean_precision']
            c_precision = \
                    forward_t['precision'] + backward_t['precision']

            x_mean = np.linalg.solve(c_precision, c_mean_precision)
            xxt_mean = np.linalg.inv(c_precision) + np.outer(x_mean, x_mean)

            # Gradient of C
            C_grad += weight_t * (np.outer(np.dot(Rinv, y_t), x_mean) + \
                    -1.0 * np.dot(RinvC, xxt_mean))

            # Gradient of LRinv
            Cxyt = np.outer(np.dot(C, x_mean), y_t)
            CxxtCt = np.dot(C, np.dot(xxt_mean, C.T))
            LRinv_grad += weight_t * (LRinv_diaginv + \
                -1.0*np.dot(np.outer(y_t, y_t) - Cxyt - Cxyt.T + CxxtCt, LRinv)
                )

        # Transition Gradients
        if include_init:
            pbar = zip(
                forward_messages[:-1], backward_messages[1:], observations)
        else:
            pbar = zip(
                forward_messages[1:-1], backward_messages[2:], observations[1:])
        for t, (forward_t, backward_t, y_t) in enumerate(pbar):
            weight_t = 1.0 if weights is None else weights[t]
            # Pr(x_t, x_t+1 | y)
            c_mean_precision = \
                np.concatenate([
                    forward_t['mean_precision'],
                    backward_t['mean_precision'] + np.dot(RinvC.T,y_t)
                    ])
            c_precision = \
                np.block([
                    [forward_t['precision'] + AtQinvA, -QinvA.T],
                    [-QinvA, backward_t['precision'] + CtRinvC + Qinv]
                    ])

            c_mean = np.linalg.solve(c_precision, c_mean_precision)
            c_cov = np.linalg.inv(c_precision)

            xp_mean = c_mean[0:self.n]
            xn_mean = c_mean[self.n:]
            xpxpt_mean = c_cov[0:self.n, 0:self.n] + np.outer(xp_mean, xp_mean)
            xnxpt_mean = c_cov[self.n:, 0:self.n] + np.outer(xn_mean, xp_mean)
            xnxnt_mean = c_cov[self.n:, self.n:] + np.outer(xn_mean, xn_mean)

            # Gradient of A
            A_grad += weight_t * np.dot(Qinv, xnxpt_mean - np.dot(A,xpxpt_mean))

            # Gradient of LQinv
            Axpxnt = np.dot(A, xnxpt_mean.T)
            AxpxptAt = np.dot(A, np.dot(xpxpt_mean, A.T))
            LQinv_grad += weight_t * (LQinv_diaginv + \
                -1.0*np.dot(xnxnt_mean - Axpxnt - Axpxnt.T + AxpxptAt, LQinv))

        grad = dict(
                A=A_grad,
                LQinv_vec=LQinv_grad[np.tril_indices_from(LQinv_grad)],
                C=C_grad,
                LRinv_vec=LRinv_grad[np.tril_indices_from(LRinv_grad)],
                )
        return grad

    def gradient_complete_data_loglikelihood(self, observations, latent_vars,
            parameters, forward_message=None, weights=None, tqdm=None,
            **kwargs):
        if forward_message is None:
            forward_message = self.default_forward_message
        A = parameters.A
        LQinv = parameters.LQinv
        Qinv = parameters.Qinv
        LQinv_Tinv = lower_tri_mat_inv(LQinv).T
        C = parameters.C
        LRinv = parameters.LRinv
        Rinv = parameters.Rinv
        LRinv_Tinv = lower_tri_mat_inv(LRinv).T

        # Gradients
        grad = {var: np.zeros_like(value)
                for var, value in parameters.as_dict().items()}

        if len(np.shape(latent_vars)) == 2:
            # Only One Sample
            # Transition Gradients
            x_prev = forward_message.get('x_prev')
            for t, x_t in enumerate(latent_vars):
                weight_t = 1.0 if weights is None else weights[t]
                if x_prev is not None:
                    diff = x_t - np.dot(A, x_prev)
                    grad['A'] += weight_t * np.outer(np.dot(Qinv, diff), x_prev)
                    grad['LQinv'] += weight_t * (LQinv_Tinv + \
                        -1.0*np.dot(np.outer(diff, diff), LQinv))
                x_prev = x_t

            # Emission Gradients
            for t, (x_t, y_t) in enumerate(zip(latent_vars, observations)):
                weight_t = 1.0 if weights is None else weights[t]
                diff = y_t - np.dot(C, x_t)
                grad['C'] += weight_t * np.outer(np.dot(Rinv, diff), x_t)
                grad['LRinv'] += weight_t * (LRinv_Tinv + \
                    -1.0*np.dot(np.outer(diff, diff), LRinv))
        elif len(np.shape(latent_vars)) == 3:
            # Average over Multiple Latent Vars
            num_samples = np.shape(latent_vars)[2]

            # Transition Gradients
            x_prev = forward_message.get('x_prev')
            for t, x_t in enumerate(latent_vars):
                weight_t = 1.0 if weights is None else weights[t]
                if x_prev is not None:
                    diff = x_t - np.dot(A, x_prev)
                    grad['A'] += weight_t * np.dot(Qinv,
                            np.dot(diff, x_prev.T))/num_samples
                    grad['LQinv'] += weight_t * (LQinv_Tinv + \
                        -1.0*np.dot(np.dot(diff, diff.T), LQinv)/num_samples)
                x_prev = x_t

            # Emission Gradients
            for t, (x_t, y_t_) in enumerate(zip(latent_vars, observations)):
                y_t = np.array([y_t_ for _ in range(num_samples)]).T
                weight_t = 1.0 if weights is None else weights[t]

                diff = y_t - np.dot(C, x_t)
                grad['C'] += weight_t * (np.dot(Rinv,
                    np.dot(diff, x_t.T))/num_samples)
                grad['LRinv'] += weight_t * (LRinv_Tinv + \
                    -1.0*np.dot(np.dot(diff, diff.T), LRinv)/num_samples)
        else:
            raise ValueError("Incorrect latent_var shape")

        grad['LQinv_vec'] = grad.pop('LQinv')[np.tril_indices(self.n)]
        grad['LRinv_vec'] = grad.pop('LRinv')[np.tril_indices(self.m)]
        return grad

    def gradient_loglikelihood(self, kind='marginal', **kwargs):
        if kind == 'marginal':
            return self.gradient_marginal_loglikelihood(**kwargs)
        elif kind == 'complete':
            return self.gradient_complete_data_loglikelihood(**kwargs)
        else:
            raise ValueError("Unrecognized `kind' {0}".format(kind))

    ## Gibbs Functions
    def calc_gibbs_sufficient_statistic(self, observations, latent_vars,
            **kwargs):
        """ Gibbs Sample Sufficient Statistics
        Args:
            observations (ndarray): num_obs observations
            latent_vars (ndarray): latent vars

        Returns:
            sufficient_stat (dict)
        """
        x = latent_vars
        y = observations

        # Sufficient Statistics for A and Q
        # From Emily Fox's Thesis Page 147
        PsiT = x[1:]
        PsiT_prev = x[:-1]
        transition_count = len(PsiT)
        Sx_prevprev = PsiT_prev.T.dot(PsiT_prev)
        Sx_curprev = PsiT.T.dot(PsiT_prev)
        Sx_curcur = PsiT.T.dot(PsiT)

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
        sufficient_stat['A'] = dict(
                S_prevprev=Sx_prevprev,
                S_curprev=Sx_curprev,
                )
        sufficient_stat['Q'] = dict(
                S_count=transition_count,
                S_prevprev=Sx_prevprev,
                S_curprev=Sx_curprev,
                S_curcur=Sx_curcur,
                )
        sufficient_stat['R'] = dict(
                S_count=emission_count,
                S_prevprev=S_prevprev,
                S_curprev=S_curprev,
                S_curcur=S_curcur,
                )
        sufficient_stat['C'] = dict(
                S_prevprev=S_prevprev,
                S_curprev=S_curprev,
                )
        return sufficient_stat

    ## Predict Functions
    def latent_var_distr(self, observations, parameters,
            distr='marginal', lag=None,
            forward_message=None, backward_message=None,
            tqdm=None):
        if distr != 'marginal':
            raise NotImplementedError()
        if forward_message is None:
            forward_message = self.default_forward_message
        if backward_message is None:
            backward_message = self.default_backward_message

        # Setup Output
        L = np.shape(observations)[0]
        mean = np.zeros((L, self.n))
        cov = np.zeros((L, self.n, self.n))

        # Forward Pass
        forward_messages = self.forward_pass(
                observations=observations,
                parameters=parameters,
                forward_message=forward_message,
                tqdm=tqdm
                )

        pbar = range(L)
        if tqdm is not None:
            pbar = tqdm(pbar, leave=False)
            pbar.set_description('calc latent var distr')

        if lag is None:
            # Smoothing
            backward_messages = self.backward_pass(
                observations=observations,
                parameters=parameters,
                backward_message=backward_message,
                tqdm=tqdm
                )
            for t in pbar:
                mean_precision = \
                        forward_messages[t]['mean_precision'] + \
                        backward_messages[t]['mean_precision']
                precision = \
                        forward_messages[t]['precision'] + \
                        backward_messages[t]['precision']

                mean[t] = np.linalg.solve(precision, mean_precision)
                cov[t] = np.linalg.inv(precision)
            return mean, cov

        elif lag <= 0:
            # Prediction/Filtering
            A, Q = parameters.A, parameters.Q
            for t in pbar:
                if t+lag >= 0:
                    mean_precision = forward_messages[t+lag]['mean_precision']
                    precision = forward_messages[t+lag]['precision']
                else:
                    mean_precision = forward_message['mean_precision']
                    precision = forward_message['precision']
                mean_lag = np.linalg.solve(precision, mean_precision)
                cov_lag = np.linalg.inv(precision)

                # Forward Simulate
                for _ in range(-lag):
                    mean_lag = np.dot(A, mean_lag)
                    cov_lag = np.dot(np.dot(A, cov_lag), A.T) + Q

                mean[t] = mean_lag
                cov[t] = cov_lag
            return mean, cov

        else:
            # Fixed-lag Smoothing
            for t in pbar:
                # Backward Messages
                back_obs = observations[t:min(t+lag, L)]
                fixed_lag_message = self.backward_message(
                        observations=back_obs,
                        parameters=parameters,
                        backward_message=backward_message,
                        )
                # Output
                mean_precision = \
                        forward_messages[t]['mean_precision'] + \
                        fixed_lag_message['mean_precision']
                precision = \
                        forward_messages[t]['precision'] + \
                        fixed_lag_message['precision']
                mean[t] = np.linalg.solve(precision, mean_precision)
                cov[t] = np.linalg.inv(precision)
            return mean, cov

    def latent_var_sample(self, observations, parameters,
            forward_message=None, backward_message=None,
            distr='joint', lag=None, num_samples=None,
            tqdm=None, include_init=False, **kwargs):
        if forward_message is None:
            forward_message = self.default_forward_message
        if backward_message is None:
            backward_message = self.default_backward_message
        if distr == 'joint' and lag is not None:
            raise ValueError("Must set distr to 'marginal' for lag != None")

        A, LQinv = parameters.A, parameters.LQinv
        AtQinv = np.dot(A.T, np.dot(LQinv, LQinv.T))
        AtQinvA = np.dot(AtQinv, A)

        if distr == 'joint':
            # Forward Pass
            forward_messages = self.forward_pass(
                    observations=observations,
                    parameters=parameters,
                    forward_message=forward_message,
                    include_init_message=include_init,
                    tqdm=tqdm
                    )
            L = len(forward_messages)

            if num_samples is not None:
                x = np.zeros((L, self.n, num_samples))
            else:
                x = np.zeros((L, self.n))

            # Backward Sampler
            x_cov = np.linalg.inv(forward_messages[-1]['precision'])
            x_mean = np.dot(x_cov, forward_messages[-1]['mean_precision'])
            x[-1] = np.random.multivariate_normal(mean=x_mean, cov=x_cov,
                    size=num_samples).T

            pbar = reversed(range(L-1))
            if tqdm is not None:
                pbar = tqdm(pbar, leave=False)
                pbar.set_description("backward smoothed sampling x")
            for t in pbar:
                x_next = x[t+1]
                x_cov = np.linalg.inv(forward_messages[t]['precision'] +
                        AtQinvA)
                if num_samples is None:
                    x_mean = np.dot(x_cov,
                            forward_messages[t]['mean_precision'] + \
                                np.dot(AtQinv, x_next))
                else:
                    x_mean = np.dot(x_cov,
                            np.outer(forward_messages[t]['mean_precision'],
                                np.ones(num_samples)) + \
                                np.dot(AtQinv, x_next))
                x[t] = x_mean + np.random.multivariate_normal(
                        mean=np.zeros(self.n), cov=x_cov, size=num_samples,
                        ).T
            return x

        elif distr == 'marginal':
            # Calculate Distribution
            x_mean, x_cov = self.latent_var_distr(observations, parameters,
                    lag=lag, forward_message=forward_message,
                    backward_message=backward_message, tqdm=tqdm,
                    )
            # Sample from Distribution
            L = x_mean.shape[0]
            if num_samples is not None:
                x = np.zeros((x_mean.shape[0], self.n, num_samples))
            else:
                x = np.zeros((x_mean.shape[0], self.n))
            pbar = reversed(range(L))
            if tqdm is not None:
                pbar = tqdm(pbar, leave=False)
                pbar.set_description("sampling x")
            for t in pbar:
                x[t] = x_mean[t] + np.random.multivariate_normal(
                        mean=np.zeros(self.n), cov=x_cov[t], size=num_samples,
                        ).T

        else:
            raise ValueError("Unrecognized `distr'; {0}".format(distr))
        return

    def latent_var_pairwise_marginal(self, observations, parameters,
            forward_message=None, backward_message=None,
            distribution='smoothed', tqdm=None):
        """ Calculate latent var marginal distribution

        Args:
            observations (ndarray): num_obs by n observations
            parameters (LGSSMParameters): parameters
            forward_message (dict): alpha message
                (e.g. Pr(x_{-1} | y_{-inf:-1}))
            backward_message (dict): beta message
                (e.g. Pr(y_{T:inf} | x_{T-1}))
                'likelihood_vector' (ndarray) dimension num_states
                'y_next' (ndarray) dimension p by m, optional
            distr (string): 'smoothed', 'filtered', 'predict'
                smoothed: Pr(X | Y, theta)
                filtered: Pr(X_t | Y_<=t, theta)
                predictive:Pr(X_t | Y_<t, theta)

        Returns
            mean (ndarray): num_obs by 2n, pairwise mean (x_t, x_{t+1})
            cov (ndarray): num_obs by 2n by  2n, pairwise covariance
        """
        if forward_message is None:
            forward_message = self.default_forward_message
        if backward_message is None:
            backward_message = self.default_backward_message

        num_obs = np.shape(observations)[0]
        mean = np.zeros((num_obs, 2*self.n))
        cov = np.zeros((num_obs, 2*self.n, 2*self.n))
        if distribution == 'smoothed':
            forward_messages = self.forward_pass(
                    observations=observations,
                    parameters=parameters,
                    forward_message=forward_message,
                    include_init_message=True,
                    tqdm=tqdm
                    )
            backward_messages = self.backward_pass(
                    observations=observations,
                    parameters=parameters,
                    backward_message=backward_message,
                    include_init_message=True,
                    tqdm=tqdm
                    )
            # Helper Constants
            C = parameters.C
            Rinv = parameters.Rinv
            RinvC = np.dot(Rinv, C)
            CtRinvC = np.dot(C.T, RinvC)

            A = parameters.A
            Qinv = parameters.Qinv
            QinvA = np.dot(Qinv, A)
            AtQinvA = np.dot(A.T, QinvA)

            for t in range(num_obs):
                y_t = observations[t]
                mean_precision = \
                    np.concatenate([
                        forward_messages[t]['mean_precision'],
                        backward_messages[t+1]['mean_precision'] + \
                                np.dot(RinvC.T, y_t)
                        ])
                precision = \
                    np.block([
                        [forward_messages[t]['precision'] + AtQinvA, -QinvA.T],
                        [-QinvA,
                        backward_messages[t+1]['precision'] + CtRinvC + Qinv]
                        ])

                mean[t] = np.linalg.solve(precision, mean_precision)
                cov[t] = np.linalg.inv(precision)
            return mean, cov

        elif distribution == 'filtered':
            raise NotImplementedError()
        elif distribution == 'predictive':
            raise NotImplementedError()

        else:
            raise ValueError("Invalid `distribution'; {0}".format(distribution))
        return

    def y_distr(self, observations, parameters,
            distr='marginal', lag=None,
            forward_message=None, backward_message=None,
            latent_vars=None, tqdm=None):
        if forward_message is None:
            forward_message = self.default_forward_message
        if backward_message is None:
            backward_message = self.default_backward_message
        if latent_vars is not None:
            x_mean, x_cov = self.latent_var_distr(
                    observations=observations,
                    parameters=parameters,
                    distr=distr,
                    lag=lag,
                    forward_message=forward_message,
                    backward_message=backward_message,
                    latent_vars=latent_vars,
                    tqdm=tqdm,
                    )
            C, R = parameters.C, parameters.R
            y_mean = x_mean.dot(C.T)
            y_cov = np.array([C.dot(x_cov_t).dot(C.T) + R for x_cov_t in x_cov])
            return y_mean, y_cov
        else:
            if lag is None or lag >= 0:
                C, R = parameters.C, parameters.R
                y_mean = latent_vars.dot(C.T)
                y_cov = np.array([R for _ in range(L)])
                return y_mean, y_cov
            else:
                L = observations.shape[0]
                y_mean = np.nan(L, self.m)
                y_cov = np.nan(L, self.m, self.m)
                A, Q = parameters.A, parameters.Q
                C, R = parameters.C, parameters.R

                # Apply the transition + noise for lag steps
                tran = np.eye(self.m)
                cov = np.zeros(self.m, self.m)
                for _ in range(-lag):
                    tran = A.dot(tran)
                    cov = A.dot(cov).dot(A.T) + Q
                tran = C.dot(tran)
                cov = C.dot(cov).dot(C.T) + R

                # Calculate mean + cov based on latent_var[t+lag]
                y_mean[-lag:] = latent_vars[:lag].dot(tran.T)
                y_cov[-lag:] = np.array([cov for _ in range(L+lag)])

                # Calculate mean + cov based on forward message
                default_mean = np.linalg.solve(forward_messages['precision'],
                        forward_message['mean_precision'])
                default_cov = np.linalg.inv(forward_message['precision'])
                for _ in range(-lag):
                    default_cov = A.dot(default_cov).dot(A.T) + Q
                default_cov = C.dot(default_cov).dot(C) + R
                y_mean[:-lag] = np.array([default_mean.dot(tran.T)])
                y_cov[:-lag] = np.array([default_cov for _ in range(-lag)])

                return y_mean, y_cov

    def y_sample(self, observations, parameters,
            distr='marginal', lag=None, num_samples=None,
            forward_message=None, backward_message=None,
            latent_var=None, tqdm=None):
        if forward_message is None:
            forward_message = self.default_forward_message
        if backward_message is None:
            backward_message = self.default_backward_message
        if latent_vars is not None:
            latent_vars = self.latent_var_sample(
                    observations=observations,
                    parameters=parameters,
                    distr=distr,
                    lag=lag,
                    num_samples=num_samples,
                    forward_message=forward_message,
                    backward_message=backward_message,
                    latent_vars=latent_vars,
                    tqdm=tqdm,
                    )

        if num_samples is not None:
            y = np.zeros((L, self.m, num_samples))
        else:
            y = np.zeros((L, self.m))

        C, R = parameters.C, parameters.R
        y = x.dot(C.T) + np.random.multivariate_normal(
            mean=np.zeros(self.m), cov=R, size=num_samples).T
        return y

    def simulate_distr(self, T, parameters, include_init=True,
            init_message=None, tqdm=None):
        if init_message is None:
            init_message = self.default_forward_message
        m, n = np.shape(parameters.C)
        A = parameters.A
        C = parameters.C
        Q = parameters.Q
        R = parameters.R

        # Outputs
        latent_vars_mean = np.zeros((T+1, n), dtype=float)
        latent_vars_cov = np.zeros((T+1, n, n), dtype=float)
        obs_mean = np.zeros((T+1, m), dtype=float)
        obs_cov = np.zeros((T+1, m, m), dtype=float)

        # Init
        latent_vars_mean[0] = np.linalg.solve(init_message['precision'],
                init_message['mean_precision'])
        latent_vars_cov[0] = np.linalg.inv(init_message['precision'])
        obs_mean[0] = np.dot(C, latent_vars_mean[0])
        obs_cov[0] = np.dot(C, latent_vars_cov[0]).dot(C.T) + R

        pbar = range(1,T+1)
        if tqdm is not None:
            pbar = tqdm(pbar, leave=False)
            pbar.set_description("simulating data")
        for t in pbar:
            latent_vars_mean[t] = np.dot(A, latent_vars_mean[t-1])
            latent_vars_cov[t] = np.dot(A, latent_vars_cov[t-1]).dot(A.T) + Q
            obs_mean[t] = np.dot(C, latent_vars_mean[t])
            obs_cov[t] = np.dot(C, latent_vars_cov[t]).dot(C.T) + R

        if include_init:
            return dict(
                obs_mean=obs_mean,
                obs_cov=obs_cov,
                latent_vars_mean=latent_vars_mean,
                latent_vars_cov=latent_vars_cov,
                )
        else:
            return dict(
                obs_mean=obs_mean[1:],
                obs_cov=obs_cov[1:],
                latent_vars_mean=latent_vars_mean[1:],
                latent_vars_cov=latent_vars_cov[1:],
                )

    def simulate(self, T, parameters, init_message=None, num_samples=None,
            include_init=True, tqdm=None):
        if init_message is None:
            init_message = self.default_forward_message
        m, n = np.shape(parameters.C)
        A = parameters.A
        C = parameters.C
        Q = parameters.Q
        R = parameters.R

        # Outputs
        if num_samples is not None:
            latent_vars = np.zeros((T+1, num_samples, n), dtype=float)
            obs_vars = np.zeros((T+1, num_samples, m), dtype=float)
        else:
            latent_vars = np.zeros((T+1, n), dtype=float)
            obs_vars = np.zeros((T+1, m), dtype=float)

        # Init
        latent_vars[0] = np.random.multivariate_normal(
                mean=np.linalg.solve(init_message['precision'],
                    init_message['mean_precision']),
                cov=np.linalg.inv(init_message['precision']),
                size=num_samples,
                )
        obs_vars[0] = np.dot(latent_vars[0], C.T) + \
            np.random.multivariate_normal(
                    mean=np.zeros(m), cov=R, size=num_samples)

        pbar = range(1,T+1)
        if tqdm is not None:
            pbar = tqdm(pbar, leave=False)
            pbar.set_description("simulating data")
        for t in pbar:
            latent_vars[t] = np.dot(latent_vars[t-1], A.T) + \
                np.random.multivariate_normal(
                        mean=np.zeros(n), cov=Q, size=num_samples)
            obs_vars[t] = np.dot(latent_vars[t], C.T) + \
                np.random.multivariate_normal(
                        mean=np.zeros(m), cov=R, size=num_samples)

        if num_samples is not None:
            obs_vars = np.swapaxes(obs_vars, 1, 2)
            latent_vars = np.swapaxes(latent_vars, 1, 2)

        if include_init:
            return dict(
                observations=obs_vars,
                latent_vars=latent_vars,
                )
        else:
            return dict(
                observations=obs_vars[1:],
                latent_vars=latent_vars[1:],
                )

    ## PF Functions
    def pf_loglikelihood_estimate(self, observations, parameters,
            subsequence_start=0, subsequence_end=None, weights=None,
            pf="poyiadjis_N", N=1000, kernel=None, forward_message=None,
            **kwargs):
        # Set kernel
        Kernel = self._get_kernel(kernel)

        # Prior Mean + Variance
        if forward_message is None:
            forward_message = self.default_forward_message
        prior_var = np.linalg.inv(forward_message['precision'])
        prior_mean = np.linalg.solve(prior_var,
                forward_message['mean_precision'])

        # Run buffered pf
        out = buffered_pf_wrapper(pf=pf,
                observations=observations,
                parameters=parameters,
                N=N,
                kernel=Kernel,
                additive_statistic_func=gaussian_sufficient_statistics,
                statistic_dim=self.n+2*self.n**2,
                t1=subsequence_start,
                tL=subsequence_end,
                weights=weights,
                prior_mean=prior_mean,
                prior_var=prior_var,
                **kwargs
                )
        loglikelihood = out['loglikelihood_estimate']
        return loglikelihood

    def pf_predictive_loglikelihood_estimate(self, observations, parameters,
            num_steps_ahead=1, subsequence_start=0, subsequence_end=None,
            weights=None,
            pf="filter", N=1000, kernel=None, forward_message=None,
            **kwargs):
        # Set kernel
        Kernel = self._get_kernel(kernel)

        # Prior Mean + Variance
        if forward_message is None:
            forward_message = self.default_forward_message
        prior_var = np.linalg.inv(forward_message['precision'])
        prior_mean = np.linalg.solve(prior_var,
                forward_message['mean_precision'])

        from functools import partial
        additive_statistic_func = partial(gaussian_predictive_loglikelihood,
                num_steps_ahead=num_steps_ahead,
                observations=observations,
                )

        # Run buffered pf
        out = buffered_pf_wrapper(pf=pf,
                observations=observations,
                parameters=parameters,
                N=N,
                kernel=Kernel,
                additive_statistic_func=additive_statistic_func,
                statistic_dim=num_steps_ahead+1,
                t1=subsequence_start,
                tL=subsequence_end,
                weights=weights,
                prior_mean=prior_mean,
                prior_var=prior_var,
                logsumexp=True,
                **kwargs
                )
        predictive_loglikelihood = out['statistics']
        predictive_loglikelihood[0] = out['loglikelihood_estimate']
        return predictive_loglikelihood

    def pf_gradient_estimate(self, observations, parameters,
            subsequence_start=0, subsequence_end=None, weights=None,
            pf="poyiadjis_N", N=1000, kernel=None, forward_message=None,
            **kwargs):
        # Set kernel
        Kernel = self._get_kernel(kernel)

        # Prior Mean + Variance
        if forward_message is None:
            forward_message = self.default_forward_message
        prior_var = np.linalg.inv(forward_message['precision'])
        prior_mean = np.linalg.solve(prior_var,
                forward_message['mean_precision'])


        # Run buffered pf
        complete_grad_dim = 2*self.n**2+self.n*self.m+self.m**2
        out = buffered_pf_wrapper(pf=pf,
                observations=observations,
                parameters=parameters,
                N=N,
                kernel=Kernel,
                additive_statistic_func=lgssm_complete_data_loglike_gradient,
                statistic_dim=complete_grad_dim,
                t1=subsequence_start,
                tL=subsequence_end,
                weights=weights,
                prior_mean=prior_mean,
                prior_var=prior_var,
                **kwargs
                )
        grad_estimate = average_statistic(out)
        if self.n*self.m > 1:
            grad = dict(
                LRinv_vec = grad_estimate[:(self.m+1)*self.m//2],
                LQinv_vec = grad_estimate[(self.m+1)*self.m//2:
                    (self.m+1)*self.m//2+(self.n+1)*self.n//2],
                C = np.reshape(grad_estimate[
                    (self.m+1)*self.m//2+(self.n+1)*self.n//2:
                    (self.m+1)*self.m//2+(self.n+1)*self.n//2+self.n*self.m],
                    (self.m, self.n),
                    ),
                A = np.reshape(grad_estimate[
                    (self.m+1)*self.m//2+(self.n+1)*self.n//2+self.n*self.m:],
                    (self.n, self.n),
                    ),
                )
        else:
            grad = dict(
                LRinv_vec = grad_estimate[0],
                LQinv_vec = grad_estimate[1],
                C = grad_estimate[2],
                A = grad_estimate[3],
                )
        return grad

    def pf_latent_var_distr(self, observations, parameters, lag=None,
            subsequence_start=0, subsequence_end=None, weights=None,
            pf="poyiadjis_N", N=1000, kernel=None, forward_message=None,
            **kwargs):
        if lag == 0 and pf != 'filter':
            raise ValueError("pf must be filter for lag = 0")
        elif lag is None and pf == 'filter':
            raise ValueError("pf must not be filter for smoothing")
        elif lag is not None and lag != 0:
            raise NotImplementedError("lag can only be None or 0")

        # Set kernel
        Kernel = self._get_kernel(kernel)

        # Prior Mean + Variance
        if forward_message is None:
            forward_message = self.default_forward_message
        prior_var = np.linalg.inv(forward_message['precision'])
        prior_mean = np.linalg.solve(prior_var,
                forward_message['mean_precision'])

        # Run buffered pf
        out = buffered_pf_wrapper(pf=pf,
                observations=observations,
                parameters=parameters,
                N=N,
                kernel=Kernel,
                additive_statistic_func=gaussian_sufficient_statistics,
                statistic_dim=self.n+2*self.n**2,
                t1=subsequence_start,
                tL=subsequence_end,
                weights=weights,
                prior_mean=prior_mean,
                prior_var=prior_var,
                elementwise_statistic=True,
                **kwargs
                )
        avg_statistic = average_statistic(out)
        if self.n > 1:
            avg_statistic = np.reshape(avg_statistic, (-1, self.n+2*self.n**2))
            x_mean = avg_statistic[:, 0:self.n]
            x_cov = np.reshape(
                avg_statistic[:, self.n:self.n+self.n**2],
                (-1, self.n, self.n),
                ) - np.einsum('ij,ik->ijk', x_mean, x_mean)
        else:
            avg_statistic = np.reshape(avg_statistic, (-1, 3))
            x_mean = avg_statistic[:, 0]
            x_cov = avg_statistic[:, 1] - x_mean**2

            x_mean = np.reshape(x_mean, (x_mean.shape[0], 1))
            x_cov = np.reshape(x_cov, (x_cov.shape[0], 1, 1))

        return x_mean, x_cov

    def _get_kernel(self, kernel):
        if kernel is None:
            if self.n*self.m == 1:
                kernel = 'optimal'
            else:
                kernel = 'highdim'
        if kernel == "prior":
            Kernel = LGSSMPriorKernel()
        elif kernel == "optimal":
            Kernel = LGSSMOptimalKernel()
        elif kernel == "highdim":
            Kernel = LGSSMHighDimOptimalKernel()
        else:
            raise ValueError("Unrecognized kernel = {0}".format(kernel))
        return Kernel

# Additive Statistics
def lgssm_complete_data_loglike_gradient(x_t, x_next, y_next, parameters,
        **kwargs):
    """ Gradient of Complete Data Log-Likelihood

    Gradient w/r.t. parameters of log Pr(y_{t+1}, x_{t+1} | x_t, parameters)

    Args:
        x_t (N by n ndarray): particles for x_t
        x_next (N by n ndarray): particles for x_{t+1}
        y_next (m ndarray): y_{t+1}
        parameters (Parameters): parameters
    Returns:
        grad_complete_data_loglike (N by p ndarray):
            gradient of complete data loglikelihood for particles
            [ grad_LRinv_vec, grad_LQinv_vec, grad_C, grad_A ]
    """
    N, n = np.shape(x_next)
    m = np.shape(y_next)[0]

    A = parameters.A
    LQinv = parameters.LQinv
    Qinv = parameters.Qinv
    C = parameters.C
    LRinv = parameters.LRinv
    Rinv = parameters.Rinv

    grad_complete_data_loglike = [None] * N
    if (n != 1) or (m != 1):
        LQinv_Tinv = np.linalg.inv(LQinv).T
        LRinv_Tinv = np.linalg.inv(LRinv).T
        for i in range(N):
            grad = {}
            diff = x_next[i] - np.dot(A, x_t[i])
            grad['A'] = np.outer(
                np.dot(Qinv, diff), x_t[i])
            grad['LQinv_vec'] = (
                    LQinv_Tinv + -1.0*np.dot(np.outer(diff, diff), LQinv)
                    )[np.tril_indices(n)]

            diff = y_next - np.dot(C, x_next[i])
            grad['C'] = np.outer(np.dot(Rinv, diff), x_next[i])
            grad['LRinv_vec'] = (
                    LRinv_Tinv + -1.0*np.dot(np.outer(diff, diff), LRinv)
                    )[np.tril_indices(m)]

            grad_complete_data_loglike[i] = np.concatenate([
                grad['LRinv_vec'].flatten(),
                grad['LQinv_vec'].flatten(),
                grad['C'].flatten(),
                grad['A'].flatten(),
                ])
        grad_complete_data_loglike = np.array(grad_complete_data_loglike)
    else:
        diff_x = x_next - A * x_t
        grad_A = Qinv * diff_x * x_t
        grad_LQinv = (LQinv**-1) - (diff_x**2) * LQinv
        diff_y = y_next - C * x_next
        grad_C = Rinv * diff_y * x_next
        grad_LRinv = (LRinv**-1) - (diff_y**2) * LRinv
        grad_complete_data_loglike = np.hstack([
            grad_LRinv, grad_LQinv, grad_C, grad_A])

    return grad_complete_data_loglike

def gaussian_predictive_loglikelihood(x_t, x_next, t, num_steps_ahead,
        parameters, observations,
        **kwargs):
    """ Predictive Log-Likelihood

    Calculate [Pr(y_{t+1+k} | x_{t+1} for k in [0,..., num_steps_ahead]]


    Args:
        x_t (N by n ndarray): particles for x_t
        x_next (N by n ndarray): particles for x_{t+1}
        num_steps_ahead
        parameters (Parameters): parameters
        observations (T by m ndarray): y
    Returns:
        predictive_loglikelihood (N by num_steps_ahead+1 ndarray)

    """
    N, n = np.shape(x_next)
    T, m = np.shape(observations)

    predictive_loglikelihood = np.zeros((N, num_steps_ahead+1))

    x_pred_mean = x_next + 0.0
    x_pred_cov = np.zeros((n, n))
    R, Q = parameters.R, parameters.Q
    for k in range(num_steps_ahead+1):
        if t+k >= T:
            break
        diff = (
            np.outer(np.ones(N), observations[t+k]) - \
            np.dot(x_pred_mean, parameters.C.T)
            )
        y_pred_cov = R + np.dot(parameters.C,
                np.dot(x_pred_cov, parameters.C.T))

        if m > 1:
            pred_loglike = (
                -0.5*np.sum(diff*np.linalg.solve(y_pred_cov, diff.T).T, axis=1)+\
                -0.5*m*np.log(2.0*np.pi) +\
                -0.5*np.linalg.slogdet(y_pred_cov)[1]
                    )
        else:
            pred_loglike = -0.5*diff**2/y_pred_cov + \
                    -0.5*np.log(2.0*np.pi) - 0.5*np.log(y_pred_cov)
            pred_loglike = pred_loglike[:,0]

        predictive_loglikelihood[:,k] = pred_loglike

        x_pred_mean = np.dot(x_pred_mean, parameters.A.T)
        x_pred_cov = Q + \
                np.dot(parameters.A,
                    np.dot(x_pred_cov, parameters.A.T))


    return predictive_loglikelihood

def gaussian_sufficient_statistics(x_t, x_next, y_next, **kwargs):
    """ Gaussian Sufficient Statistics

    h[0] = sum(x_{t+1})
    h[1] = sum(x_{t+1} x_{t+1}^T)
    h[2] = sum(x_t x_{t+1})

    Args:
        x_t (N by n ndarray): particles for x_t
        x_next (N by n ndarray): particles for x_{t+1}
        y_next (m ndarray): y_{t+1}
    Returns:
        h (N by p ndarray): sufficient statistic
    """
    N = np.shape(x_t)[0]
    if (len(np.shape(x_t)) > 1) and (np.shape(x_t)[1] > 1):
        # x is vector
        h = [x_next,
                np.einsum('ij,ik->ijk', x_next, x_next),
                np.einsum('ij,ik->ijk', x_t, x_next),
            ]
        h = np.hstack([np.reshape(h_, (N, -1)) for h_ in h])
    else:
        # n = 1, x is scalar
        h = np.hstack([x_next, x_next**2, x_t*x_next])
    return h


