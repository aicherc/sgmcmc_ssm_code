import numpy as np
import logging
logger = logging.getLogger(name=__name__)

from ..base_parameter import (
        BaseParameters, BasePrior, BasePreconditioner,
        )
from ..variable_mixins import (
        ASingleMixin, ASinglePrior, ASinglePreconditioner,
        CSingleMixin, CSinglePrior, CSinglePreconditioner,
        QSingleMixin, QSinglePrior, QSinglePreconditioner,
        RSingleMixin, RSinglePrior, RSinglePreconditioner,
        )
from ..sgmcmc_sampler import (
        SGMCMCSampler,
        SGMCMCHelper,
        SeqSGMCMCSampler,
        )
from ..particle_filters.kernels import (
        LGSSMPriorKernel,
        LGSSMOptimalKernel,
        LGSSMHighDimOptimalKernel,
        )
from ..particle_filters.buffered_smoother import (
        buffered_pf_wrapper,
        average_statistic,
        )
from sgmcmc_ssm.particle_filters.pf import (
        gaussian_sufficient_statistics,
        )

from .._utils import (
        var_stationary_precision,
        lower_tri_mat_inv,
        )

class LGSSMParameters(RSingleMixin, QSingleMixin, CSingleMixin, ASingleMixin,
        BaseParameters):
    """ LGSSM Parameters """
    def __str__(self):
        my_str = "LGSSMParameters:"
        my_str += "\nA:\n" + str(self.A)
        my_str += "\nC:\n" + str(self.C)
        my_str += "\nQ:\n" + str(self.Q)
        my_str += "\nR:\n" + str(self.R)
        return my_str

    def project_parameters(self, **kwargs):
        if kwargs.get('fix_C', True):
            k = min(self.n, self.m)
            C = self.C
            C[0:k, 0:k] = np.eye(k)
            self.C = C
        return super().project_parameters(**kwargs)

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

class LGSSMPrior(RSinglePrior, QSinglePrior, CSinglePrior, ASinglePrior,
        BasePrior):
    """ LGSSM Prior
    See individual Prior Mixins for details
    """
    @staticmethod
    def _parameters(**kwargs):
        return LGSSMParameters(**kwargs)

class LGSSMPreconditioner(RSinglePreconditioner, QSinglePreconditioner,
        CSinglePreconditioner, ASinglePreconditioner, BasePreconditioner):
    """ Preconditioner for LGSSM
    See individual Preconditioner Mixin for details
    """
    pass

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

    def _forward_messages(self, observations, parameters, forward_message,
            weights=None, tqdm=None):
        # Return list of forward messages Pr(x_{t} | y_{<=t})
        # y is num_obs x m matrix
        num_obs = np.shape(observations)[0]
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
            pbar = tqdm(pbar)
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
            forward_messages[t+1] = {
                'mean_precision': mean_precision,
                'precision': precision,
                'log_constant': log_constant,
            }
        return forward_messages

    def _backward_messages(self, observations, parameters, backward_message,
            weights=None, tqdm=None):
        # Return list of backward messages Pr(y_{>t} | x_t)
        # y is num_obs x n matrix
        num_obs = np.shape(observations)[0]
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
            pbar = tqdm(pbar, total=num_obs)
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

            backward_messages[t] = {
                'mean_precision': mean_precision,
                'precision': precision,
                'log_constant': log_constant,
            }

        return backward_messages

    def marginal_loglikelihood(self, observations, parameters,
            forward_message=None, backward_message=None, weights=None,
            **kwargs):
        # Run forward pass + combine with backward pass
        # y is num_obs x n matrix
        if forward_message is None:
            forward_message = self.default_forward_message
        if backward_message is None:
            backward_message = self.default_backward_message

        # forward_pass is Pr(x_{T-1} | y_{<=T-1})
        forward_pass = self._forward_message(
                observations=observations,
                parameters=parameters,
                forward_message=forward_message,
                weights=weights,
                **kwargs)

        loglikelihood = _marginal_loglikelihood_helper(
                forward_pass,
                backward_message,
                weight=1.0 if weights is None else weights[-1],
                )
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

    def predictive_loglikelihood(self, observations, parameters, lag=10,
            forward_message=None, backward_message=None, **kwargs):
        if forward_message is None:
            forward_message = self.default_forward_message
        if backward_message is None:
            backward_message = self.default_backward_message

        # Calculate Filtered
        if lag == 0:
            forward_messages = self.forward_pass(observations,
                    parameters, forward_message, **kwargs)
        else:
            forward_messages = self.forward_pass(observations[0:-lag],
                    parameters, forward_message, **kwargs)
        loglike = 0.0
        A = parameters.A
        Q = parameters.Q
        C = parameters.C
        R = parameters.R
        for t in range(lag, np.shape(observations)[0]):
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

    def latent_var_sample(self, observations, parameters,
            forward_message=None, backward_message=None,
            distribution='smoothed', num_samples=None,
            tqdm=None, include_init=False):
        """ Sample latent vars from observations

        Backward pass + forward sampler for LGSSM

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
                smoothed: sample X from Pr(X | Y, theta)
                filtered: sample X_t from Pr(X_t | Y_<=t, theta) iid for all t
                predictive: sample X_t from Pr(X_t | Y_<t, theta) iid for all t
            num_samples (int, optional) number of samples
            include_init (bool, optional): whether to sample x_{-1} | y

        Returns
            x (ndarray): (num_obs by n) latent values (if num_samples is None)
            or
            xs (ndarray): (num_obs by n by num_samples) latent variables

        """
        if forward_message is None:
            forward_message = self.default_forward_message
        if backward_message is None:
            backward_message = self.default_backward_message


        A, LQinv = parameters.A, parameters.LQinv
        AtQinv = np.dot(A.T, np.dot(LQinv, LQinv.T))
        AtQinvA = np.dot(AtQinv, A)

        # Forward Pass
        forward_messages = self.forward_pass(
                observations=observations,
                parameters=parameters,
                forward_message=forward_message,
                include_init_message=include_init,
                tqdm=tqdm
                )
        L = len(forward_messages)

        if num_samples is None:
            num_samples = 1
            only_return_one = True
        else:
            only_return_one = False
        x = np.zeros((L, self.n, num_samples))


        if distribution == 'smoothed':
            # Backward Sampler
            x_cov = np.linalg.inv(forward_messages[-1]['precision'])
            x_mean = np.dot(x_cov, forward_messages[-1]['mean_precision'])
            x[-1,:,:] = np.random.multivariate_normal(mean=x_mean, cov=x_cov,
                    size=num_samples).T

            pbar = reversed(range(L-1))
            if tqdm is not None:
                pbar = tqdm(pbar)
                pbar.set_description("backward smoothed sampling x")
            for t in pbar:
                x_next = x[t+1,:,:]
                x_cov = np.linalg.inv(forward_messages[t]['precision'] +
                        AtQinvA)
                x_mean = np.dot(x_cov,
                        np.outer(forward_messages[t]['mean_precision'],
                            num_samples) +
                        np.dot(AtQinv, x_next))
                x[t,:,:] = x_mean + np.random.multivariate_normal(
                        mean=np.zeros(self.n), cov=x_cov, size=num_samples,
                        ).T
            if only_return_one:
                return x[:,:,0]
            else:
                return x

        elif distribution == 'filtered':
            pbar = range(L)
            if tqdm is not None:
                pbar = tqdm(pbar)
                pbar.set_description("filtered sampling x")
            for t in pbar:
                x_cov = np.linalg.inv(forward_messages[t]['precision'])
                x_mean = np.dot(x_cov, forward_messages[t]['mean_precision'])
                x[t,:,:] = np.random.multivariate_normal(x_mean, x_cov,
                        size=num_samples).T
            if only_return_one:
                return x[:,:,0]
            else:
                return x

        elif distribution == 'predictive':
            if include_init:
                raise NotImplementedError()
            # Backward Sampler
            Q = parameters.Q
            pbar = range(L)
            if tqdm is not None:
                pbar = tqdm(pbar)
                pbar.set_description("predictive sampling x")
            for t in pbar:
                x_prev_cov = np.linalg.inv(forward_messages[t]['precision'])
                x_prev_mean = np.dot(x_prev_cov,
                        forward_messages[t]['mean_precision'])
                x_cov = np.dot(A, np.dot(x_prev_cov, A.T)) + Q
                x_mean = np.dot(A, x_prev_mean)
                x[t,:,:] = np.random.multivariate_normal(x_mean, x_cov,
                        size=num_samples).T
            if only_return_one:
                return x[:,:,0]
            else:
                return x
        else:
            raise ValueError("Invalid `distribution'; {0}".format(distribution))
        return

    def latent_var_marginal(self, observations, parameters,
            forward_message=None, backward_message=None,
            distribution='smoothed', tqdm=None,
            include_init=False):
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
            include_init (bool, optional): whether to include x_{-1} | y

        Returns
            mean (ndarray): num_obs by n, marginal mean
            cov (ndarray): num_obs by n by n, marginal covariance
        """
        if forward_message is None:
            forward_message = self.default_forward_message
        if backward_message is None:
            backward_message = self.default_backward_message

        L = np.shape(observations)[0]
        if include_init:
            L = L + 1
        mean = np.zeros((L, self.n))
        cov = np.zeros((L, self.n, self.n))
        if distribution == 'smoothed':
            forward_messages = self.forward_pass(
                    observations=observations,
                    parameters=parameters,
                    forward_message=forward_message,
                    include_init_message=include_init,
                    tqdm=tqdm
                    )
            backward_messages = self.backward_pass(
                    observations=observations,
                    parameters=parameters,
                    backward_message=backward_message,
                    include_init_message=include_init,
                    tqdm=tqdm
                    )
            for t in range(L):
                mean_precision = \
                        forward_messages[t]['mean_precision'] + \
                        backward_messages[t]['mean_precision']
                precision = \
                        forward_messages[t]['precision'] + \
                        backward_messages[t]['precision']

                mean[t] = np.linalg.solve(precision, mean_precision)
                cov[t] = np.linalg.inv(precision)
            return mean, cov

        elif distribution == 'filtered':
            forward_messages = self.forward_pass(
                    observations=observations,
                    parameters=parameters,
                    forward_message=forward_message,
                    include_init_message=include_init,
                    tqdm=tqdm
                    )
            for t in range(L):
                mean_precision = forward_messages[t]['mean_precision']
                precision = forward_messages[t]['precision']
                mean[t] = np.linalg.solve(precision, mean_precision)
                cov[t] = np.linalg.inv(precision)
            return mean, cov
        elif distribution == 'predictive':
            raise NotImplementedError()

        else:
            raise ValueError("Invalid `distribution'; {0}".format(distribution))
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
        sufficient_stat = dict(
                Sx_count=transition_count,
                Sx_prevprev=Sx_prevprev,
                Sx_curprev=Sx_curprev,
                Sx_curcur=Sx_curcur,
                S_count=emission_count,
                S_prevprev=S_prevprev,
                S_curprev=S_curprev,
                S_curcur=S_curcur,
                )
        return sufficient_stat

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
            p_bar = tqdm(p_bar, total=np.shape(observations)[0])
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

        grad = dict(A=A_grad, LQinv=LQinv_grad, C=C_grad, LRinv=LRinv_grad)
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

        return grad

    def gradient_loglikelihood(self, kind='marginal', **kwargs):
        if kind == 'marginal':
            return self.gradient_marginal_loglikelihood(**kwargs)
        elif kind == 'complete':
            return self.gradient_complete_data_loglikelihood(**kwargs)
        else:
            raise ValueError("Unrecognized `kind' {0}".format(kind))

    def pf_score_estimate(self, observations, parameters,
            subsequence_start=0, subsequence_end=None, weights=None,
            pf="poyiadjis_N", N=100, kernel=None,
            **kwargs):
        """ Particle Filter Score Estimate

        Args:
            observations (ndarray): num_obs bufferd observations
            parameters (Parameters): parameters
            subsequence_start (int): relative start of subsequence
                (0:subsequence_start are left buffer)
            subsequence_end (int): relative end of subsequence
                (subsequence_end: is right buffer)
            weights (ndarray): weights for [subsequence_start, subsequence_end)
            pf (string): particle filter name
                "nemeth" - use Nemeth et al. O(N)
                "poyiadjis_N" - use Poyiadjis et al. O(N)
                "poyiadjis_N2" - use Poyiadjis et al. O(N^2)
                "paris" - use PaRIS Olsson + Westborn O(N log N)
            N (int): number of particles used by particle filter
            kernel (string): kernel to use
                "prior" - bootstrap filter P(X_t | X_{t-1})
                "optimal" - bootstrap filter P(X_t | X_{t-1}, Y_t)
            **kwargs - additional keyword args for individual filters

        Return:
            grad (dict): grad of variables in parameters

        """
        # Set kernel
        Kernel = self._get_kernel(kernel)

        # Prior Mean + Variance
        prior_var = self.default_forward_message['precision'][0,0]**-1
        prior_mean = \
                self.default_forward_message['mean_precision'][0] * prior_var

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
        score_estimate = average_statistic(out)
        if self.n*self.m > 1:
            grad = dict(
                LRinv = np.reshape(
                    score_estimate[:self.m**2], (self.m, self.m),
                    ),
                LQinv = np.reshape(
                    score_estimate[self.m**2:self.m**2+self.n**2],
                    (self.n, self.n),
                    ),
                C = np.reshape(score_estimate[
                    self.m**2+self.n**2:self.m**2+self.n**2+self.n*self.m],
                    (self.m, self.n),
                    ),
                A = np.reshape(
                    score_estimate[self.m**2+self.n**2+self.n*self.m:],
                    (self.n, self.n),
                    ),
                )
        else:
            grad = dict(
                LRinv = score_estimate[0],
                LQinv = score_estimate[1],
                C = score_estimate[2],
                A = score_estimate[3],
                )

        return grad

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

    def pf_loglikelihood_estimate(self, observations, parameters,
            subsequence_start=0, subsequence_end=None, weights=None,
            pf="poyiadjis_N", N=100, kernel=None,
            **kwargs):
        """ Particle Filter Marginal Log-Likelihood Estimate

        Args:
            observations (ndarray): num_obs bufferd observations
            parameters (Parameters): parameters
            subsequence_start (int): relative start of subsequence
                (0:subsequence_start are left buffer)
            subsequence_end (int): relative end of subsequence
                (subsequence_end: is right buffer)
            weights (ndarray): weights for [subsequence_start, subsequence_end)
            pf (string): particle filter name
                "nemeth" - use Nemeth et al. O(N)
                "poyiadjis_N" - use Poyiadjis et al. O(N)
                "poyiadjis_N2" - use Poyiadjis et al. O(N^2)
                "paris" - use PaRIS Olsson + Westborn O(N log N)
            N (int): number of particles used by particle filter
            kernel (string): kernel to use
                "prior" - bootstrap filter P(X_t | X_{t-1})
                "optimal" - bootstrap filter P(X_t | X_{t-1}, Y_t)
            **kwargs - additional keyword args for individual filters

        Return:
            loglikelihood (double): marignal log likelihood estimate

        """
        # Set kernel
        Kernel = self._get_kernel(kernel)

        # Prior Mean + Variance
        prior_var = self.default_forward_message['precision'][0,0]**-1
        prior_mean = \
                self.default_forward_message['mean_precision'][0] * prior_var

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
            num_steps_ahead=5,
            subsequence_start=0, subsequence_end=None,
            pf="pf_filter", N=1000, kernel=None,
            **kwargs):
        """ Particle Filter Predictive Log-Likelihood Estimate

        Returns predictive log-likleihood for k = [0,1, ...,num_steps_ahead]

        Args:
            observations (ndarray): num_obs bufferd observations
            parameters (Parameters): parameters
            num_steps_ahead (int): number of steps
            subsequence_start (int): relative start of subsequence
                (0:subsequence_start are left buffer)
            subsequence_end (int): relative end of subsequence
                (subsequence_end: is right buffer)
            N (int): number of particles used by particle filter
            kernel (string): kernel to use
            **kwargs - additional keyword args for individual filters

        Return:
            predictive_loglikelihood (num_steps_ahead + 1 ndarray)

        """
        if pf != "pf_filter":
            raise ValueError("Only can use pf = 'pf_filter' since we are filtering")
        # Set kernel
        Kernel = self._get_kernel(kernel)

        # Prior Mean + Variance
        prior_var = self.default_forward_message['precision'][0,0]**-1
        prior_mean = \
                self.default_forward_message['mean_precision'][0] * prior_var

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
                prior_mean=prior_mean,
                prior_var=prior_var,
                **kwargs
                )
        predictive_loglikelihood = out['statistics']
        predictive_loglikelihood[0] = out['loglikelihood_estimate']
        return predictive_loglikelihood

    def pf_latent_var_marginal(self, observations, parameters,
            subsequence_start=0, subsequence_end=None, weights=None,
            pf="poyiadjis_N", N=100, kernel=None,
            **kwargs):
        # Set kernel
        Kernel = self._get_kernel(kernel)

        # Prior Mean + Variance
        prior_var = self.default_forward_message['precision'][0,0]**-1
        prior_mean = \
                self.default_forward_message['mean_precision'][0] * prior_var

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

def _marginal_loglikelihood_helper(forward_message, backward_message,
        weight=1.0):
    # Calculate the marginal loglikelihood of forward + backward message
    f_mean_precision = forward_message['mean_precision']
    f_precision = forward_message['precision']
    c_mean_precision = f_mean_precision + backward_message['mean_precision']
    c_precision = f_precision + backward_message['precision']

    log_constant = forward_message['log_constant'] + \
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
    return log_constant

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
            [ grad_LRinv, grad_LQinv, grad_C, grad_A ]
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
            grad['LQinv'] = LQinv_Tinv + -1.0*np.dot(np.outer(diff, diff), LQinv)

            diff = y_next - np.dot(C, x_next[i])
            grad['C'] = np.outer(np.dot(Rinv, diff), x_next[i])
            grad['LRinv'] = LRinv_Tinv + -1.0*np.dot(np.outer(diff, diff), LRinv)

            grad_complete_data_loglike[i] = np.concatenate([
                grad['LRinv'].flatten(),
                grad['LQinv'].flatten(),
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


class LGSSMSampler(SGMCMCSampler):
    def __init__(self, n, m, name="LGSSMSampler", **kwargs):
        self.options = kwargs
        self.n = n
        self.m = m
        self.name = name

        Helper = kwargs.get('Helper', LGSSMHelper)
        self.message_helper=Helper(
                n=self.n,
                m=self.m,
                )
        return

    def setup(self, observations, prior, parameters=None, forward_message=None):
        """ Initialize the sampler

        Args:
            observations (ndarray): T by m ndarray of time series values
            prior (LGSSMPrior): prior
            forward_message (ndarray): prior probability for latent state
            parameters (LGSSMParameters): initial parameters
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
            if not isinstance(parameters, LGSSMParameters):
                raise ValueError("parameters is not a LGSSMParameter")
            self.parameters = parameters


        if forward_message is None:
             forward_message = {
                    'log_constant': 0.0,
                    'mean_precision': np.zeros(self.n),
                    'precision': np.eye(self.n)/10,
                    }
        self.forward_message = forward_message
        self.backward_message = {
                'log_constant': 0.0,
                'mean_precision': np.zeros(self.n),
                'precision': np.zeros((self.n, self.n)),
                }

        return

    def sample_x(self, parameters=None, observations=None, tqdm=None,
            num_samples=None, **kwargs):
        """ Sample X """
        if parameters is None:
            parameters = self.parameters
        if observations is None:
            observations = self.observations
        x = self.message_helper.latent_var_sample(
                observations=observations,
                parameters=parameters,
                forward_message=self.forward_message,
                backward_message=self.backward_message,
                tqdm=tqdm,
                num_samples=None,
                )
        return x

    def sample_gibbs(self, tqdm=None):
        """ One Step of Blocked Gibbs Sampler

        Returns:
            parameters (LGSSMParameters): sampled parameters after one step
        """
        x = self.sample_x(tqdm=tqdm)
        new_parameters = self.message_helper.parameters_gibbs_sample(
                observations=self.observations,
                latent_vars=x,
                prior=self.prior,
                )
        self.parameters = new_parameters
        return self.parameters

class SeqLGSSMSampler(SeqSGMCMCSampler, LGSSMSampler):
    pass

