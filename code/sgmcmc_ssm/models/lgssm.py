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
        )
from .._utils import (
        var_stationary_precision,
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
            tqdm=None):
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

            # Calculate Predict Parameters
            J = np.linalg.solve(AtQinvA + precision, AtQinv)
            pred_mean_precision = np.dot(J.T, mean_precision)
            pred_precision = Qinv - np.dot(AtQinv.T, J)

            # Calculate Observation Parameters
            y_mean = np.dot(C,
                    np.linalg.solve(pred_precision, pred_mean_precision))
            y_precision = Rinv - np.dot(CtRinv.T,
                    np.linalg.solve(CtRinvC + pred_precision, CtRinv))
            log_constant = log_constant + \
                    -0.5 * np.dot(y_cur-y_mean,
                            np.dot(y_precision, y_cur-y_mean)) + \
                     0.5 * np.linalg.slogdet(y_precision)[1] + \
                    -0.5 * self.m * np.log(2*np.pi)

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
            tqdm=None):
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

            # Helper Values
            xi = Qinv + precision + CtRinvC
            L = np.linalg.solve(xi, AtQinv.T)
            vi = mean_precision + np.dot(CtRinv, y_cur)

            # Calculate new parameters
            log_constant = log_constant + \
                    -0.5 * self.m * np.log(2.0*np.pi) + \
                    np.sum(np.log(np.diag(LRinv))) + \
                    np.sum(np.log(np.diag(LQinv))) + \
                    -0.5 * np.linalg.slogdet(xi)[1] + \
                    -0.5 * np.dot(y_cur, np.dot(Rinv, y_cur)) + \
                    0.5 * np.dot(vi, np.linalg.solve(xi, vi))

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
            forward_message=None, backward_message=None, **kwargs):
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
                **kwargs)

        loglikelihood = _marginal_loglikelihood_helper(
                forward_pass, backward_message,
                )
        return loglikelihood

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
            distribution='smoothed', tqdm=None):
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

        Returns
            x (ndarray): num_obs sampled latent values (R^n)
        """
        if forward_message is None:
            forward_message = self.default_forward_message
        if backward_message is None:
            backward_message = self.default_backward_message

        A, LQinv = parameters.A, parameters.LQinv
        AtQinv = np.dot(A.T, np.dot(LQinv, LQinv.T))
        AtQinvA = np.dot(AtQinv, A)

        L = np.shape(observations)[0]

        if distribution == 'smoothed':
            # Forward Pass
            forward_messages = self.forward_pass(
                    observations=observations,
                    parameters=parameters,
                    forward_message=forward_message,
                    include_init_message=False,
                    tqdm=tqdm
                    )

            # Backward Sampler
            x = np.zeros((L, self.n))
            x_cov = np.linalg.inv(forward_messages[-1]['precision'])
            x_mean = np.dot(x_cov, forward_messages[-1]['mean_precision'])
            x[-1, :] = np.random.multivariate_normal(mean=x_mean, cov=x_cov)

            pbar = reversed(range(L-1))
            if tqdm is not None:
                pbar = tqdm(pbar)
                pbar.set_description("backward smoothed sampling x")
            for t in pbar:
                x_next = x[t+1,:]
                x_cov = np.linalg.inv(forward_messages[t]['precision'] +
                        AtQinvA)
                x_mean = np.dot(x_cov, forward_messages[t]['mean_precision'] +
                        np.dot(AtQinv, x_next))
                x[t,:] = np.random.multivariate_normal(x_mean, x_cov)
            return x

        elif distribution == 'filtered':
            # Forward Pass (not a valid probability density)
            forward_messages = self.forward_pass(
                    observations=observations,
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
                x_cov = np.linalg.inv(forward_messages[t]['precision'])
                x_mean = np.dot(x_cov, forward_messages[t]['mean_precision'])
                x[t,:] = np.random.multivariate_normal(x_mean, x_cov)
            return x

        elif distribution == 'predictive':
            # Forward Sampler (not a valid probability density)
            forward_messages = self.forward_pass(
                    observations=observations,
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
                x_prev_cov = np.linalg.inv(forward_messages[t]['precision'])
                x_prev_mean = np.dot(x_prev_cov,
                        forward_messages[t]['mean_precision'])
                x_cov = np.dot(A, np.dot(x_prev_cov, A.T)) + Q
                x_mean = np.dot(A, x_prev_mean)
                x[t,:] = np.random.multivariate_normal(x_mean, x_cov)
            return x
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
            forward_message=None, backward_message=None,
            tqdm=None):
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
            pbar = tqdm(pbar)
            pbar.set_description("gradient loglike")
        for t, (forward_t, backward_t, y_t) in enumerate(p_bar):
            # Pr(x_t | y)
            c_mean_precision = \
                    forward_t['mean_precision'] + backward_t['mean_precision']
            c_precision = \
                    forward_t['precision'] + backward_t['precision']

            x_mean = np.linalg.solve(c_precision, c_mean_precision)
            xxt_mean = np.linalg.inv(c_precision) + np.outer(x_mean, x_mean)

            # Gradient of C
            C_grad += np.outer(np.dot(Rinv, y_t), x_mean) + \
                    -1.0 * np.dot(RinvC, xxt_mean)

            # Gradient of LRinv
            #raise NotImplementedError("SHOULD CHECK THE MATH FOR LRINV")
            Cxyt = np.outer(np.dot(C, x_mean), y_t)
            CxxtCt = np.dot(C, np.dot(xxt_mean, C.T))
            LRinv_grad += LRinv_diaginv + \
                -1.0*np.dot(np.outer(y_t, y_t) - Cxyt - Cxyt.T + CxxtCt, LRinv)

        # Transition Gradients
        for t, (forward_t, backward_t, y_t) in enumerate(
            zip(forward_messages[0:-1], backward_messages[1:], observations)):
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
            A_grad += np.dot(Qinv, xnxpt_mean - np.dot(A,xpxpt_mean))

            # Gradient of LQinv
            Axpxnt = np.dot(A, xnxpt_mean.T)
            AxpxptAt = np.dot(A, np.dot(xpxpt_mean, A.T))
            LQinv_grad += LQinv_diaginv + \
                -1.0*np.dot(xnxnt_mean - Axpxnt - Axpxnt.T + AxpxptAt, LQinv)

        grad = dict(A=A_grad, LQinv=LQinv_grad, C=C_grad, LRinv=LRinv_grad)
        return grad

def _marginal_loglikelihood_helper(forward_message, backward_message):
    # Calculate the marginal loglikelihood of forward + backward message
    f_mean_precision = forward_message['mean_precision']
    f_precision = forward_message['precision']
    c_mean_precision = f_mean_precision + backward_message['mean_precision']
    c_precision = f_precision + backward_message['precision']

    log_constant = forward_message['log_constant'] + \
            backward_message['log_constant'] + \
            +0.5 * np.linalg.slogdet(f_precision)[1] + \
            -0.5 * np.linalg.slogdet(c_precision)[1] + \
            -0.5 * np.dot(f_mean_precision,
                    np.linalg.solve(f_precision, f_mean_precision)
                ) + \
            0.5 * np.dot(c_mean_precision,
                np.linalg.solve(c_precision, c_mean_precision)
                )
    return log_constant

class LGSSMSampler(SGMCMCSampler):
    def __init__(self, n, m, name="LGSSMSampler", **kwargs):
        self.options = kwargs
        self.n = n
        self.m = m
        self.name = name
        self.message_helper=LGSSMHelper(
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

    def sample_x(self, parameters=None, observations=None, tqdm=None, **kwargs):
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



