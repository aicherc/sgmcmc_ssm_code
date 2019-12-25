import numpy as np
import logging
logger = logging.getLogger(name=__name__)

from ..hmm_helper import HMMHelper
from .parameters import stack_y
from ..._utils import random_categorical

class ARPHMMHelper(HMMHelper):
    """ ARPHMM Helper

        forward_message (dict) with keys
            prob_vector (ndarray) dimension num_states
            log_constant (double) log scaling const

        backward_message (dict) with keys
            likelihood_vector (ndarray) dimension num_states
            log_constant (double) log scaling const
            y_next (ndarray) y_{t+1}
    """
    def __init__(self, num_states, m, d,
            forward_message=None,
            backward_message=None,
            **kwargs):
        self.num_states = num_states
        self.m = m
        self.d = d
        super().__init__(
                forward_message=forward_message,
                backward_message=backward_message,
                **kwargs)
        return

    def y_sample(self, observations, parameters,
            distr='joint', lag=None, num_samples=None,
            forward_message=None, backward_message=None,
            latent_vars=None, tqdm=None):
        if forward_message is None:
            forward_message = self.default_forward_message
        if backward_message is None:
            backward_message = self.default_backward_message
        if distr == 'joint' and lag is not None:
            raise ValueError("Must set distr to 'marginal' for lag != None")

        if (lag is None) or (lag > parameters.p):
            # Handle Smoothing Case
            if latent_vars is None:
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
            D, R = parameters.D, parameters.R
            L = latent_vars.shape[0]
            if num_samples is None:
                y = np.zeros((L, self.m))
                z = latent_vars
                for k in range(self.num_states):
                    num_k = np.sum(z==k)
                    if num_k == 0:
                        continue
                    mu_k = observations[z==k,1:].reshape(num_k, -1).dot(D[k])
                    y[z==k] = mu[k] + np.random.multivariate_normal(
                    mean=np.zeros(self.m), cov=R[k], size=num_k)
                return y

            else:
                y = np.zeros((L, num_samples, self.m))
                pbar = range(num_samples)
                if tqdm is not None:
                    pbar = tqdm(pbar)
                    tqdm.set_description('sample y')
                for s in pbar:
                    z = latent_vars[:,s]
                    for k in range(self.num_states):
                        num_k = np.sum(z==k)
                        if num_k == 0:
                            continue
                        mu_k = observations[z==k,1:].reshape(num_k, -1).dot(D[k])
                        y[z==k,s] = mu_k + np.random.multivariate_normal(
                            mean=np.zeros(self.m), cov=R[k], size=num_k)
                y = np.swapaxes(y, 1, 2)
                return y
        else:
            # Predictive Lag
            raise NotImplementedError()

    def simulate(self, T, parameters, init_message=None, num_samples=None,
            include_init=True, tqdm=None):
        if init_message is None:
            init_message = self.default_forward_message
        num_states, m, p = parameters.num_states, parameters.m, parameters.p
        Pi, D, R = parameters.pi, parameters.D, parameters.R

        # Outputs
        if num_samples is not None:
            latent_vars = np.zeros((T+1, num_samples), dtype=int)
            obs_vars = np.zeros((T+p+1, num_samples, m), dtype=float)
            y_prev = init_message.get('y_prev', np.zeros((p,m)))
            y_prev = np.repeat(y_prev[np.newaxis,:,:], num_samples, 0)
            obs_vars[:p] = y_prev.transpose(1,0,2)
        else:
            latent_vars = np.zeros((T+1), dtype=int)
            obs_vars = np.zeros((T+p+1, m), dtype=float)
            y_prev = init_message.get('y_prev', np.zeros((p,m)))
            obs_vars[:p] = y_prev

        latent_vars[0] = random_categorical(init_message['prob_vector'],
                size=num_samples)
        pbar = range(T+1)
        if tqdm is not None:
            pbar = tqdm(pbar)
            pbar.set_description("simulating data")
        for t in pbar:
            latent_prev = latent_vars[t]
            if num_samples is None:
                k = latent_prev
                mu_k = y_prev.flatten().dot(D[k].T)
                obs_vars[t+p] = \
                        mu_k + np.random.multivariate_normal(
                        mean=np.zeros(m),
                        cov=R[k],
                        )
                if t < T+p+1:
                    latent_vars[t+1] = random_categorical(Pi[k])
                y_prev = np.vstack([obs_vars[t], y_prev[:-1,:]])

            else:
                for k in range(num_states):
                    num_k = np.sum(latent_prev == k)
                    if num_k == 0:
                        continue
                    mu_k = y_prev[latent_prev == k].reshape(num_k, -1).dot(D[k].T)
                    obs_vars[t+p,latent_prev == k] = \
                            mu_k + np.random.multivariate_normal(
                            mean=np.zeros(m),
                            cov=R[k],
                            size=num_k
                            )
                    if t+1 < latent_vars.shape[0]:
                        latent_vars[t+1, latent_prev == k] = \
                                random_categorical(Pi[k], size=num_k)
                y_prev = np.hstack([
                    obs_vars[t].reshape(num_samples,1,m),
                    y_prev[:,:-1,:]
                    ])

        if num_samples is not None:
            obs_vars = np.array([stack_y(obs_vars[:,s,:],p)
                for s in range(num_samples)])
            obs_vars = np.transpose(obs_vars, (1,2,3,0))
        else:
            obs_vars = stack_y(obs_vars, p)

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

    def calc_gibbs_sufficient_statistic(self, observations, latent_vars,
            **kwargs):
        """ Gibbs Sample Sufficient Statistics
        Args:
            observations (ndarray): num_obs observations
            latent_vars (ndarray): latent vars

        Returns:
            sufficient_stat (dict) containing:
                alpha (ndarray) num_states by num_states, pairwise z counts
                S_count (ndarray) num_states, z counts
                S_prevprev (ndarray) num_states mp by mp, z counts
                S_curprev (ndarray) num_states m by mp, y sum
                S_curcur (ndarray) num_states m by m, yy.T sum
        """
        z = latent_vars

        # Sufficient Statistics for Pi
        z_pair_count = np.zeros((self.num_states, self.num_states))

        for t in range(1, np.size(z)):
            z_pair_count[z[t-1], z[t]] += 1.0

        # Sufficient Statistics for mu and R
        S_count = np.zeros(self.num_states)
        y_sum = np.zeros((self.num_states, self.m))
        yy_prevprev = np.zeros((self.num_states, self.d, self.d))
        yy_curprev = np.zeros((self.num_states, self.m, self.d))
        yy_curcur = np.zeros((self.num_states, self.m, self.m))

        for k in range(self.num_states):
            S_count[k] = np.sum(z == k)
            if S_count[k] == 0:
                # No Sufficient Statistics for No Observations
                continue
            yk = observations[z==k, 0, :]
            yprevk = np.reshape(observations[z==k, 1:, :], (yk.shape[0], -1))

            # Sufficient Statistics for group k
            y_sum[k] = np.sum(yk, axis=0)
            yy_prevprev[k] = np.dot(yprevk.T, yprevk)
            yy_curprev[k] = np.dot(yk.T, yprevk)
            yy_curcur[k] = np.dot(yk.T, yk)

        # Return sufficient Statistics
        sufficient_stat = {}
        sufficient_stat['pi'] = dict(alpha = z_pair_count)
        sufficient_stat['D'] = dict(
                S_prevprev = yy_prevprev,
                S_curprev = yy_curprev,
                )
        sufficient_stat['R'] = dict(
                S_count=S_count,
                S_prevprev = yy_prevprev,
                S_curprev = yy_curprev,
                S_curcur=yy_curcur,
                )
        return sufficient_stat

    def _emission_loglikelihoods(self, y_cur, parameters):
        # y_cur should be p+1 by m,
        loglikelihoods = np.zeros(self.num_states, dtype=float)
        y_prev = y_cur[1:].flatten()
        for k, (D_k, LRinv_k) in enumerate(
                zip(parameters.D, parameters.LRinv)):
            delta = y_cur[0] - np.dot(D_k, y_prev)
            LRinvTdelta = np.dot(delta, LRinv_k)
            loglikelihoods[k] = \
                -0.5 * np.dot(LRinvTdelta, LRinvTdelta) + \
                -0.5 * self.m * np.log(2*np.pi) + \
                np.sum(np.log(np.abs(np.diag(LRinv_k))))
        return loglikelihoods

    def _emission_loglikelihood(self, y_cur, z_cur, parameters):
        # y_cur should be p+1 by m,
        loglikelihoods = np.zeros(self.num_states, dtype=float)
        y_prev = y_cur[1:].flatten()
        D_k, LRinv_k = parameters.D[z_cur], parameters.LRinv[z_cur]
        delta = y_cur[0] - np.dot(D_k, y_prev)
        LRinvTdelta = np.dot(delta, LRinv_k)
        loglikelihood = \
            -0.5 * np.dot(LRinvTdelta, LRinvTdelta) + \
            -0.5 * self.m * np.log(2*np.pi) + \
            np.sum(np.log(np.abs(np.diag(LRinv_k))))
        return loglikelihood

    def gradient_marginal_loglikelihood(self, observations, parameters,
            forward_message=None, backward_message=None, weights=None,
            tqdm=None):
        # Forward Pass
        forward_messages = self.forward_pass(observations, parameters,
                forward_message, include_init_message=True)
        # Backward Pass
        backward_messages = self.backward_pass(observations, parameters,
                backward_message, include_init_message=True)

        # Gradients
        grad = {var: np.zeros_like(value)
                for var, value in parameters.as_dict().items()}

        Pi, expanded_pi = parameters.pi, parameters.expanded_pi
        D = parameters.D
        LRinv, Rinv, R = parameters.LRinv, parameters.Rinv, parameters.R

        pbar = enumerate(zip(forward_messages[:-1], backward_messages[1:]))
        if tqdm is not None:
            pbar = tqdm(pbar)
            pbar.set_description("gradient loglike")
        for t, (forward_t, backward_t) in pbar:
            # r_t is Pr(z_{t-1} | y_{< t})
            # s_t is Pr(z_t | y_{< t})
            # q_t is Pr(y_{> t} | z_t)
            r_t = forward_t['prob_vector']
            s_t = np.dot(r_t, Pi)
            q_t = backward_t['likelihood_vector']

            weight_t = 1.0 if weights is None else weights[t]

            # Calculate P_t = Pr(y_t | z_t)
            y_cur = observations[t]
            P_t, _ = self._likelihoods(
                    y_cur=y_cur,
                    parameters=parameters
                )

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

            # grad for mu and LRinv
            y_prev = y_cur[1:].flatten()
            for k, D_k, LRinv_k, Rinv_k, R_k in zip(
                    range(self.num_states), D, LRinv, Rinv, R):
                diff_k = y_cur[0] - np.dot(D_k, y_prev)
                grad['D'][k] += weight_t * (
                        np.outer(Rinv_k.dot(diff_k), y_prev) * marg_post[k])
                grad_LRinv_k = weight_t * (
                        (R_k - np.outer(diff_k, diff_k)).dot(LRinv_k)
                        ) * marg_post[k]
                grad['LRinv_vec'][k] += grad_LRinv_k[np.tril_indices(self.m)]

        return grad


