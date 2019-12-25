import numpy as np
import logging
logger = logging.getLogger(name=__name__)

from ..hmm_helper import HMMHelper
from ..._utils import random_categorical

class GaussHMMHelper(HMMHelper):
    """ GaussHMM Helper

        forward_message (dict) with keys
            prob_vector (ndarray) dimension num_states
            log_constant (double) log scaling const

        backward_message (dict) with keys
            likelihood_vector (ndarray) dimension num_states
            log_constant (double) log scaling const
            y_next (ndarray) y_{t+1}
    """
    def __init__(self, num_states, m,
            forward_message=None,
            backward_message=None,
            **kwargs):
        self.num_states = num_states
        self.m = m
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

        mu, R = parameters.mu, parameters.R
        L = latent_vars.shape[0]

        if num_samples is not None:
            y = np.zeros((L, num_samples, self.m))
            pbar = range(num_samples)
            if tqdm is not None:
                pbar = tqdm(pbar)
                tqdm.set_description('sample y')
            for s in pbar:
                z = latent_vars[:,s]
                for k in range(self.num_states):
                    y[z==k,s] = mu[k] + np.random.multivariate_normal(
                        mean=np.zeros(self.m), cov=R[k], size=np.sum(z==k))
            y = np.swapaxes(y, 1, 2)
            return y
        else:
            y = np.zeros((L, self.m))
            z = latent_vars
            for k in range(self.num_states):
                y[z==k] = mu[k] + np.random.multivariate_normal(
                mean=np.zeros(self.m), cov=R[k], size=np.sum(z==k))
        return y

    def calc_gibbs_sufficient_statistic(self, observations, latent_vars,
            **kwargs):
        """ Gibbs Sample Sufficient Statistics
        Args:
            observations (ndarray): num_obs observations
            latent_vars (ndarray): latent vars

        Returns:
            sufficient_stat (dict) containing:
                alpha_Pi (ndarray) num_states by num_states, pairwise z counts
                S_count (ndarray) num_states, z counts
                S_prevprev (ndarray) num_states, z counts
                S_curprev (ndarray) num_states m, y sum
                S_curcur (ndarray) num_states m by m, yy.T sum
        """
        y, z = observations, latent_vars

        # Sufficient Statistics for Pi
        z_pair_count = np.zeros((self.num_states, self.num_states))

        for t in range(1, np.size(z)):
            z_pair_count[z[t-1], z[t]] += 1.0

        # Sufficient Statistics for mu and R
        S_count = np.zeros(self.num_states)
        y_sum = np.zeros((self.num_states, self.m))
        yy_curcur = np.zeros((self.num_states, self.m, self.m))

        for k in range(self.num_states):
            S_count[k] = np.sum(z == k)
            if S_count[k] == 0:
                # No Sufficient Statistics for No Observations
                continue
            yk = y[z==k, :]
            # Sufficient Statistics for group k
            y_sum[k] = np.sum(yk, axis=0)
            yy_curcur[k] = np.dot(yk.T, yk)

        # Return sufficient Statistics
        sufficient_stat = {}
        sufficient_stat['pi'] = dict(alpha = z_pair_count)
        sufficient_stat['mu'] = dict(S_prevprev = S_count, S_curprev = y_sum)
        sufficient_stat['R'] = dict(
                S_count=S_count,
                S_prevprev = S_count,
                S_curprev = y_sum,
                S_curcur=yy_curcur,
                )
        return sufficient_stat

    def _emission_loglikelihoods(self, y_cur, parameters):
        # y_cur should be m,
        # mu is num_states by m
        # LRinv is num_states by m by m
        loglikelihoods = np.zeros(self.num_states, dtype=float)
        for k, (mu_k, LRinv_k) in enumerate(
                zip(parameters.mu, parameters.LRinv)):
            delta = y_cur - mu_k
            LRinvTdelta = np.dot(delta, LRinv_k)
            loglikelihoods[k] = \
                -0.5 * np.dot(LRinvTdelta, LRinvTdelta) + \
                -0.5 * self.m * np.log(2*np.pi) + \
                np.sum(np.log(np.diag(LRinv_k)))
        return loglikelihoods

    def _emission_loglikelihood(self, y_cur, z_cur, parameters):
        mu_k, LRinv_k = parameters.mu[z_cur], parameters.LRinv[z_cur]
        delta = y_cur - mu_k
        LRinvTdelta = np.dot(delta, LRinv_k)
        loglikelihood = \
            -0.5 * np.dot(LRinvTdelta, LRinvTdelta) + \
            -0.5 * self.m * np.log(2*np.pi) + \
            np.sum(np.log(np.diag(LRinv_k)))
        return loglikelihood

    def gradient_marginal_loglikelihood(self, observations, parameters,
            forward_message=None, backward_message=None,
            weights=None, use_scir=False, tqdm=None):
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
        mu = parameters.mu
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

            if use_scir:
                # Sufficient statistics for Stochastic Cox-Ingersoll-Ross
                if parameters.pi_type == 'logit':
                    grad['logit_pi'] = weight_t * joint_post
                elif parameters.pi_type == 'expanded':
                    grad['expanded_pi'] = weight_t * joint_post
                else:
                    raise RuntimeError()
            else:
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
            for k, mu_k, LRinv_k, Rinv_k, R_k in zip(
                    range(self.num_states), mu, LRinv, Rinv, R):
                diff_k = y_cur - mu_k
                grad['mu'][k] += weight_t * Rinv_k.dot(diff_k) * marg_post[k]
                grad_LRinv_k = weight_t * (
                        (R_k - np.outer(diff_k, diff_k)).dot(LRinv_k)
                        ) * marg_post[k]
                grad['LRinv_vec'][k] += grad_LRinv_k[np.tril_indices(self.m)]
        return grad


