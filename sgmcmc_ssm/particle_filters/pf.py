"""
Bootstrap Particle Filter
"""
import numpy as np
import functools

def pf(particles, log_weights, kernel):
    """ Algorithm 1 of PaRIS

    {particles_t, weights_t} -> {particles_{t+1}, weights_{t+1}}

    Args:
        particles (ndarray): N by n, latent states (xi_t)
        log_weights (ndarray): N, log importance weights (log omega_t)
        kernel (Kernel): kernel with functions
            rv (func): proposal function for new particles
            reweight (func): reweighting function for new particles

    Returns:
        new_particles (ndarray): N by n, latent states (xi_{t+1})
        new_log_weights (ndarray): N, log importance weights (log omega_{t+1})
        ancestor_indices (ndarray): N, ancestor indicies (I)
    """
    N = np.shape(particles)[0]

    # Multinomial Resampling
    ancestor_log_weights = kernel.ancestor_log_weights(particles, log_weights)
    ancestor_indices = np.random.choice(range(N),
            size=N, replace=True, p=log_normalize(ancestor_log_weights))
    sampled_particles = particles[ancestor_indices]

    # Propose Descendents
    new_particles = kernel.rv(sampled_particles)

    # Weight Descendents
    new_log_weights = kernel.reweight(sampled_particles, new_particles)

    return new_particles, new_log_weights, ancestor_indices

def pf_filter(particles, log_weights, statistics, additive_statistic_func, kernel, **kwargs):
    """ Calculate SUM[ E[h(x_t, x_{t+1}) | Y_{<=t+1}] ]

    Args:
        particles (ndarray): N by n, latent states (x_t)
        log_weights (ndarray): N, log importance weights (log w_t)
        statistics (ndarray): h, estimated aux statistics (m_t)

        additive_statistic_func (func): h_t(x_t, x_{t+1})
            function of x_t, x_{t+1}, return h_t(x_t, x_{t+1})
        kernel (Kernel): aux kernel for pf
            rv (func): proposal function for new particles
            reweight (func): reweighting function for new particles
            log_density (func): return log_density of K(xi_t, xi_{t+1})

    Return:
        new_particles (ndarray): N by n, latent states (x_{t+1})
        new_log_weights (ndarray): N, log importance weights (log w_{t+1})
        new_statistics (ndarray): h, estimated aux statistics (m_{t+1})
    """
    N = np.shape(particles)[0]

    new_particles, new_log_weights, ancestor_indices = pf(
            particles=particles, log_weights=log_weights,
            kernel=kernel,
            )
    additive_statistic = \
            additive_statistic_func(
                    x_t=particles[ancestor_indices],
                    x_next=new_particles,
                    )
    additive_statistic *= kwargs.get('additive_scale', 1.0)

    if kwargs.get('logsumexp', False):
        max_add_stat = np.max(additive_statistic.T, axis=1)
        new_statistics = statistics + max_add_stat + \
                np.log(np.sum(np.exp(additive_statistic.T- max_add_stat[:,np.newaxis]) * log_normalize(new_log_weights)))
    else:
        new_statistics = statistics + \
                np.sum(additive_statistic.T * log_normalize(new_log_weights),
                        axis=1)

    return new_particles, new_log_weights, new_statistics

def poyiadjis_smoother(particles, log_weights, statistics,
        additive_statistic_func, kernel,
        **kwargs):
    """ Algorithm 2 of Poyiadjis et al. Biometrika (2011) O(N^2) algorithm

    Calculate E[SUM[h(x_t, x_{t+1})] | Y]

    Args:
        particles (ndarray): N by n, latent states (x_t)
        log_weights (ndarray): N, log importance weights (log w_t)
        statistics (ndarray): N by h, estimated aux statistics (m_t)

        additive_statistic_func (func): h_t(x_t, x_{t+1})
            function of x_t, x_{t+1}, return h_t(x_t, x_{t+1})
        kernel (Kernel): aux kernel for pf
            rv (func): proposal function for new particles
            reweight (func): reweighting function for new particles
            log_density (func): return log_density of K(xi_t, xi_{t+1})

    Return:
        new_particles (ndarray): N by n, latent states (x_{t+1})
        new_log_weights (ndarray): N, log importance weights (log w_{t+1})
        new_statistics (ndarray): N by h, estimated aux statistics (m_{t+1})
    """
    N = np.shape(particles)[0]

    new_particles, new_log_weights, ancestor_indices = pf(
            particles=particles, log_weights=log_weights,
            kernel=kernel,
            )

    backward_weights = np.zeros((N, N)) # Weights for Eq. (20)
    for i, x_next in enumerate(new_particles):
        child_loglikelihood = kernel.prior_log_density(
                particles, np.outer(np.ones(N), x_next),
                )
        # J_{t+1}
        backward_weights[i] = log_normalize(log_weights + child_loglikelihood)

    indices = np.array([ii for _ in range(N) for ii in range(N)])     # 0,1,2,..., 0,1,2,...,0,1,2,...
    new_indices = np.array([ii for ii in range(N) for _ in range(N)]) # 0,0,0,...,1,1,1,...,2,2,2,...
    additive_statistic = additive_statistic_func(
            x_t=particles[indices],
            x_next=new_particles[new_indices],
            )
    additive_statistic *= kwargs.get('additive_scale', 1.0)

    new_statistics = np.reshape(
        statistics[indices] + additive_statistic,
        (N, N, -1)
        )
    new_statistics = np.einsum('ijk,ij->ik', new_statistics, backward_weights)
    return new_particles, new_log_weights, new_statistics

def nemeth_smoother(particles, log_weights, statistics,
        additive_statistic_func, kernel,
        lambduh = 0.95, **kwargs):
    """ Algorithm 2 of Nemeth et al. (2015)

    Calculate E[SUM[h(x_t, x_{t+1})] | Y]

    Args:
        particles (ndarray): N by n, latent states (x_t)
        log_weights (ndarray): N, log importance weights (log w_t)
        statistics (ndarray): N by h, estimated aux statistics (m_t)

        additive_statistic_func (func): h_t(x_t, x_{t+1})
            function of x_t, x_{t+1}, return h_t(x_t, x_{t+1})
        kernel (Kernel): aux kernel for pf
        lambduh (double, optional): shrinkage parameter

    Return:
        new_particles (ndarray): N by n, latent states (x_{t+1})
        new_log_weights (ndarray): N, log importance weights (log w_{t+1})
        new_statistics (ndarray): N by h, estimated aux statistics (m_{t+1})
    """
    N = np.shape(particles)[0]
    S = np.sum(statistics.T * log_normalize(log_weights), axis=1)

    new_particles, new_log_weights, ancestor_indices = pf(
            particles=particles, log_weights=log_weights,
            kernel=kernel,
            )

    additive_statistic = \
            additive_statistic_func(
                    x_t=particles[ancestor_indices],
                    x_next=new_particles,
                    )
    additive_statistic *= kwargs.get('additive_scale', 1.0)

    new_statistics = (
            lambduh * statistics[ancestor_indices] +
            (1.0-lambduh) * np.outer(np.ones(N), S) +
            additive_statistic
            )

    return new_particles, new_log_weights, new_statistics

def paris_smoother(particles, log_weights, statistics,
        additive_statistic_func, kernel,
        Ntilde=2, accept_reject=True,
        max_accept_reject=None, manual_sample_threshold=None,
        **kwargs):
    """ Algorithm 2 of PaRIS

    Calculate E[SUM[h(x_t, x_{t+1})] | Y]

    Args:
        particles (ndarray): N by n, latent states (xi_t)
        log_weights (ndarray): N, log importance weights (log omega_t)
        statistics (ndarray): N by h, estimated aux statistics (tau_t)

        additive_statistic_func (func): h_t(xi_t, xi_{t+1})
            function of xi_t, xi_{t+1}, return h_t(xi_t, xi_{t+1})
        kernel (Kernel): kernel with functions
            rv (func): proposal function for new particles
            reweight (func): reweighting function for new particles
            log_density (func): return log_density of K(xi_t, xi_{t+1})

        Ntilde (int, optional): precision parameter

    Return:
        new_particles (ndarray): N by n, latent states (xi_{t+1})
        new_log_weights (ndarray): N, log importance weights (log omega_{t+1})
        new_statistics (ndarray): N by h, estimated aux statistics (tau_{t+1})
    """
    N = np.shape(particles)[0]

    new_particles, new_log_weights, _ = pf(
            particles=particles, log_weights=log_weights,
            kernel=kernel,
            )

    if accept_reject:
        # Accept-Reject O(NK) Implementation
        rewired_ancestor_indices = accept_reject_based_backward_sampling(
                particles, log_weights, new_particles, kernel, Ntilde,
                max_accept_reject=max_accept_reject,
                manual_sample_threshold=manual_sample_threshold,
                )

    else:
        # Naive O(N^2) Implementation
        rewired_ancestor_indices = [None] * N
        for i, xi_next in enumerate(new_particles):
            child_loglikelihood = kernel.prior_log_density(
                    particles, np.outer(np.ones(N), xi_next),
                    )
            # J_{t+1}
            rewired_ancestor_indices[i] = np.random.choice(
                    range(N), size = Ntilde, replace = True,
                    p=log_normalize(log_weights + child_loglikelihood))
        rewired_ancestor_indices = np.array(rewired_ancestor_indices)

    # Vectorized Update of Additive Statistics
    rewired_statistics = statistics[rewired_ancestor_indices.flatten()]

    rewired_parents = particles[rewired_ancestor_indices.flatten()]
    indices = np.array([ii for ii in range(N) for _ in range(Ntilde)])
    xi_next = new_particles[indices]
    additive_statistic = additive_statistic_func(
            x_t=rewired_parents,
            x_next=xi_next,
            )
    additive_statistic *= kwargs.get('additive_scale', 1.0)

    new_statistics = np.reshape(
            rewired_statistics + additive_statistic,
            (N, Ntilde, -1)
            )

    new_statistics = np.mean(new_statistics, axis=1)

    return new_particles, new_log_weights, new_statistics

def accept_reject_based_backward_sampling(particles, log_weights, new_particles,
        kernel, Ntilde, max_accept_reject=None, manual_sample_threshold=None):
    """ Algorithm 3 of PaRIS to sample J (rewired ancestor indices)

    Args:
        particles (ndarray): N by n, latent states (xi_t)
        log_weights (ndarray): N, log importance weights (log omega_t)
        new_particles (ndarray): N by n, latent states (xi_{t+1})
        kernel (Kernel): kernel
        Ntilde (int): precision parameter
        max_accept_reject (int, optional): number of accept_reject tries
            (default is 100*log10(N/10))
        manual_sample_threshold (int, optional):
            threshold number of samples to manually sample
            early terminates accept_reject
            (default is 10*log10(N/10))

    Returns:
        rewired_ancestor_indices (ndarray): N by Ntilde, indices J
    """
    N = np.shape(particles)[0]
    weights = log_normalize(log_weights)
    loglikelihood_max = kernel.get_prior_log_density_max()

    if max_accept_reject is None:
        max_accept_reject = int(100*np.log10(N/10))
    if manual_sample_threshold is None:
        manual_sample_threshold = int(10*np.log10(N/10))

    # J
    rewired_ancestor_indices = np.zeros((N, Ntilde), dtype=int)

    for j in range(Ntilde):
        L = [ii for ii in range(N)]
        converged = False
        for _ in range(max_accept_reject):
            size_L = len(L)
            new_L = []

            # Exit when L is empty
            if size_L == 0:
                converged = True
                break
            # Early terminate to manual resample when L is small
            if size_L <= manual_sample_threshold:
                break

            # Draw I
            indices = np.random.choice(
                    range(N), size=size_L, replace=True, p=weights,
                    )
            # Draw U
            uniforms = np.random.rand((size_L))


            # Calculate q(xi^I, xi^L)
            child_loglikelihood = kernel.prior_log_density(
                    particles[indices], new_particles[L])
            threshold = np.exp(child_loglikelihood-loglikelihood_max)
            for k in range(size_L):
                if uniforms[k] <= threshold[k]:
                    # Set J[L[k], j] = I[k]
                    rewired_ancestor_indices[L[k], j] = indices[k]
                else:
                    new_L.append(L[k])
            L = new_L
            #print("Not Converged {0} of {1} after {2} of {3} steps".format(len(L), N, _, max_accept_reject))

        if not converged:
            # Manually Sample remaining i
            #print("Manually Sampling {0} of {1} after {2} of {3} steps".format(len(L), N, _, max_accept_reject))
            for i in L:
                child_loglikelihood = kernel.prior_log_density(
                        particles,
                        np.outer(np.ones(N), new_particles[i]),
                        )
                # J_{t+1}
                rewired_ancestor_indices[i,j] = np.random.choice(
                        range(N), size = 1, replace = True,
                        p=log_normalize(log_weights + child_loglikelihood))

    return rewired_ancestor_indices

def efficient_multiomial_sampling(num_samples, prob_weight):
    """ Efficient multinomial sampling of num_samples from N
        Takes O(n + n log(1+N/n)
        Algorithm 2 in Appendix B.1 of https://arxiv.org/pdf/1202.2945.pdf

        Note I did not find this faster than numpy's np.random.choice
    """
    N = len(prob_weight)
    indices = np.zeros((num_samples), dtype=int) # I

    cum_weights = np.cumsum(prob_weight) # q
    ordered_U = np.random.rand(num_samples) # U
    ordered_U.sort()

    l, r = 0, 1
    for k in range(num_samples):
        d = 1
        while ordered_U[k] > cum_weights[r-1]:
            l = r
            r = min([r+2**d, N])
            d = d+1
        while r-l > 1:
            m = int(np.floor((r+l)/2))
            if ordered_U[k] > cum_weights[m-1]:
                l = m
            else:
                r = m
        indices[k] = r-1

    return np.random.permutation(indices)

def log_normalize(log_weights):
    probs = np.exp(log_weights-np.max(log_weights))
    probs /= np.sum(probs)
    return probs





