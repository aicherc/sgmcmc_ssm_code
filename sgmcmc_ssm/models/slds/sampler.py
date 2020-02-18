import numpy as np
from ...sgmcmc_sampler import SGMCMCSampler, SeqSGMCMCSampler
from .parameters import SLDSPrior, SLDSParameters, SLDSPreconditioner
from .helper import SLDSHelper

class SLDSSampler(SGMCMCSampler):
    def __init__(self, num_states, n, m,
            observations=None, prior=None, parameters=None,
            forward_message=None, backward_message=None,
            name="SLDSSampler", **kwargs):
        self.options = kwargs
        self.num_states = num_states
        self.n = n
        self.m = m
        self.name = name
        self.setup(
                observations=observations,
                prior=prior,
                parameters=parameters,
                forward_message=forward_message,
                backward_message=backward_message,
                )
        return

    def setup(self, observations, prior, parameters=None, forward_message=None,
            backward_message=None):
        """ Initialize the sampler

        Args:
            observations (ndarray): T by m ndarray of time series values
            prior (SLDSPrior): prior
            forward_message (ndarray): prior probability for latent state
            parameters (SLDSParameters): initial parameters
                (optional, will sample from prior by default)

        """
        self.observations = observations

        if prior is None:
            prior = SLDSPrior.generate_default_prior(
                    num_states=self.num_states, n=self.n, m=self.m,
                    )
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
        self.backward_message = backward_message

        self.message_helper=SLDSHelper(
                num_states=self.num_states,
                n=self.n,
                m=self.m,
                forward_message=forward_message,
                backward_message=backward_message,
                )
        return

    def _check_observation_shape(self, observations):
        if observations is None:
            return
        if np.shape(observations)[1] != self.m:
            raise ValueError("observations second dimension does not match m")
        return

    def _get_preconditioner(self, preconditioner=None):
        if preconditioner is None:
            preconditioner = SLDSPreconditioner()
        return preconditioner


    def init_parameters_from_x_and_z(self, x, z):
        """ Get initial parameters for the sampler

        Args:
            x (ndarray): latent var
            z (ndarray): latent var

        Return:
            init_parameters (SLDSParameters): init_parameters
        """
        # Check z is appropriate size
        if np.shape(z)[0] != self._get_T():
            raise ValueError("z must be length T = {0}".format(self._get_T()))

        if not np.issubdtype(z.dtype, np.integer):
            raise ValueError("z must be integers, not {0}".format(z.dtype))

        if np.max(z) >= self.num_states or np.min(z) < 0:
            raise ValueError("z must be in (0, \ldots, {0}-1)".format(
                self.num_states))

        # Check x is appropriate size
        if np.shape(x)[0] != self._get_T() or np.shape(x)[1] != self.n:
            raise ValueError("x must be size {0} not {1}".format(
                (self._get_T(), self.n), np.shape(x)))

        # Init on Gibb Step
        init_parameters = self.message_helper.parameters_gibbs_sample(
                observations=self.observations,
                latent_vars=dict(x=x, z=z),
                forward_message=self.forward_message,
                backward_message=self.backward_message,
                prior=self.prior,
                )
        self.parameters = init_parameters

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
            if self._get_T() <= 10**6:
                kmeans = KMeans(n_clusters = self.num_states, **kwargs)
            else:
                kmeans = MiniBatchKMeans(n_clusters = self.num_states, **kwargs)

        X = self.observations.reshape((self._get_T(), -1))
        X_lagged = np.hstack([
            X[max(lags)-lag:X.shape[0]-lag] for lag in lags
        ])

        z = kmeans.fit_predict(X=X_lagged)
        if z.size < self._get_T():
            z = np.concatenate([np.zeros(self._get_T()-z.size, dtype=int), z])
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
                    noisy_gradient[var] /= self._get_T()
        else:
            scale = 1.0/self._get_T() if is_scaled else 1.0
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


