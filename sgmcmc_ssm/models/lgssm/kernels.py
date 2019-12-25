import numpy as np
from scipy.special import expit, logsumexp
from ...particle_filters.kernels import LatentGaussianKernel

# LGSSMKernels:
# Prior Kernel
class LGSSMPriorKernel(LatentGaussianKernel):
    """ Prior Kernel for LGSSM
        K(x_{t+1} | x_t) = Pr(x_{t+1} | x_t, parameters)
    """
    def rv(self, x_t, **kwargs):
        """ Prior Kernel for LGSSM

        Sample x_{t+1} ~ Pr(x_{t+1} | x_t, parameters)

        Args:
            x_t (ndarray): N by n, x_t
        Return:
            x_next (ndarray): N by n, x_{t+1}

        """
        if (len(np.shape(x_t)) > 1) and (np.shape(x_t)[1] > 1):
            # x is vector
            x_next_mean = x_t.dot(self.parameters.A.T)
            x_next = np.linalg.solve(self.parameters.LQinv.T,
                    np.random.normal(size=x_t.shape).T).T + x_next_mean
            return x_next
        else:
            # n = 1, x is scalar
            x_next_mean = x_t * self.parameters.A
            x_next = self.parameters.LQinv**-1 * np.random.normal(
                    size=x_t.shape) + x_next_mean
            return x_next

    def reweight(self, x_t, x_next, **kwargs):
        """ Reweight function for Prior Kernel for LGSSM

        weight_t = Pr(y_{t+1} | x_{t+1}, parameters)

        Args:
            x_t (ndarray): N by n, x_t
            x_next (ndarray): N by n, x_{t+1}
        Return:
            log_weights (ndarray): N, importance weights

        """
        N = np.shape(x_next)[0]
        if (len(np.shape(x_t)) > 1) and (np.shape(x_t)[1] > 1):
            # x is vector
            diff = np.outer(self.y_next, np.ones(N)) - \
                    np.dot(self.parameters.C, x_next.T)
            log_weights = \
                -0.5*np.shape(self.parameters.LRinv)[0]*np.log(2.0*np.pi) + \
                -0.5*np.sum(diff*np.dot(self.parameters.Rinv, diff), axis=0) + \
                np.sum(np.log(np.diag(self.parameters.LRinv)))
        else:
            # n = 1, x is scalar
            diff = self.y_next - self.parameters.C*x_next
            log_weights = \
                    -0.5*np.log(2.0*np.pi) + \
                    -0.5*(diff**2)*self.parameters.Rinv + \
                    np.log(self.parameters.LRinv)
        log_weights = np.reshape(log_weights, (N))
        return log_weights

# "Optimal" instrumental Kernel
class LGSSMOptimalKernel(LatentGaussianKernel):
    """ Optimal Instrumental Kernel for LGSSM
        K(x_{t+1} | x_t) = Pr(x_{t+1} | x_t, y_{t+1}, parameters)
    """
    def rv(self, x_t, **kwargs):
        """ optimal Kernel for LGSSM

        Sample x_{t+1} ~ Pr(x_{t+1} | x_t, y_{t+1}, parameters)

        Args:
            x_t (ndarray): N by n, x_t
        Return:
            x_next (ndarray): N by n, x_{t+1}

        """
        if (len(np.shape(x_t)) > 1) and (np.shape(x_t)[1] > 1):
            # x is vector
             raise NotImplementedError()
        else:
            # n = 1, x is scalar
            x_next_mean_precision = \
                    x_t * self.parameters.A * self.parameters.Qinv + \
                    self.y_next * self.parameters.C * self.parameters.Rinv
            x_next_precision = \
                    self.parameters.Qinv + \
                    (self.parameters.C**2)*self.parameters.Rinv

            x_next = (x_next_precision)**-0.5 * np.random.normal(
                    size=x_t.shape) + \
                    x_next_mean_precision/x_next_precision
            return x_next

    def reweight(self, x_t, x_next, **kwargs):
        """ Reweight function for Optimal Kernel for LGSSM

        weight_t = \Pr(y_t | x_{t-1}, parameters)

        Args:
            x_t (ndarray): N by n, x_t
            x_next (ndarray): N by n, x_{t+1}
        Return:
            log_weights (ndarray): N, importance weights

        """
        N = np.shape(x_next)[0]
        if (len(np.shape(x_t)) > 1) and (np.shape(x_t)[1] > 1):
            # x is vector
            raise NotImplementedError()
        else:
            # n = 1, x is scalar
            diff = self.y_next - self.parameters.A*x_t
            variance = self.parameters.Qinv**-1 + self.parameters.Rinv**-1
            log_weights = -0.5*(diff)**2 / variance - \
                    0.5*np.log(2.0*np.pi) - 0.5*np.log(variance)
        log_weights = np.reshape(log_weights, (N))
        return log_weights

# "Optimal" instrumental Kernel for high dimensional latent variables
class LGSSMHighDimOptimalKernel(LatentGaussianKernel):
    def set_parameters(self, parameters):
        self.parameters = parameters
        self._param = dict(
            AtQinv = np.dot(parameters.A.T, parameters.Qinv),
            CtRinv = np.dot(parameters.C.T, parameters.Rinv),
            CtRinvC = np.dot(parameters.C.T,
                np.dot(parameters.Rinv, parameters.C)
                ),
            )
        x_next_Lvar = np.linalg.inv(np.linalg.cholesky(
                parameters.Qinv + self._param['CtRinvC']
                )).T
        self._param['x_next_Lvar'] = x_next_Lvar
        self._param['x_next_var'] = np.dot(x_next_Lvar, x_next_Lvar.T)
        self._param['predictive_var'] = (self.parameters.R + np.dot(
                self.parameters.C,
                np.dot(self.parameters.Q, self.parameters.C.T),
                ))
        self._param['predictive_var_logdet'] = np.linalg.slogdet(
                self._param['predictive_var'])[1]
        return

    def rv(self, x_t, **kwargs):
        """ optimal Kernel for LGSSM

        Sample x_{t+1} ~ Pr(x_{t+1} | x_t, y_{t+1}, parameters)

        Args:
            x_t (ndarray): N by n, x_t
        Return:
            x_next (ndarray): N by n, x_{t+1}

        """
        if (len(np.shape(x_t)) > 1) and (np.shape(x_t)[1] > 1):
            # x is vector
            x_next_mean = np.dot(
                    np.dot(x_t, self._param['AtQinv']) +
                    np.outer(
                        np.ones(np.shape(x_t)[0]),
                        np.dot(self._param['CtRinv'], self.y_next),
                    ),
                    self._param['x_next_var']
                    )
            x_next = np.dot(
                    np.random.normal(size=x_t.shape),
                    self._param['x_next_Lvar'].T,
                    ) + x_next_mean
            return x_next
        else:
            raise NotImplementedError()

    def reweight(self, x_t, x_next, **kwargs):
        """ Reweight function for Optimal Kernel for LGSSM

        weight_t = \Pr(y_t | x_{t-1}, parameters)

        Args:
            x_t (ndarray): N by n, x_t
            x_next (ndarray): N by n, x_{t+1}
        Return:
            log_weights (ndarray): N, importance weights

        """
        N = np.shape(x_next)[0]
        if (len(np.shape(x_t)) > 1) and (np.shape(x_t)[1] > 1):
            # x is vector
            diff = \
                np.outer(np.ones(N), self.y_next) - np.dot(x_t, self.parameters.A)
            log_weights = (
                    -0.5*np.sum(diff *
                    np.linalg.solve(self._param['predictive_var'], diff.T).T,
                    axis=1) +\
                    -0.5*np.shape(x_t)[1]*np.log(2.0) + \
                    -0.5*self._param['predictive_var_logdet']
                    )
            return log_weights
        else:
            # n = 1, x is scalar
            raise NotImplementedError()

# EOF
