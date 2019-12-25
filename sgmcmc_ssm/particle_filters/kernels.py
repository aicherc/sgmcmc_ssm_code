"""
Kernels for particle filters

"""
import numpy as np
from scipy.special import expit, logsumexp

# Kernel
class Kernel(object):
    def __init__(self, **kwargs):
        self.parameters = kwargs.get('parameters', None)
        self.y_next = kwargs.get('y_next', None)
        return

    def set_parameters(self, parameters):
        self.parameters = parameters
        return

    def set_y_next(self, y_next):
        self.y_next = y_next
        return

    def sample_x0(self, prior_mean, prior_var, N, n):
        """ Initialize x_t

        Returns:
            x_t (N by n ndarray)
        """
        raise NotImplementedError()

    def rv(self, x_t, **kwargs):
        """ Sample x_{t+1} ~ K(x_{t+1} | x_t, parameters)

        Args:
            x_t (ndarray): N by n, x_t
            parameters (dict): parameters
        Return:
            x_next (ndarray): N by n, x_{t+1}

        """
        raise NotImplementedError()

    def reweight(self, x_t, x_next, **kwargs):
        """ Reweight function for Kernel

        weight_t = Pr(y_{t+1}, x_{t+1} | x_t, parameters) /
                    K(x_{t+1} | x_t, parameters)

        Args:
            x_t (ndarray): N by n, x_t
            x_next (ndarray): N by n, x_{t+1}
        Return:
            log_weights (ndarray): N, importance weights

        """
        raise NotImplementedError()

    def log_density(self, x_t, x_next, **kwargs):
        """ Density of kernel K(x_{t+1} | x_t, parameters)

        Args:
            x_t (N by n ndarray): x_t
            x_next (N by n ndarray): x_{t+1}

        Returns:
            loglikelihoods (N ndarray): K(x_next | x_t, parameters)
                (ignores constants with respect to x_t & x_next)
        """
        raise NotImplementedError()

    def get_prior_log_density_max(self):
        """ Upper bound for prior log density """
        raise NotImplementedError()

    def ancestor_log_weights(self, particles, log_weights):
        """ Weights for ancestor sampling
        Default is log_weights
        """
        return log_weights

# LatentGaussianKernel
class LatentGaussianKernel(Kernel):
    def sample_x0(self, prior_mean, prior_var, N, n=1):
        """ Initialize x_t

        Returns:
            x_t (N by n ndarray)
        """
        if n == 1:
            x_t = np.random.normal(
                    loc=prior_mean,
                    scale=np.sqrt(prior_var),
                    size=(N, n))
        else:
            x_t = np.random.multivariate_normal(
                    mean=prior_mean,
                    cov=prior_var,
                    size=N,
                    )
        return x_t

    def prior_log_density(self, x_t, x_next, **kwargs):
        """ log density of prior kernel

        Args:
            x_t (N by n ndarray): x_t
            x_next (N by n ndarray): x_{t+1}

        Returns:
            loglikelihoods (N ndarray): q(x_next | x_t, parameters)
                (ignores constants with respect to x_t & x_next
        """
        N = np.shape(x_t)[0]
        if (len(np.shape(x_t)) > 1) and (np.shape(x_t)[1] > 1):
            # x is vector
            diff = x_next.T - np.dot(self.parameters.A, x_t.T)
            loglikelihoods = -0.5*np.sum(diff*np.dot(
                self.parameters.Qinv, diff), axis=0) - \
                        0.5*np.shape(x_t)[1] * np.log(2.0*np.pi) + \
                        np.sum(np.log(np.diag(self.parameters.LQinv)))
        else:
            # n = 1, x is scalar
            diff = x_next - self.parameters.A*x_t
            loglikelihoods = -0.5*(diff**2)*self.parameters.Qinv + \
                    -0.5*np.log(2.0*np.pi) + np.log(self.parameters.LQinv)
        loglikelihoods = np.reshape(loglikelihoods, (N))
        return loglikelihoods

    def get_prior_log_density_max(self):
        """ Return max value of log density based on current parameters

        Returns max_{x,x'} log q(x | x', parameters)
        """
        LQinv = self.parameters.LQinv
        n = np.shape(LQinv)[0]
        loglikelihood_max = -0.5*n*np.log(2.0*np.pi) + \
                np.sum(np.log(np.diag(LQinv)))
        return loglikelihood_max





