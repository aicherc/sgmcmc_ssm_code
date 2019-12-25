import numpy as np
from ...particle_filters.kernels import Kernel

# GARCH Kernel:
class GARCHPriorKernel(Kernel):
    # n = 2, first dimension is x_t, second is sigma2_t
    def sample_x0(self, prior_mean, prior_var, N, **kwargs):
        """ Initialize x_t

        Returns:
            x_t (N by n ndarray)
        """
        x_t = np.zeros((N, 2))
        x_t[:,0] = np.random.normal(
                loc=prior_mean,
                scale=np.sqrt(prior_var),
                size=(N))
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
        N = np.shape(x_next)[0]
        sigma2_next = self.parameters.alpha + \
                self.parameters.beta * x_t[:,0]**2 + \
                self.parameters.gamma * x_t[:,1]
        loglikelihoods = -0.5*x_next[:,0]**2/sigma2_next - \
                0.5 * np.log(2.0*np.pi) - 0.5 * np.log(sigma2_next)
        loglikelihoods = np.reshape(loglikelihoods, (N))
        return loglikelihoods

    def get_prior_log_density_max(self):
        """ Return max value of log density based on current parameters

        Returns max_{x,x'} log q(x | x', parameters)
        """
        alpha = self.parameters.alpha
        loglikelihood_max = -0.5*np.log(2.0*np.pi) - 0.5*np.log(alpha)
        return loglikelihood_max

    def rv(self, x_t, **kwargs):
        """ Prior Kernel for GARCH

        Sample x_{t+1} ~ Pr(x_{t+1} | x_t, parameters)

        Args:
            x_t (ndarray): N by n, x_t
        Return:
            x_next (ndarray): N by n, x_{t+1}

        """
        N = np.shape(x_t)[0]
        sigma2_next = self.parameters.alpha + \
                self.parameters.beta * x_t[:,0]**2 + \
                self.parameters.gamma * x_t[:,1]

        x_next = np.zeros((N,2))
        x_next[:,0] = np.sqrt(sigma2_next) * np.random.normal(size=N)
        x_next[:,1] = sigma2_next
        return x_next

    def reweight(self, x_t, x_next, **kwargs):
        """ Reweight function for Prior Kernel for GARCH

        weight_t = Pr(y_{t+1} | x_{t+1}, parameters)

        Args:
            x_t (ndarray): N by n, x_t
            x_next (ndarray): N by n, x_{t+1}
        Return:
            log_weights (ndarray): N, importance weights

        """
        N = np.shape(x_next)[0]
        diff = self.y_next - x_next[:,0]
        log_weights = \
                -0.5*np.log(2.0*np.pi) + \
                -0.5*(diff**2)*self.parameters.Rinv + \
                np.log(self.parameters.LRinv)
        log_weights = np.reshape(log_weights, (N))
        return log_weights

class GARCHOptimalKernel(Kernel):
    # n = 2, first dimension is x_t, second is sigma2_t
    def sample_x0(self, prior_mean, prior_var, N, **kwargs):
        """ Initialize x_t

        Returns:
            x_t (N by n ndarray)
        """
        x_t = np.zeros((N, 2))
        x_t[:,0] = np.random.normal(
                loc=prior_mean,
                scale=np.sqrt(prior_var),
                size=(N))
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
        N = np.shape(x_next)[0]
        sigma2_next = self.parameters.alpha + \
                self.parameters.beta * x_t[:,0]**2 + \
                self.parameters.gamma * x_t[:,1]
        loglikelihoods = -0.5*x_next[:,0]**2/sigma2_next - \
                0.5 * np.log(2.0*np.pi) - 0.5 * np.log(sigma2_next)
        loglikelihoods = np.reshape(loglikelihoods, (N))
        return loglikelihoods

    def get_prior_log_density_max(self):
        """ Return max value of log density based on current parameters

        Returns max_{x,x'} log q(x | x', parameters)
        """
        alpha = self.parameters.alpha
        loglikelihood_max = -0.5*np.log(2.0*np.pi) - 0.5*np.log(alpha)
        return loglikelihood_max


    def rv(self, x_t, **kwargs):
        """ Optimal Kernel for GARCH

        Sample x_{t+1} ~ Pr(x_{t+1} | x_t, y_{t+1}, parameters)

        Args:
            x_t (ndarray): N by n, x_t
        Return:
            x_next (ndarray): N by n, x_{t+1}

        """
        N = np.shape(x_t)[0]
        sigma2_next = self.parameters.alpha + \
                self.parameters.beta * x_t[:,0]**2 + \
                self.parameters.gamma * x_t[:,1]

        x_next = np.zeros((N,2))

        var_next = (self.parameters.Rinv + sigma2_next**-1)**-1
        mean_next = var_next*(self.y_next * self.parameters.Rinv)
        x_next[:,0] = mean_next + np.sqrt(var_next) * np.random.normal(size=N)
        x_next[:,1] = sigma2_next
        return x_next

    def reweight(self, x_t, x_next, **kwargs):
        """ Reweight function for Optimal Kernel for GARCH

        weight_t = Pr(y_{t+1} | x_t, parameters)

        Args:
            x_t (ndarray): N by n, x_t
            x_next (ndarray): N by n, x_{t+1}
        Return:
            log_weights (ndarray): N, importance weights

        """
        N = np.shape(x_next)[0]
        diff = self.y_next
        var = x_next[:,1] + self.parameters.R
        log_weights = \
                -0.5*np.log(2.0*np.pi) + \
                -0.5*(diff**2)/var + \
                -0.5*np.log(var)
        log_weights = np.reshape(log_weights, (N))
        return log_weights


