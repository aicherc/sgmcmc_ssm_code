import numpy as np
from ...particle_filters.kernels import LatentGaussianKernel

# SVM Kernels:
class SVMPriorKernel(LatentGaussianKernel):
    def set_parameters(self, parameters):
        self.parameters = parameters
        if np.abs(parameters.A) > 1:
            raise ValueError("Current AR parameter is |A| = {0} > 1".format(
                np.abs(parameters.A)) + "\nTry calling project_parameters?")
        return



    def rv(self, x_t, **kwargs):
        """ Prior Kernel for SVM

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
        """ Reweight function for Prior Kernel for SVM

        weight_t = Pr(y_{t+1} | x_{t+1}, parameters)

        Args:
            x_t (ndarray): N by n, x_t
            x_next (ndarray): N by n, x_{t+1}
        Return:
            log_weights (ndarray): N, importance weights

        """
        N = np.shape(x_next)[0]
        if (len(np.shape(x_t)) > 1) and (np.shape(x_t)[1] > 1):
            # n > 1
            raise NotImplementedError()
        else:
            # n = 1, x is scalar
            diff = self.y_next
            log_weights = \
                    -0.5*np.log(2.0*np.pi) + \
                    -0.5*(diff**2)*np.exp(-x_next)*self.parameters.Rinv + \
                    np.log(self.parameters.LRinv) + \
                    -0.5*x_next
        log_weights = np.reshape(log_weights, (N))
        return log_weights

