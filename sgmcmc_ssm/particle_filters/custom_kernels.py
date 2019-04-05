"""
Advanced Kernels for particle filters

"""
from .kernels import *
from scipy.optimize import root_scalar
from scipy.special import logsumexp, roots_hermitenorm

class SVMLaplaceKernel(LatentGaussianKernel):
    def __init__(self, **kwargs):
        self.approx_param=dict(mean=0, var=1)
        super().__init__(**kwargs)
        return

    def rv(self, x_t, **kwargs):
        """ Laplace Kernel for SVM

        Sample x_{t+1} ~ q(x_{t+1} | x_t, parameters)

        Args:
            x_t (ndarray): N by n, x_t
        Return:
            x_next (ndarray): N by n, x_{t+1}

        """
        x_next_mean = np.mean(x_t) * self.parameters.A
        x_next_var = np.var(x_t) * self.parameters.A**2 + self.parameters.Q

        scaled_y2 = (self.parameters.LRinv*self.y_next)**2
        taylor_deriv = lambda x: 0.5*scaled_y2*np.exp(-x) - 0.5 - (x-x_next_mean)/x_next_var

        laplace_mean = root_scalar(taylor_deriv,
                bracket=[-100*np.sqrt(x_next_var), 100*np.sqrt(x_next_var)]
                ).root
        laplace_var = 0.5*scaled_y2*exp(-laplace_mean) + x_next_var**-1
        self.approx_param['mean'] = laplace_mean
        self.approx_param['var'] = laplace_var

        x_next = np.sqrt(laplace_var) * np.random.normal(
                size=x_t.shape) + laplace_mean
        return x_next

    def reweight(self, x_t, x_next, **kwargs):
        """ Reweight function for Laplace Kernel for SVM

        weight_t = p(y_{t+1}, x_{t+1} | x_t, parameters)/q(x_{t+1}, x_t)

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
            scaled_y2 = (self.parameters.LRinv*self.y_next)**2
            log_weights = \
                    -0.5*np.log(2.0*np.pi) + \
                    -0.5*scaled_y2*np.exp(-x_next) + \
                    np.log(self.parameters.LRinv) + \
                    -0.5*x_next
            diff = x_next - self.parameters.A*x_t
            loglike = -0.5*(diff**2)*self.parameters.Qinv + \
                -0.5*np.log(2.0*np.pi) + np.log(self.parameters.LQinv)
            kernel_diff = x_next - self.approx_param['mean']
            kernel_like = -0.5*(diff**2)/self.approx_param['var'] + \
                -0.5*np.log(2.0*np.pi) - 0.5*np.log(self.approx_param['var'])

        log_weights = np.reshape(log_weights+loglike-kernel_like, (N))
        return log_weights

class SVMEPKernel(LatentGaussianKernel):
    def __init__(self, **kwargs):
        self.approx_param=dict(mean=0, var=1)
        super().__init__(**kwargs)
        return

    def rv(self, x_t, **kwargs):
        """ EP Approx Kernel for SVM

        Sample x_{t+1} ~ q(x_{t+1} | x_t, parameters)

        Args:
            x_t (ndarray): N by n, x_t
        Return:
            x_next (ndarray): N by n, x_{t+1}

        """
        x_next_mean = np.mean(x_t) * self.parameters.A
        x_next_var = np.var(x_t) * self.parameters.A**2 + self.parameters.Q

        scaled_y2 = (self.parameters.LRinv*self.y_next)**2

        # Gauss Quadrature for EP
        x_i = roots_hermitenorm(100)[0] * np.sqrt(x_next_var) + x_next_mean
        log_w_i = -0.5*(x_i - x_next_mean)/x_next_var + \
                -0.5*np.log(2*np.pi*x_next_var) + \
                -0.5*scaled_y2*np.exp(-x_i) + \
                -0.5*x_i -0.5*np.log(2*np.pi)

        w_i = np.exp(log_w_i - logsumexp(log_w_i))
        approx_mean = np.sum(x_i*w_i)
        approx_var = np.sum(x_i**2*w_i) - approx_mean**2
        self.approx_param['mean'] = approx_mean
        self.approx_param['var'] = approx_var

        x_next = np.sqrt(approx_var) * np.random.normal(
                size=x_t.shape) + approx_mean
        return x_next

    def reweight(self, x_t, x_next, **kwargs):
        """ Reweight function for EP Approx Kernel for SVM

        weight_t = p(y_{t+1}, x_{t+1} | x_t, parameters)/q(x_{t+1}, x_t)

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
            scaled_y2 = (self.parameters.LRinv*self.y_next)**2
            log_weights = \
                    -0.5*np.log(2.0*np.pi) + \
                    -0.5*scaled_y2*np.exp(-x_next) + \
                    np.log(self.parameters.LRinv) + \
                    -0.5*x_next
            diff = x_next - self.parameters.A*x_t
            loglike = -0.5*(diff**2)*self.parameters.Qinv + \
                -0.5*np.log(2.0*np.pi) + np.log(self.parameters.LQinv)
            kernel_diff = x_next - self.approx_param['mean']
            kernel_like = -0.5*(diff**2)/self.approx_param['var'] + \
                -0.5*np.log(2.0*np.pi) - 0.5*np.log(self.approx_param['var'])

        log_weights = np.reshape(log_weights+loglike-kernel_like, (N))
        return log_weights

class SVJMEPKernel(SVJMPriorKernel):
    def _calc_ep_fit(self, x_t):
        # Gauss Quadrature for EP
        pJ = self.parameters.pJ
        x_next_mean = x_t.T * self.parameters.phi
        x_next_var = self.parameters.sigma2
        x_nextJ_var = x_next_var + self.parameters.sigmaJ2

        scaled_y2 = (self.parameters.Ltau2inv*self.y_next)**2

        x_i = roots_hermitenorm(100)[0]
        x_1 = x_i[:, np.newaxis] * np.sqrt(x_nextJ_var) + x_next_mean
        x_2 = x_i[:, np.newaxis] * np.sqrt(x_next_var) + x_next_mean

        log_w_1 = -0.5*(x_1 - x_next_mean)**2/x_nextJ_var + \
                -0.5*np.log(2*np.pi*x_nextJ_var)
        log_perturb_1 =-0.5*scaled_y2*np.exp(-x_1) + \
                -0.5*x_1 -0.5*np.log(2*np.pi)
        w_1 = np.exp(log_perturb_1 + log_w_1 - logsumexp(log_w_1, axis=0))
        x_z1 = np.sum(w_1, axis=0)
        x_bar1 = np.sum(x_1*w_1, axis=0)/x_z1
        x_var1 = np.sum(x_1**2*w_1, axis=0)/x_z1 - x_bar1**2

        log_w_2 = -0.5*(x_2 - x_next_mean)**2/x_next_var + \
                -0.5*np.log(2*np.pi*x_next_var)
        log_perturb_2 =-0.5*scaled_y2*np.exp(-x_2) + \
                -0.5*x_2 -0.5*np.log(2*np.pi)
        w_2 = np.exp(log_perturb_2 + log_w_2 - logsumexp(log_w_2, axis=0))
        x_z2 = np.sum(w_2, axis=0)
        x_bar2 = np.sum(x_2*w_2, axis=0)/x_z2
        x_var2 = np.sum(x_2**2*w_2, axis=0)/x_z2 - x_bar2**2

        x_pJ = pJ*x_z1/(pJ*x_z1 + (1-pJ)*x_z2)
        return dict(xJ_bar = x_bar1, xJ_var=x_var1,
                x_bar=x_bar2, x_var=x_var2, x_pJ=x_pJ)

    def rv(self, x_t, **kwargs):
        """ EP Mixture Approx Kernel for SVM

        Sample x_{t+1} ~ q(x_{t+1} | x_t, parameters)

        Args:
            x_t (ndarray): N by n, x_t
        Return:
            x_next (ndarray): N by n, x_{t+1}

        """
        ep_fit = self._calc_ep_fit(x_t)

        jump_ind = np.random.rand(x_t.size) < ep_fit['x_pJ']
        x_next_sd = (jump_ind*np.sqrt(ep_fit['xJ_var']) + \
                (1-jump_ind)*np.sqrt(ep_fit['x_var']))
        x_next_mean = (jump_ind*ep_fit['xJ_bar'] + (1-jump_ind)*ep_fit['x_bar'])
        x_next = np.random.normal(size=x_t.shape) * x_next_sd[:,np.newaxis] + x_next_mean[:,np.newaxis]
        return x_next

    def reweight(self, x_t, x_next, **kwargs):
        """ Reweight function for EP Approx Kernel for SVM

        weight_t = p(y_{t+1}, x_{t+1} | x_t, parameters)/q(x_{t+1}, x_t)

        Args:
            x_t (ndarray): N by n, x_t
            x_next (ndarray): N by n, x_{t+1}
        Return:
            log_weights (ndarray): N, importance weights

        """
        N = np.shape(x_next)[0]
        # log P(y_t+1 | x_t+1)
        scaled_y2 = (self.parameters.Ltau2inv*self.y_next)**2
        log_weights = \
                -0.5*np.log(2.0*np.pi) + \
                -0.5*scaled_y2*np.exp(-x_next) + \
                np.log(self.parameters.Ltau2inv) + \
                -0.5*x_next

        # log P(x_t+1 | x_t)
        x_diff = x_next - self.parameters.phi*x_t
        sigma2_nojump = self.parameters.sigma2
        sigma2_jump = self.parameters.sigma2 + self.parameters.sigmaJ2
        pJ = self.parameters.pJ
        loglike_nojump = -0.5*(x_diff**2)/sigma2_nojump+\
                -0.5*np.log(2.0*np.pi) - 0.5*np.log(sigma2_nojump)
        loglike_jump = -0.5*(x_diff**2)/sigma2_jump+\
                -0.5*np.log(2.0*np.pi) - 0.5*np.log(sigma2_jump)
        loglike_max = np.max(np.array([loglike_jump, loglike_nojump]), axis=0)
        loglike = np.log(
                pJ*np.exp(loglike_jump-loglike_max) +
                (1-pJ)*np.exp(loglike_nojump-loglike_max)
                ) + loglike_max

        # log Q(x_t+1 | x_t)
        ep_fit = self._calc_ep_fit(x_t)

        logker_nojump = \
            -0.5*((x_next[:,0]-ep_fit['x_bar'])**2)/ep_fit['x_var'] + \
            -0.5*np.log(2.0*np.pi) - 0.5*np.log(ep_fit['x_var'])
        logker_jump = \
            -0.5*((x_next[:,0]-ep_fit['xJ_bar'])**2)/ep_fit['xJ_var'] + \
            -0.5*np.log(2.0*np.pi) - 0.5*np.log(ep_fit['xJ_var'])
        logker_max = np.max(np.array([logker_jump, logker_nojump]), axis=0)
        logker = np.log(
                ep_fit['x_pJ']*np.exp(logker_jump-logker_max) +
                (1-ep_fit['x_pJ'])*np.exp(logker_nojump-logker_max)
                ) + logker_max

        log_weights = np.reshape(log_weights+loglike, (N)) - logker
        return log_weights

class SVJMEPAvgKernel(SVJMPriorKernel):
    def __init__(self, **kwargs):
        self.approx_param=dict(mean=0, var=1, mean_J=0, var_J=1, pJ=0.5)
        super().__init__(**kwargs)
        return

    def rv(self, x_t, **kwargs):
        """ EP Mixture Approx Kernel for SVM

        Sample x_{t+1} ~ q(x_{t+1} | x_t, parameters)

        Args:
            x_t (ndarray): N by n, x_t
        Return:
            x_next (ndarray): N by n, x_{t+1}

        """
        pJ = self.parameters.pJ
        x_next_mean = np.mean(x_t) * self.parameters.phi
        x_next_var = np.var(x_t) * self.parameters.phi**2 + \
                self.parameters.sigma2
        x_nextJ_var = x_next_var + self.parameters.sigmaJ2

        scaled_y2 = (self.parameters.Ltau2inv*self.y_next)**2

        # Gauss Quadrature for EP
        x_i = roots_hermitenorm(100)[0]
        x_1 = x_i * np.sqrt(x_nextJ_var) + x_next_mean
        x_2 = x_i * np.sqrt(x_next_var) + x_next_mean

        log_w_1 = -0.5*(x_1 - x_next_mean)**2/x_nextJ_var + \
                -0.5*np.log(2*np.pi*x_nextJ_var)
        log_perturb_1 =-0.5*scaled_y2*np.exp(-x_1) + \
                -0.5*x_1 -0.5*np.log(2*np.pi)
        w_1 = np.exp(log_perturb_1 + log_w_1 - logsumexp(log_w_1))
        x_z1 = np.sum(w_1)
        x_bar1 = np.sum(x_1*w_1)/x_z1
        x_var1 = np.sum(x_1**2*w_1)/x_z1 - x_bar1**2

        log_w_2 = -0.5*(x_2 - x_next_mean)**2/x_next_var + \
                -0.5*np.log(2*np.pi*x_next_var)
        log_perturb_2 =-0.5*scaled_y2*np.exp(-x_2) + \
                -0.5*x_2 -0.5*np.log(2*np.pi)
        w_2 = np.exp(log_perturb_2 + log_w_2 - logsumexp(log_w_2))
        x_z2 = np.sum(w_2)
        x_bar2 = np.sum(x_2*w_2)/x_z2
        x_var2 = np.sum(x_2**2*w_2)/x_z2 - x_bar2**2

        x_pJ = pJ*x_z1/(pJ*x_z1 + (1-pJ)*x_z2)

        self.approx_param['mean'] = x_bar2
        self.approx_param['var'] = x_var2
        self.approx_param['mean_J'] = x_bar1
        self.approx_param['var_J'] = x_var1
        self.approx_param['pJ'] = x_pJ

        if x_var1 > x_var2:
            nojump_ind = np.random.rand(x_t.size) > x_pJ
            x_next = np.sqrt(x_var2)*np.random.normal(size=x_t.shape) + x_bar2
            x_next += nojump_ind[:,np.newaxis]*(
                    np.random.normal(size=x_t.shape)*np.sqrt(x_var1-x_var2) +
                    x_bar1-x_bar2)
            return x_next
        else:
            jump_ind = np.random.rand(x_t.size) < x_pJ
            x_next = np.sqrt(x_var1)*np.random.normal(size=x_t.shape) + x_bar1
            x_next += jump_ind[:,np.newaxis]*(
                    np.random.normal(size=x_t.shape)*np.sqrt(x_var2-x_var1) +
                    x_bar2-x_bar1)
            return x_next

    def reweight(self, x_t, x_next, **kwargs):
        """ Reweight function for EP Approx Kernel for SVM

        weight_t = p(y_{t+1}, x_{t+1} | x_t, parameters)/q(x_{t+1}, x_t)

        Args:
            x_t (ndarray): N by n, x_t
            x_next (ndarray): N by n, x_{t+1}
        Return:
            log_weights (ndarray): N, importance weights

        """
        N = np.shape(x_next)[0]
        # log P(y_t+1 | x_t+1)
        scaled_y2 = (self.parameters.Ltau2inv*self.y_next)**2
        log_weights = \
                -0.5*np.log(2.0*np.pi) + \
                -0.5*scaled_y2*np.exp(-x_next) + \
                np.log(self.parameters.Ltau2inv) + \
                -0.5*x_next

        # log P(x_t+1 | x_t)
        x_diff = x_next - self.parameters.phi*x_t
        sigma2_nojump = self.parameters.sigma2
        sigma2_jump = self.parameters.sigma2 + self.parameters.sigmaJ2
        pJ = self.parameters.pJ
        loglike_nojump = -0.5*(x_diff**2)/sigma2_nojump+\
                -0.5*np.log(2.0*np.pi) - 0.5*np.log(sigma2_nojump)
        loglike_jump = -0.5*(x_diff**2)/sigma2_jump+\
                -0.5*np.log(2.0*np.pi) - 0.5*np.log(sigma2_jump)
        loglike_max = np.max(np.array([loglike_jump, loglike_nojump]), axis=0)
        loglike = np.log(
                pJ*np.exp(loglike_jump-loglike_max) +
                (1-pJ)*np.exp(loglike_nojump-loglike_max)
                ) + loglike_max

        # log Q(x_t+1 | x_t)
        logker_nojump = \
            -0.5*((x_next-self.approx_param['mean'])**2)/self.approx_param['var'] + \
            -0.5*np.log(2.0*np.pi) - 0.5*np.log(self.approx_param['var'])
        logker_jump = \
            -0.5*((x_next-self.approx_param['mean_J'])**2)/self.approx_param['var_J'] + \
            -0.5*np.log(2.0*np.pi) - 0.5*np.log(self.approx_param['var_J'])
        logker_max = np.max(np.array([logker_jump, logker_nojump]), axis=0)
        logker = np.log(
                self.approx_param['pJ']*np.exp(logker_jump-logker_max) +
                (1-self.approx_param['pJ'])*np.exp(logker_nojump-logker_max)
                ) + logker_max

        log_weights = np.reshape(log_weights+loglike-logker, (N))
        return log_weights




