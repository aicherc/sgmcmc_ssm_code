"""

Metric Functions for full parameter traces
(e.g. Kernel Stein Divergence)

"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
import logging

LOGGING_FORMAT = '%(levelname)s: %(asctime)s - %(name)s: %(message)s ...'
logging.basicConfig(
        level = logging.INFO,
        format = LOGGING_FORMAT,
        )
logger = logging.getLogger(name=__name__)

def IMQ_KSD(x, gradlogp, c=1, beta=0.5, max_block_size=1000, tqdm_out=None):
    """ Uses the Inverse MultiQuadratic Kernel Stein Discrepancy (IMQ KSD)
        IMQ(x,y) = (c^2 + (x-y)^T(x-y))^-beta (for c in \R, beta in (0,1))
        See SteinDiscrepancy.jl (on GitHub) for full details

    Args:
        x (ndarray): num_points by d
        gradlogp (ndarray): num_points by d
        c (double): parameter of IMQ
        beta (double): parameter of IMQ

    Returns:
        IMQ_KSD (double)
    """
    c2 = c**2
    if x.shape != gradlogp.shape:
        raise ValueError("x and gradlogp dimensions do not match")

    IMQ_KSD_sum = 0

    if np.shape(x)[0] <= max_block_size:
        blocks = [np.arange(np.shape(x)[0], dtype=int)]
    else:
        chunks = int(np.ceil(np.shape(x)[0]*1.0/max_block_size))
        blocks = [np.arange(max_block_size, dtype=int) + max_block_size*chunk
                  for chunk in range(chunks)]
        blocks[-1] = blocks[-1][blocks[-1] < np.shape(x)[0]]

    block_pairs = list(product(blocks, blocks))
    if len(block_pairs) == 1:
        p_bar = block_pairs
    else:
        p_bar = tqdm(block_pairs, file=tqdm_out, mininterval=60)

    for block0, block1 in p_bar:
        index0_, index1_ = np.meshgrid(block0, block1)
        index0 = index0_.flatten()
        index1 = index1_.flatten()

        x0 = x[index0]
        x1 = x[index1]
        gradlogp0 = gradlogp[index0]
        gradlogp1 = gradlogp[index1]
        dim_x = np.shape(x)[1]

        diff = x0-x1
        diff2 = np.sum(diff**2, axis=1)

        # Calculate KSD
        base = diff2 + c2
        base_beta = base**-beta
        base_beta1 = base_beta/base

        kterm_sum = np.sum(np.sum(gradlogp0*gradlogp1, axis=1) * base_beta)
        coeffgrad = -2.0 * beta * base_beta1
        gradx0term_sum = np.sum(np.sum(gradlogp0*-diff, axis=1) * coeffgrad)
        gradx1term_sum = np.sum(np.sum(gradlogp1*diff, axis=1) * coeffgrad)
        gradx0x1term_sum = np.sum((-dim_x + 2*(beta+1)*diff2/base) * coeffgrad)
        IMQ_KSD_sum += kterm_sum + gradx0term_sum + gradx1term_sum + gradx0x1term_sum

    IMQ_KSD = np.sqrt(IMQ_KSD_sum)/np.shape(x)[0]
    return IMQ_KSD

def compute_KSD(param_list, grad_list, variables=None, **kwargs):
    """ kwargs are passed to IMQ_KDS
    Args:
        param_list (list of Parameters): list of parameters used
        grad_list (list of np.ndarray): must be same size as `variables`
        variables (list of string): variables of Parameters to calculate KSD
    """
    res = {}
    if variables is not None:
        logger.info("Processing parameters_list for variables {0}".format(
            variables))
        logger.info("Processing grad_list for variables {0}".format(
            variables))

        for ii, var in enumerate(variables):
            if hasattr(param_list[0], var):
                x = np.array([getattr(parameters, var).flatten()
                    for parameters in param_list])
                gradlogp = np.array([grad[ii] for grad in grad_list])
                if len(x.shape) == 1:
                    x = np.reshape(x, (-1, 1))
                if len(gradlogp.shape) == 1:
                    gradlogp = np.reshape(gradlogp, (-1, 1))

                logger.info("Calculating IMQ_KSD for {0}".format(var))
                res[var] = IMQ_KSD(x, gradlogp, **kwargs)
            else:
                logger.warning("Did not find {0} in parameters".format(var))

    return res





