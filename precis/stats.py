
import numpy as np


def cronbach_alpha(scores, key):
    ''' Compute Cronbach's alpha for one or more scales.
    Args:
        scores (ndarray): An n_items x n_scales numpy 2D array
        key (ndarray): A scoring matrix indicating the mapping from items (in
            rows) to scales (in columns). 1 = positively keyed, -1 = reverse
            keyed, 0 = unused for that scale.
    Returns:
        An n_scales x 1 numpy array of Cronbach's alpha values.
    '''
    n_scales = key.shape[1]
    alpha = [0.0] * n_scales
    for i in range(n_scales):
        inds = np.where(key[:, i] != 0)[0]
        signs = key[inds, i]
        X = np.dot(scores[:, inds], np.diag(signs))
        k = len(signs)
        total = np.sum(X, 1)
        tot_var = np.var(total)
        item_var = np.sum(np.var(X, 0))
        a = (float(k) / (k - 1)) * (1 - np.divide(item_var, tot_var))
        alpha[i] = a
    return np.round(alpha, 3)
