from sklearn.metrics import (accuracy_score,
                             f1_score,
                             log_loss,
                             precision_score,
                             recall_score)

from prg import prg

def auprgc_score(y_true, scores):
    """Compute the Area Under the Precision Recall Gain Curve (AUPRG)

    Note: this implementation is restricted to the binary classification task.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True binary labels.

    scores : array, shape = [n_samples]
        Estimated probabilities or decision function.

    Examples
    --------
    >>> from prg import prg
    >>> import numpy as np
    >>> y_true = np.array([1, 1, 1, 1, 0, 1, 0, 1, 0, 0], dtype='int')
    >>> scores = np.arange(10, 1, -1)
    >>> auprgc_score(y_true, scores)
    0.683125
    >>> y_true = np.array([1, 1, 0, 0], dtype='int')
    >>> scores = np.arange(4, 1, -1)
    >>> auprgc_score(y_true, scores)
    1.0
    >>> y_true = np.array([0, 0, 1, 1], dtype='int')
    >>> scores = np.arange(4, 1, -1)
    >>> auprgc_score(y_true, scores)
    0
    >>> y_true = np.array([0, 1, 0], dtype='int')
    >>> scores = np.arange(3, 1, -1)
    >>> auprgc_score(y_true, scores)
    0.0
    """
    prg_curve = prg.create_prg_curve(y_true, scores)
    auprg = prg.calc_auprg(prg_curve)
    return auprg



if __name__ == "__main__":
    import doctest
    doctest.testmod()
