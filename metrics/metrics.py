import numpy as np
from sklearn.metrics import (accuracy_score,
                             f1_score,
                             log_loss,
                             precision_score,
                             recall_score,
                             roc_auc_score,
                             confusion_matrix)
from prg import prg

# FIXME The current code assumes there is at least one positive and one negative
# sample in y_true. If this is not the case, the functions need to have an
# assert.

def odds_ratio(y_true, scores, threshold=0.5):
    '''
    With Haldane-Anscombe correction

    y_true: binary vector
        0: negative class
        1: positive class
    returns:
        (true pos. * true neg.) / (false pos. * false neg.)

    # TODO Clarify what do do with denominator equal zero (eg. 0/0, x/0)
    Examples
    --------
    >>> from prg import prg
    >>> import numpy as np
    >>> y_true = np.array([1, 1, 1, 1, 0, 1, 0, 1, 0, 0], dtype='int')
    >>> scores = np.array([1, 1, 1, 1, 0, 1, 0, 1, 0, 0], dtype='int')
    >>> odds_ratio(y_true, scores)
    117.0
    >>> y_true = np.array([1, 1, 0, 0], dtype='int')
    >>> scores = np.array([1, 1, 1, 1], dtype='int')
    >>> odds_ratio(y_true, scores)
    1.0
    >>> y_true = np.array([1, 1, 0, 0], dtype='int')
    >>> scores = np.array([0, 0, 0, 0], dtype='int')
    >>> odds_ratio(y_true, scores)
    1.0
    >>> y_true = np.array([1, 1, 0, 0], dtype='int')
    >>> scores = np.array([0, 0, 1, 0], dtype='int')
    >>> odds_ratio(y_true, scores)
    0.2
    >>> y_true = np.array([1, 1, 0, 0], dtype='int')
    >>> scores = np.array([0, 0, 1, 1], dtype='int')
    >>> odds_ratio(y_true, scores)
    0.04
    '''
    y_pred = scores > threshold
    cm = confusion_matrix(y_true, y_pred)
    cm = cm + 0.5
    numerator = cm[0,0]*cm[1,1]
    denominator = cm[0,1]*cm[1,0]
    return numerator/denominator

def specificity(y_true, scores, threshold=0.5):
    '''
    y_true: binary vector
        0: negative class
        1: positive class
    returns:
        true negatives divided by all negatives

    Examples
    --------
    >>> from prg import prg
    >>> import numpy as np
    >>> y_true = np.array([1, 1, 1, 1, 0, 1, 0, 1, 0, 0], dtype='int')
    >>> scores = np.array([1, 1, 1, 1, 0, 1, 0, 1, 0, 0], dtype='int')
    >>> specificity(y_true, scores)
    1.0
    >>> y_true = np.array([1, 1, 0, 0], dtype='int')
    >>> scores = np.array([1, 1, 1, 1], dtype='int')
    >>> specificity(y_true, scores)
    0.0
    >>> y_true = np.array([1, 1, 0, 0], dtype='int')
    >>> scores = np.array([0, 0, 0, 0], dtype='int')
    >>> specificity(y_true, scores)
    1.0
    >>> y_true = np.array([1, 1, 0, 0], dtype='int')
    >>> scores = np.array([0, 0, 1, 0], dtype='int')
    >>> specificity(y_true, scores)
    0.5
    >>> y_true = np.array([1, 1, 0, 0], dtype='int')
    >>> scores = np.array([0, 0, 1, 1], dtype='int')
    >>> specificity(y_true, scores)
    0.0
    '''
    y_pred = scores > threshold
    cm = confusion_matrix(y_true, y_pred)
    return cm[0,0]/(cm[0,0] + cm[0,1])

def sensitivity(y_true, scores, threshold=0.5):
    '''
    y_true: binary vector
        0: negative class
        1: positive class
    returns:
        true positives divided by all positives

    Examples
    --------
    >>> from prg import prg
    >>> import numpy as np
    >>> y_true = np.array([1, 1, 1, 1, 0, 1, 0, 1, 0, 0], dtype='int')
    >>> scores = np.array([1, 1, 1, 1, 0, 1, 0, 1, 0, 0], dtype='int')
    >>> sensitivity(y_true, scores)
    1.0
    >>> y_true = np.array([1, 1, 0, 0], dtype='int')
    >>> scores = np.array([1, 1, 1, 1], dtype='int')
    >>> sensitivity(y_true, scores)
    1.0
    >>> y_true = np.array([1, 1, 0, 0], dtype='int')
    >>> scores = np.array([0, 0, 1, 1], dtype='int')
    >>> sensitivity(y_true, scores)
    0.0
    '''
    y_pred = scores > threshold
    cm = confusion_matrix(y_true, y_pred)
    return cm[1,1]/(cm[1,0] + cm[1,1])

def accuracy(y_true, scores, threshold=0.5):
    '''
    Examples
    --------
    >>> from prg import prg
    >>> import numpy as np
    >>> y_true = np.array([1, 1, 1, 1, 0, 1, 0, 1, 0, 0], dtype='int')
    >>> scores = np.array([1, 1, 1, 1, 0, 1, 0, 1, 0, 0], dtype='int')
    >>> accuracy(y_true, scores)
    1.0
    >>> y_true = np.array([1, 1, 0, 0], dtype='int')
    >>> scores = np.array([1, 1, 1, 1], dtype='int')
    >>> accuracy(y_true, scores)
    0.5
    >>> y_true = np.array([1, 1, 0, 0], dtype='int')
    >>> scores = np.array([0, 0, 1, 1], dtype='int')
    >>> accuracy(y_true, scores)
    0.0
    '''
    y_pred = scores > threshold
    return accuracy_score(y_true, y_pred)

def f1(y_true, scores, threshold=0.5):
    y_pred = scores > threshold
    return f1_score(y_true, y_pred)

def precision(y_true, scores, threshold=0.5):
    y_pred = scores > threshold
    return precision_score(y_true, y_pred)

def recall(y_true, scores, threshold=0.5):
    y_pred = scores > threshold
    return recall_score(y_true, y_pred)

def odds_ratio_score(y_true, scores):
    """Compute the Area Under the Precision Recall Gain Curve (AUPRG)

    Note: this implementation is restricted to the binary classification task.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True binary labels.

    scores : array, shape = [n_samples]
        Estimated probabilities or decision function.
    """
    return 3

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
