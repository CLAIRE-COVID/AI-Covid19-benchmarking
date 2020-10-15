import os
import shutil
import csv
from metrics.metrics import (accuracy, f1, log_loss, precision,
                     recall, auprgc_score, roc_auc_score)
import numpy as np

METRICS_MAP = {'accuracy': accuracy,
               'f1': f1,
               'log_loss': log_loss,
               'precision': precision,
               'recall': recall,
               'auc': roc_auc_score,
               'auprgc': auprgc_score,
               # TODO Add 'odds_ratio': odds_ratio_score
              }

class Results(object):
    '''
    Currently only for binary problems
    # TODO Incorporate the subject, ct, and slice id into the metrics
    '''
    def __init__(self, path='results'):
        self.path = path
        self.results = []

    def add(self, output, target, subj_id, ct_id, slice_id):
        '''
        Parameters
        ==========
        output: float
            Probabilities of a sample belonging to the positive class
        target_id: integer
            Original class of the sample. 1 for positive, 0 for negative.
        subj_id: string or integer
            Identification of the subject to which the sample belongs
        ct_id: string or integer
            Identification of the CT scan
        slice_id:  string or integer
            Identification of the slice of the ct scan
        '''
        row = [output, target, subj_id, ct_id, slice_id]
        self.results.append(row)

    def save(self, overwrite=False):
        if overwrite and os.path.exists(self.path):
            shutil.rmtree(self.path)

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        with open(os.path.join(self.path, 'predictions.csv'), 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|',
                                quoting=csv.QUOTE_NONNUMERIC)
            writer.writerows(self.results)

        self._compute_metrics()
        self._save_metrics()

    def _compute_metrics(self):
        # TODO Consider the subj_id, ct_id, and slice_id
        outputs = np.array([res[0] for res in self.results], dtype=float)
        target = np.array([res[1] for res in self.results])
        self.scores = {}
        for metric_name, metric_func in METRICS_MAP.items():
            try:
                self.scores[metric_name] = metric_func(target, outputs)
            except ValueError as e:
                print(e)
                from IPython import embed; embed()

    def _save_metrics(self):
        for metric_name, score in self.scores.items():
            with open(os.path.join(self.path, '{}.csv'.format(metric_name)), 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', quotechar='|',
                                    quoting=csv.QUOTE_NONNUMERIC)
                writer.writerow([score])


if __name__ == '__main__':
    res = Results(path='results')
    res.add(.9, 1, 'John', 1, 23)
    res.add(.3, 0, 'John', 2, 23)
    res.add(.4, 1, 'John', 2, 24)
    res.save(overwrite=True)
