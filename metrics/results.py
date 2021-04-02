import os
import shutil
import csv
from metrics.metrics import (accuracy, f1, log_loss, precision,
                     recall, auprgc_score, roc_auc_score,
                     sensitivity, specificity, odds_ratio)
import numpy as np
import pandas as pd
import inspect

COLUMN_NAMES = ['output', 'target', 'subj_id', 'ct_id', 'slice_id']

METRICS_MAP = {'accuracy': accuracy,
               'f1': f1,
               'log_loss': log_loss,
               'precision': precision,
               'recall': recall,
               'auc': roc_auc_score,
               'auprgc': auprgc_score,
               'sensitivity': sensitivity,
               'specificity': specificity,
               'odds_ratio': odds_ratio
              }

class Results(object):
    '''
    Currently only for binary problems
    # TODO Incorporate the subject, ct, and slice id into the metrics
    '''
    def __init__(self, path='results'):
        self.path = path
        self.results = []
        self.results_per_scan = {}
        self.results_per_scan_proportion = {}

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
        elif not overwrite and os.path.exists(self.path):
            raise ValueError('Can not save: The path already exists and overwrite=False')

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        with open(os.path.join(self.path, 'predictions.csv'), 'a', newline='') as csvfile:
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

    def _save_metrics(self):
        with open(os.path.join(self.path, 'metrics_per_slice.csv'), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|',
                                quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(['metric', 'threshold', 'score'])
            for metric_name, score in self.scores.items():
                writer.writerow([metric_name, 0.5, score])

        # Save any results considering each CT scan as one instance and using
        # the average probabilities and thresholds
        for threshold, results_per_scan in self.results_per_scan.items():
            filename = 'metrics_per_scan_threshold_{:03.0f}.csv'.format(
                threshold*100)
            with open(os.path.join(self.path, filename), 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', quotechar='|',
                                    quoting=csv.QUOTE_NONNUMERIC)
                writer.writerow(['metric', 'threshold', 'score'])
                for metric, score in results_per_scan.items():
                    writer.writerow([metric, threshold, score])

        # Save any results considering each CT scan as one instance and using
        # the proportions of slices with probability higher than 0.5
        for proportion, results_per_scan in self.results_per_scan_proportion.items():
            filename = 'metrics_per_scan_proportion_{:03.0f}.csv'.format(
                proportion*100)
            with open(os.path.join(self.path, filename), 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', quotechar='|',
                                    quoting=csv.QUOTE_NONNUMERIC)
                writer.writerow(['metric', 'proportion', 'score'])
                for metric, score in results_per_scan.items():
                    writer.writerow([metric, proportion, score])


    def load(self):
        if not os.path.exists(self.path):
            raise ValueError("Location {} doesn't contain results".format(self.path))

        with open(os.path.join(self.path, 'predictions.csv'), 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|',
                                quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                self.results.append(row)

    def compute_metrics_per_scan_threshold(self, threshold=0.5):
        '''
        Computes all metrics considering each ct-scan as an instance

        threshold: float
            The outputs of all slices of each scan are averaged, and the
            prediction is considered positive if the output is higher than the
            specified threshold for metrics that require a threshold. Methods
            that do not require threshold are not affected by this parameter.
        '''
        df = pd.DataFrame(self.results, columns=COLUMN_NAMES)
        df = df.pivot_table(index=['subj_id', 'ct_id'],
                            values=['output', 'target'],
                            aggfunc='mean')
        outputs = df['output']
        target = df['target']
        self.results_per_scan[threshold] = {}
        for metric_name, metric_func in METRICS_MAP.items():
            try:
                if 'threshold' in inspect.getfullargspec(metric_func)[0]:
                    score = metric_func(target, outputs, threshold=threshold)
                else:
                    score = metric_func(target, outputs)
            except ValueError as e:
                print(e)
            self.results_per_scan[threshold][metric_name] = score

    def compute_metrics_per_scan_proportion(self, proportion=0.5):
        '''
        Computes all metrics considering each ct-scan as an instance

        proportion: float
            Float: Proportion of slices that need to be positive to have the
            ct-scan predicted as positive. default corresponds to majority
            voting, given that half of the slides need to be positive, or
            negative.
        '''
        df = pd.DataFrame(self.results, columns=COLUMN_NAMES)
        df['output'] = (df['output'] >= 0.5).astype(float)
        df = df.pivot_table(index=['subj_id', 'ct_id'],
                            values=['output', 'target'],
                            aggfunc='mean')
        outputs = (df['output'] >= proportion).astype(float)
        target = df['target']

        self.results_per_scan_proportion[proportion] = {}
        for metric_name, metric_func in METRICS_MAP.items():
            try:
                score = metric_func(target, outputs)
            except ValueError as e:
                print(e)
            self.results_per_scan_proportion[proportion][metric_name] = score
