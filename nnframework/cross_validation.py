import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--results_root', type=str, default='saved/results')
    parser.add_argument('--dest_folder', type=str, default='saved/cross_val')
    args = parser.parse_args()
    print(os.getcwd())
    results_folders = glob.glob(args.results_root + '/*')
    results_folders = [folder.split(os.sep)[-1] for folder in results_folders]
    # filter only kfold experiments
    results_folders = [folder for folder in results_folders if 'fold' in folder]
    print(results_folders)
    results_folders = [folder for folder in results_folders if args.experiment_name in folder]

    assert (len(results_folders) > 0), "Experiment not found"
    assert (len(
        results_folders) == args.k_folds), "Nunber of experiments in folder ({}) did not match number of folds ({})".format(
        len(results_folders), args.k_folds)
    folds_names = ['fold{}'.format(num) for num in range(args.k_folds)]
    assert all([any([fold in folder for folder in results_folders]) for fold in
                folds_names]), "One ore more folds are missing or are repeated"

    folds_paths = [os.path.join(args.results_root, fld) for fld in results_folders]
    print(folds_paths)
    slice_metrics = {}
    for fold_path in folds_paths:
        with open(os.path.join(fold_path, 'metrics_per_slice.csv'), 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|',
                                quoting=csv.QUOTE_NONNUMERIC)
            next(reader, None)  # skip header
            for row in reader:
                k = row[0]
                v = row[2]
                if k not in slice_metrics.keys():
                    slice_metrics[k] = [v]
                else:
                    slice_metrics[k].append(v)

    mean_slice_metrics = {}
    for k, v in slice_metrics.items():
        avg_v = sum(v) / len(v)
        mean_slice_metrics[k] = avg_v
    print(mean_slice_metrics)

    metrics_per_scan_names = ["metrics_per_scan_proportion_{0:03}.csv".format(i) for i in range(102)]
    print(metrics_per_scan_names)

    res_scan_propritions = {k: {} for k in metrics_per_scan_names}
    for fold_path in folds_paths:
        for metrics_scan_name in metrics_per_scan_names:
            with open(os.path.join(fold_path, metrics_scan_name), 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='|',
                                    quoting=csv.QUOTE_NONNUMERIC)
                next(reader, None)  # skip header
                for row in reader:
                    k = row[0]
                    v = row[2]
                    if k not in res_scan_propritions[metrics_scan_name].keys():
                        res_scan_propritions[metrics_scan_name][k] = [v]
                    else:
                        res_scan_propritions[metrics_scan_name][k].append(v)

    mean_res_scan_propritions = {k: {} for k in metrics_per_scan_names}
    for metrics_scan_name in metrics_per_scan_names:
        for k, v in res_scan_propritions[metrics_scan_name].items():
            avg_v = sum(v) / len(v)
            mean_res_scan_propritions[metrics_scan_name][k] = avg_v

    for k, v in mean_res_scan_propritions.items():
        cross_val_dest = os.path.join(args.dest_folder, args.experiment_name, k)
        if not os.path.exists(os.path.dirname(cross_val_dest)):
            os.makedirs(os.path.dirname(cross_val_dest))
        with open(os.path.join(cross_val_dest), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|',
                                quoting=csv.QUOTE_NONNUMERIC)
            for key, value in mean_res_scan_propritions[k].items():
                writer.writerow([key, value])

    slice_metrics_dest = os.path.join(args.dest_folder, args.experiment_name, 'metrics_per_slice.csv')
    with open(os.path.join(slice_metrics_dest), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|',
                            quoting=csv.QUOTE_NONNUMERIC)
        for key, value in mean_slice_metrics.items():
            writer.writerow([key, value])

    # roc curve
    sens = []
    spec = []
    for k in metrics_per_scan_names:
        sens.append(mean_res_scan_propritions[k]['sensitivity'])
        spec.append(mean_res_scan_propritions[k]['specificity'])
    print(sens)
    print(spec)
    sens = np.array(sens)
    spec = np.array(spec)
    plt.title('ROC curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot(1 - spec, sens)
    plt.savefig(os.path.join(args.dest_folder, args.experiment_name, 'roc.png'))


if __name__ == '__main__':
    main()
