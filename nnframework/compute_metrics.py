import os

import numpy as np

from metrics.results import Results

if __name__ == '__main__':
    from tqdm import tqdm
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import sys

    res = Results(path=sys.argv[1])
    res.load()
    for p in tqdm(range(0,102)):
        res.compute_metrics_per_scan_proportion(proportion=p/100)
    res.save(overwrite=True)
    data = res.results_per_scan_proportion
    sens = []
    spec = []
    for k in sorted(data.keys()):
        sens.append(res.results_per_scan_proportion[k]['sensitivity'])
        spec.append(res.results_per_scan_proportion[k]['specificity'])
    sens = np.array(sens)
    spec = np.array(spec)
    print(sens)
    print(spec)
    plt.title('ROC curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot(1-spec, sens)
    plt.savefig(os.path.join(res.path, 'roc.png'))
