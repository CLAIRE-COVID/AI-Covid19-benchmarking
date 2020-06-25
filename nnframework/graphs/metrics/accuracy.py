__copyright__ = "Copyright 2020, UPB-CAMPUS Research Center - Multimedia Lab"
__email__ = "liviu_daniel.stefan@upb.ro, cmihaigabriel@gmail.com"

import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)
