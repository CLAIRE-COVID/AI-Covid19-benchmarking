__copyright__ = "Copyright 2020, UPB-CAMPUS Research Center - Multimedia Lab"
__email__ = "liviu_daniel.stefan@upb.ro, cmihaigabriel@gmail.com"

import torch.nn.functional as F


def cross_entropy_loss(output, target):
    return F.cross_entropy(output, target)
