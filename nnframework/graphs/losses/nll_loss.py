__copyright__ = "Copyright 2020, UPB-CAMPUS Research Center - Multimedia Lab"
__email__ = "liviu_daniel.stefan@upb.ro, cmihaigabriel@gmail.com"

import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

