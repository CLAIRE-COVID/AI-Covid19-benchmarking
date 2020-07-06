import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init

def add_net_to_params(net: nn.Module, args: dict, key: str):
    args.__dict__[key] = str(net)
    args.__dict__[f'{key}_parameters'] = np.sum([p.numel() for p in net.parameters()])

def onehot_map(x, levels):
    """
    Args:
        x (tensor: Bx1xHxW): input tensor
        levels (int): quantization levels
    """
    # Quantize
    x = (x*levels).long()
    x[x == levels] = levels-1
    # Store original size
    orig_size = x.size()
    # View
    x = x.view(-1, 1)
    # Initialize onehot
    x_onehot = torch.zeros(x.size(0), levels).to(x.device)
    # Fill onehot
    x_onehot.scatter_(1, x, 1)
    # Reshape
    x_onehot = x_onehot.view(orig_size[0], orig_size[2], orig_size[3], levels)
    x_onehot = x_onehot.permute(0, 3, 1, 2)
    return x_onehot