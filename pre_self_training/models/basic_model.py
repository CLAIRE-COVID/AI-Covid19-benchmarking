import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class FirstBlock(nn.Module):
    def __init__(self, i, o, k, s, d):
        """
        Args:
        - i (int): input channels
        - o (int): output channels
        - k (int): kernel size
        - s (int): stride
        - d (int): dilation
        """
        super().__init__()
        # Compute padding
        p = (k//2)*d
        # Layers
        self.conv = nn.Conv2d(i, o, k, s, p, d)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        return x

class Block(nn.Module):
    def __init__(self, i, o, k, s=1, d=1):
        """
        Args:
        - i (int): input channels
        - o (int): output channels
        - k (int): kernel size
        - s (int): stride
        - d (int): dilation
        """
        super().__init__()
        # Compute padding
        p = (k//2)*d
        # Layers
        self.conv1 = nn.Conv2d(i, o, k, s, p, d)
        self.bn1 = nn.BatchNorm2d(o)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(o, o, k, s, p, d)
        self.bn2 = nn.BatchNorm2d(o)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ChangeChannels(nn.Module):
    def __init__(self, i, o, act=True):
        """
        Args:
        - i (int): input channels
        - o (int): output channels
        """
        super().__init__()
        # Layers
        self.bn = nn.BatchNorm2d(i)
        self.conv = nn.Conv2d(i, o, 1)
        self.act = act

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        if self.act:
            x = F.relu(x)
        return x

class TransDown(nn.Module):
    def __init__(self, i, o):
        """
        Args:
        - i (int): input channels
        - o (int): output channels
        """
        super().__init__()
        # Layers
        self.bn = nn.BatchNorm2d(i)
        self.conv = nn.Conv2d(i, o, 3, 2, 1)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        x = F.relu(x)
        return x

class FC(nn.Module):
    def __init__(self, i, o, dropout=True, act=True):
        """
        Args:
        - i (int): input channels
        - o (int): output channels
        """
        super().__init__()
        # Layers
        self.fc = nn.Linear(i, o)
        self.dropout = nn.Dropout(0.5) if act and dropout else None
        self.act = act

    def forward(self, x):
        x = self.fc(x)
        if self.act:
            x = F.relu(x)
            if self.dropout is not None:
                x = self.dropout(x)
        return x

class Model(nn.Module):
    def __init__(self, args):
        """
        Args (dictionary):
        - data_size (int): spatial length of input signal (assuming H=W)
        - data_channels (int): channels of input signal
        - layers_base (int): see below
        - channels_base (int): starting number of channels
        - min_spatial_size (int): minimum spatial size to keep
        - start_dilation (int): initial dilation value
        - min_dil_ratio (int): min ratio between data size and dilation
        - max_fc_ratio (int): max ratio between consecutive FC layer sizes
        - max_channels (int): max number of channels per layer
        - fc_layers (int): number of fully-connected layers
        - reduce_fc (bool): keep FC features or reduce them gradually
        - fc_dropout (bool): use dropout in FC
        - num_classes (int): number of output classes
        After every group of layers_base layers, a downsampling block is added,
        as long as the spatial size is greater than or equal to  min_spatial_size.
        """
        super().__init__()
        # Store args
        self.data_size = args.get('data_size', 32)
        self.data_channels = args.get('data_channels', 3)
        self.return_features = args.get('return_features') or False
        self.layers_base = args.get('layers_base', 1)
        self.channels_base = args.get('channels_base', 64)
        self.min_spatial_size = args.get('min_spatial_size', 4)
        self.start_dilation = args.get('start_dilation', 1)
        self.min_dil_ratio = args.get('min_dil_ratio', 50)
        self.max_channels = args.get('max_channels', 256)
        self.fc_layers = args.get('fc_layers', 1)
        self.reduce_fc = args.get('reduce_fc', False)
        self.fc_dropout = args.get('fc_dropout', False)
        self.num_classes = args.get('num_classes', 10)
        # Feature extraction layers
        self.features = nn.ModuleList()
        curr_data_size = self.data_size
        curr_channels = self.channels_base
        curr_dilation = self.start_dilation
        # Add encoder first block
        self.features.append(FirstBlock(self.data_channels, curr_channels, 3, 1, curr_dilation))
        # Add blocks
        while curr_data_size//2 > self.min_spatial_size:
            # Add blocks
            for _ in range(self.layers_base):
                # Add block
                self.features.append(Block(curr_channels, curr_channels, 3, 1, curr_dilation))
            # Add downsampling block
            self.features.append(TransDown(curr_channels, min(curr_channels*2, self.max_channels)))
            # Update values
            if curr_channels < self.max_channels:
                curr_channels *= 2
            curr_data_size //= 2
            while curr_dilation > 1 and curr_data_size/curr_dilation < self.min_dil_ratio:
                curr_dilation -= 1
        # Create sequential container
        self.features = nn.Sequential(*self.features)
        if self.return_features:
            return
        # Adaptive average pooling
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        # Compute size of fully-connected layers
        if not self.reduce_fc:
            fc_size = [curr_channels]*(self.fc_layers-1) + [self.num_classes]
        else:
            fc_size = torch.linspace(curr_channels, self.num_classes, self.fc_layers + 1).int()[1:].tolist()
        # Fully-connected layers
        self.fc = nn.ModuleList()
        for i,size in enumerate(fc_size):
            # Add block
            self.fc.append(
                FC(curr_channels, size, dropout=self.fc_dropout,
                   act=(i != len(fc_size)-1)
                )
            )
            # Update current channels
            curr_channels = size
        # Create sequential container
        self.fc = nn.Sequential(*self.fc)

    def forward(self, x):
        x = self.features(x)
        if self.return_features:
            return x
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
