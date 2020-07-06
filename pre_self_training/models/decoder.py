import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class FirstBlock(nn.Module):
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

class TransUp(nn.Module):
    def __init__(self, i, o):
        """
        Args:
        - i (int): input channels
        - o (int): output channels
        """
        super().__init__()
        # Layers
        self.bn = nn.BatchNorm2d(i)
        self.conv = nn.ConvTranspose2d(i, o, 4, 2, 1)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        x = F.relu(x)
        return x

class Model(nn.Module):
    def __init__(self, args):
        """
        Args (dictionary):
        - data_size (int): spatial length of input signal (assuming H=W)
        - data_channels (int): channels of input signal
        - target_size (int): spatial length of output signal (assuming H=W)
        - target_channels (int): channels of output signal
        - layers_base (int): see below
        - channels_base (int): starting number of channels
        - max_channels (int): max number of channels per layer
        After every group of layers_base layers, an upsampling block is added,
        as long as the spatial size is smaller than or equal to max_spatial_size.
        """
        super().__init__()
        # Store args
        self.data_size = args.get('data_size', 32)
        self.data_channels = args.get('data_channels', 3)
        self.target_size = args.get('target_size', 32)
        self.target_channels = args.get('target_channels', 3)
        self.layers_base = args.get('layers_base', 1)
        self.channels_base = args.get('channels_base', 512)
        self.min_channels = args.get('min_channels') or 64 # TODO fix other args
        # Feature extraction layers
        self.features = nn.ModuleList()
        curr_data_size = self.data_size
        curr_channels = self.channels_base
        # Add first block
        self.features.append(FirstBlock(self.data_channels, curr_channels, 3, 1))
        # Add blocks
        while curr_data_size < self.target_size:
            # Add blocks
            for _ in range(self.layers_base):
                # Add block
                self.features.append(Block(curr_channels, curr_channels, 3, 1))
            # Add upsampling block
            self.features.append(TransUp(curr_channels, max(curr_channels//2, self.min_channels)))
            # Update values
            curr_channels = max(curr_channels//2, self.min_channels)
            curr_data_size = min(curr_data_size*2, self.target_size)
        # Add final block
        self.features.append(ChangeChannels(curr_channels, self.target_channels, act=False))
        # Create sequential containers
        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        x = torch.tanh(self.features(x))
        return x
