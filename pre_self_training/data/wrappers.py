import torch
from torch.utils.data import Dataset
from kornia.geometry.transform import rotate
import random

class RotationDataset(Dataset):

    # Constructor
    def __init__(self, dataset, train):
        '''
        Args:
        - dataset (Dataset): dataset object (extends Dataset).
          Dataset items should be tuples where the first element is a CxHxW image
        - train (bool): if train is False, apply rotations deterministically
        '''
        # Store params
        self.dataset = dataset
        self.train = train
        # Define angles
        self.angles = [0, 90, 180, 270]
        # Define classes
        self.num_classes = len(self.angles)

    # Dataset size
    def __len__(self):
        return len(self.dataset)

    # Items
    def __getitem__(self, idx):
        # Select angle/label
        if self.train:
            label = random.randint(0, len(self.angles)-1)
        else:
            label = idx % len(self.angles)
        angle = self.angles[label]
        # Get image
        img = self.dataset[idx][0]
        # Apply rotation
        if angle != 0:
            # Add batch dimension
            img = img.unsqueeze(0)
            # Rotate
            img = rotate(img, angle=torch.tensor(angle))
            # Remove batch dimension
            img = img.squeeze(0)
        # Return
        return img, label

        
