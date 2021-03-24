__copyright__ = "Copyright 2020, UPB-CAMPUS Research Center - Multimedia Lab"
__email__ = "liviu_daniel.stefan@upb.ro, cmihaigabriel@gmail.com"

from nnframework.base import BaseModel
from torchvision import models
import torch.nn as nn

class InceptionModel(BaseModel):

    def get_model(self):
        return self.model

    def forward(self, *inputs):
        pass

    def __init__(self, variant, num_classes, print_model=False):
        """
        variant -> defines the ResNet variant used for this experiment (18, 50, etc)
        num_classes -> the number of classes in the experiment
        """
        super().__init__()
        self.variant = variant
        self.num_classes = num_classes

        if self.variant == 3:
            self.model = models.inception_v3(pretrained=True)
            num_filters = self.model.fc.in_features
            self.model.aux_logits = False
            self.model.fc = nn.Linear(num_filters, self.num_classes)
            self.input_size = 299
