__copyright__ = "Copyright 2020, UPB-CAMPUS Research Center - Multimedia Lab"
__email__ = "liviu_daniel.stefan@upb.ro, cmihaigabriel@gmail.com"

from nnframework.base import BaseModel
from torchvision import models
import torch.nn as nn

class DenseNetModel(BaseModel):

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

        if self.variant == 121:
            self.model = models.densenet121(pretrained=True)
            num_filters = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_filters, self.num_classes)
            self.input_size = 224

        if self.variant == 161:
            self.model = models.densenet161(pretrained=True)
            num_filters = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_filters, self.num_classes)
            self.input_size = 224

        if self.variant == 169:
            self.model = models.densenet169(pretrained=True)
            num_filters = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_filters, self.num_classes)
            self.input_size = 224

        if self.variant == 201:
            self.model = models.densenet201(pretrained=True)
            num_filters = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_filters, self.num_classes)
            self.input_size = 224
