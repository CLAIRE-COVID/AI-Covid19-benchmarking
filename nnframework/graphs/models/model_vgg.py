__copyright__ = "Copyright 2020, UPB-CAMPUS Research Center - Multimedia Lab"
__email__ = "liviu_daniel.stefan@upb.ro, cmihaigabriel@gmail.com"

from base import BaseModel
from torchvision import models
import torch.nn as nn

class VGGModel(BaseModel):

    def get_model(self):
        return self.model

    def forward(self, *inputs):
        pass

    def __init__(self, variant, num_classes, print_model=False, bn=True):
        """
        variant -> defines the ResNet variant used for this experiment (18, 50, etc)
        num_classes -> the number of classes in the experiment
        """
        super().__init__()
        self.variant = variant
        self.num_classes = num_classes
        self.bn = bn

        if variant == 11:
            if self.bn:
                self.model = models.vgg11_bn(pretrained=True)
            else:
                self.model = models.vgg11(pretrained=True)
            num_filters = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_filters, self.num_classes)
            self.input_size = 224

        if variant == 13:
            if self.bn:
                self.model = models.vgg13_bn(pretrained=True)
            else:
                self.model = models.vgg13(pretrained=True)
            num_filters = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_filters, self.num_classes)
            self.input_size = 224

        if variant == 16:
            if self.bn:
                self.model = models.vgg16_bn(pretrained=True)
            else:
                self.model = models.vgg16(pretrained=True)
            num_filters = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_filters, self.num_classes)
            self.input_size = 224

        if variant == 19:
            if self.bn:
                self.model = models.vgg19_bn(pretrained=True)
            else:
                self.model = models.vgg19(pretrained=True)
            num_filters = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(num_filters, self.num_classes)
            self.input_size = 224