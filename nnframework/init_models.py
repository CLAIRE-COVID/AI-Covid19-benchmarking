import torchvision.models as models
import torch.hub
import sys
import os

# Read arg
assert len(sys.argv) > 1, "Usage: python init_models.py <model dir>"
# Set hub dir
os.makedirs(sys.argv[1], exist_ok=True)
#torch.hub.set_dir(sys.argv[1])
# Load models
print("Loading models, please wait")
model = models.densenet121(pretrained=True)
model = models.densenet161(pretrained=True)
model = models.densenet169(pretrained=True)
model = models.densenet201(pretrained=True)
model = models.inception_v3(pretrained=True)
model = models.resnet18(pretrained=True)
model = models.resnet34(pretrained=True)
model = models.resnet50(pretrained=True)
model = models.resnet101(pretrained=True)
model = models.resnet152(pretrained=True)
model = models.resnext50_32x4d(pretrained=True)
model = models.resnext101_32x8d(pretrained=True)
model = models.vgg11_bn(pretrained=True)
model = models.vgg11(pretrained=True)
model = models.vgg13_bn(pretrained=True)
model = models.vgg13(pretrained=True)
model = models.vgg16_bn(pretrained=True)
model = models.vgg16(pretrained=True)
model = models.vgg19_bn(pretrained=True)
model = models.vgg19(pretrained=True)
print("Done")
