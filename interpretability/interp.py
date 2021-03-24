import torch
import torch.nn.functional as F

from PIL import Image
from nnframework.graphs.models import DenseNetModel
import os
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import torchvision
from torchvision import models
from torchvision import transforms
from captum.attr import IntegratedGradients, NoiseTunnel, GuidedGradCam, DeepLift
from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import GuidedGradCam
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from parse_config import ConfigParser
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import numpy as np
import sys 
from nnframework.data_loader import COVID_Dataset
from tqdm import tqdm


def show_attributions(attr, img,predicted_label,true_label, show_map=False, save=None):
    # Convert to numpy
    attr = np.transpose(attr.squeeze(0).cpu().detach().numpy(), (1,2,0))
    img = np.transpose(img.squeeze(0).cpu().detach().numpy(), (1,2,0))
    # What to show
    if show_map:
        methods=["original_image", "heat_map", "blended_heat_map"]
        signs=["all", "positive", "positive"]
    else:
        methods=["original_image", "blended_heat_map"]
        signs=["all", "positive"]
    # Show
    fig,axis = viz.visualize_image_attr_multiple(attr, img, methods, signs, cmap=cm.seismic, show_colorbar=True,
                                                 outlier_perc=1, use_pyplot=(save is None))

    if predicted_label ==0:
        predicted_text = 'Negative'
    else:
        predicted_text = 'Positive'

    if true_label ==0:
        true_text = 'Negative'
    else:
        true_text = 'Positive'
    fig.suptitle('True Label: {} Predicted: {}'.format(true_text,predicted_text), fontsize=20)
    # Check save
    if save is not None:
        fig.savefig(save)


covid_loaders = COVID_Dataset('data/final3_masked',pos_neg_file='data/labels_covid19_posi.tsv', splits= [0.7, 0.15, 0.15],
replicate_channel= 1,
      batch_size= 1,
      random_seed=123,
      input_size= 224,
      mode = 'ct',
      num_workers= 0)

model = DenseNetModel(121,2)
model.model.load_state_dict(torch.load('interpretability/model_best.pth')['state_dict'])
model = model.eval()
model.model = model.model.to('cuda')
gc = GuidedGradCam(model.model, model.model.features.denseblock3)
noise_tunnel = NoiseTunnel(gc)

for (img, label, sublabel, subject_id, ct_id, slice_id) in tqdm(covid_loaders.test):
    img = img.to('cuda')
    output = model.model(img)
    prediction_score, predicted_label = torch.max(output, 1)
    attributions_ig_nt = noise_tunnel.attribute(img, nt_samples=10, nt_type='smoothgrad_sq', target=predicted_label,stdevs=0.01)
    filename = subject_id[0] + '_' + ct_id[0] + '_'+ str(slice_id.item())+'.png'
    dest = os.path.join('interp_output',subject_id[0],ct_id[0],filename)
    if not os.path.exists(os.path.dirname(dest)):
        os.makedirs(os.path.dirname(dest))
    show_attributions(attributions_ig_nt, img,predicted_label.item(),label.item(), show_map=True,save=dest)

