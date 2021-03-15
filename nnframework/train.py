__copyright__ = "Copyright 2020, UPB-CAMPUS Research Center - Multimedia Lab"
__email__ = "liviu_daniel.stefan@upb.ro, cmihaigabriel@gmail.com"

import argparse
import collections
import torch
import torch.hub
import numpy as np
from parse_config import ConfigParser
import data_loader as module_dataloader
import graphs.models as module_model
import graphs.losses as module_loss
import graphs.metrics as module_metric
from trainers import Trainer


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
from torch import nn

def main(config):
    logger = config.get_logger('train')

    data_loader = config.init_obj('data_loader', module_dataloader)

    torch.hub.set_dir(config['weights_path'])
    model = config.init_obj('arch', module_model)
    logger.info(model)

    #FIXME: loss pesata, da cambiare per renderlo modulare. Cambiare device!
    criterion = torch.nn.CrossEntropyLoss(weight= data_loader.get_label_proportions().to('cuda'))
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model.get_model(), criterion, metrics, optimizer,
                      config=config,
                      train_data_loader=data_loader.train,
                      valid_data_loader=data_loader.val,
                      test_data_loader=data_loader.test,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
