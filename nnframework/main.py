#!/usr/bin/env python

import argparse
import os
import sys

import train
from parse_config import ConfigParser

# Configure argument parser
parser = argparse.ArgumentParser(description='Training Framework CLI')
parser.add_argument('--dataset', type=str, help='Path to the dataset main directory')
parser.add_argument('--epochs', type=int, help='Number of training epochs')
parser.add_argument('--gpus', default=1, type=int, help='Number of gpus to use')
parser.add_argument('--k-fold-idx', type=int, help='Index of the cross validation fold')
parser.add_argument('--labels', type=str, help='Path to the labels file')
parser.add_argument('--learning-rate', type=float, help='Learning rate for the training algorithm')
parser.add_argument('--lr-step-size', type=int, help='Step size for learning rate decay')
parser.add_argument('--model-type', type=str, help='Type of model to train (e.g. DenseNetModel or VGGModel)')
parser.add_argument('--model-layers', type=int, help='Number of layers in the model')
parser.add_argument('--name', default='covid-training', type=str, help='Name of the training task')
parser.add_argument('--outdir', default=os.getcwd(), type=str, help='Output directory in which to store training results') 
parser.add_argument('--weight-decay', type=float, help='Weight decay for the training algorithm')
parser.add_argument('--weights-path', type=str, help='Directory where pretrained models weights are stored')


def main(args):
    args = parser.parse_args(args)

    config = {
        'name': args.name,
        'n_gpu': args.gpus,
        'weights_path': args.weights_path,
        'arch': {
            'type': args.model_type,
            'args': {
                'variant': args.model_layers,
                'num_classes': 2,
                'print_model': True
            }
        },
        'loss': 'cross_entropy_loss',
        'metrics': [
            'accuracy'
        ],
        'data_loader': {
            'type': 'COVID_Dataset',
            'args': {
                'root': args.dataset,
                'k_fold_idx': args.k_fold_idx,
                'mode': 'ct',
                'pos_neg_file': args.labels,
                'splits': [0.7, 0.15, 0.15],
                'replicate_channel': 1,
                'batch_size': 64,
                'input_size': 224,
                'num_workers': 2,
                'self_supervised': 0
            }
        },
        'optimizer': {
            'type': 'Adam',
            'args': {
                'lr': args.learning_rate,
                'weight_decay': args.weight_decay,
                'amsgrad': True
            }
        },
        'lr_scheduler': {
            'type': 'StepLR',
            'args': {
                'step_size': args.lr_step_size,
                'gamma': 0.1
            }
        },
        'trainer': {
            'epochs': args.epochs,
            'save_dir': args.outdir,
            'save_period': 1,
            'verbosity': 2,
            'monitor': 'min val_loss',
            'early_stop': 10,
            'tensorboard': False
        }
    }

    config = ConfigParser(config)
    train.main(config)


if __name__ == "__main__":
    main(sys.argv[1:])