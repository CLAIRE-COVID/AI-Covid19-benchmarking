import argparse
from pathlib import Path
from random import shuffle
from utils.saver import Saver
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset
import torchvision.transforms as T
from data.wrappers import RotationDataset

import trainers
#from utils.misc import suppress_random

def parse():
    parser = argparse.ArgumentParser()
    # Dataset options
    #parser.add_argument('--dataset') # TODO after dataset format is decided
    parser.add_argument('--workers', type=int, default=4)
    # Experiment options
    parser.add_argument('-t', '--tag', default='default_tag')
    parser.add_argument('--logdir', default='exps', type=Path)
    parser.add_argument('--log-metrics-every', type=int, default=20, help='add metrics to Tensorboard every X iterations')
    parser.add_argument('--log-plots-every', type=int, default=500, help='add plots to Tensorboard every X iterations')
    parser.add_argument('--save-every', type=int, default=5, help='save checkpoint every X epochs')
    # Model options
    parser.add_argument('--model', default='basic_model')
    parser.add_argument('--decoder', default='decoder')
    # Mixed model-specific options
    parser.add_argument('--num-classes', type=int, help='number of dataset classes (if not inferred automatically)')
    # Training options
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--mode', default='train', choices=['train', 'self_train', 'auto_encoder'], help='training mode')
    parser.add_argument('--optim', default='Adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--reduce-lr-every', type=int)
    parser.add_argument('--reduce-lr-factor', type=float)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--momentum', default=0.9, type=float, help="momentum")
    parser.add_argument('--resume', help='path to checkpoint of model to resume')
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=9)
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--overfit-batch', action='store_true', help='debug: try to overfit the model on a single batch')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # Get params
    args = parse()
    # Suppress random
    #suppress_random(args.seed)

    # TODO this has to be rewritten for the actual dataset format
    # Define data transforms
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    # Load dataset
    train_dataset = CIFAR10(root='.', train=True, transform=train_transform, download=True)
    val_dataset = CIFAR10(root='.', train=True, transform=test_transform, download=True)
    test_dataset = CIFAR10(root='.', train=False, transform=test_transform, download=True)
    # Split training into train + val (not using random_split to preserve transforms)
    num_train = int(len(train_dataset)*0.8)
    idxs = list(range(len(train_dataset)))
    shuffle(idxs)
    train_idxs = idxs[:num_train]
    val_idxs = idxs[num_train:]
    train_dataset = Subset(train_dataset, train_idxs)
    val_dataset = Subset(val_dataset, train_idxs)

    # Adapt datasets to training mode
    if args.mode == 'self_train':
        # Rotation datasets
        train_dataset = RotationDataset(train_dataset, train=True)
        val_dataset = RotationDataset(val_dataset, train=False)
        test_dataset = RotationDataset(test_dataset, train=False)

    # Create dataset dictionary
    datasets = {'train': train_dataset,
                'val': val_dataset,
                'test': test_dataset}

    # Check overfit-batch (for debug)
    if args.overfit_batch:
        datasets = {k: Subset(datasets[k], list(range(args.batch_size))) for k in datasets}

    # Add dataset options to arguments
    args.num_channels = datasets['train'][0][0].shape[0]
    try:
        args.num_classes = datasets['train'].num_classes
    except:
        try:
            args.num_classes = len(datasets['train'].classes)
        except:
            pass
    assert args.mode == 'auto_encoder' or args.num_classes is not None, "couldn't get number of classes"

    # Define saver
    saver = Saver(args.logdir, args, sub_dirs=list(datasets.keys()), tag=args.tag)
    # Add saver to args (e.g. visualizing weights, outputs)
    args.saver = saver
    # Save splits
    # TODO adapt this to non-CIFAR dataset
    saver.save_data({'train': train_idxs, 'val': val_idxs}, 'split_idxs')

    # Define trainer
    trainer_module = getattr(trainers, "ae_trainer" if args.mode == 'auto_encoder' else 'trainer')
    trainer_class = getattr(trainer_module, 'Trainer')
    trainer = trainer_class(args)
    # Run training
    model, metrics = trainer.train(datasets)
    # Close saver
    saver.close()