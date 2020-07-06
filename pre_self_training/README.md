*Draft: this could change when adapting to actual data loaders*

### Description

This code allows to train a model from scratch, both on the target task or on a pretext task (rotation classification or auto-encoding) and resume a previously-trained model, even from a different training modality.

### Requirements

This code requires the following libraries (and possibly others; version numbers to be added):
* kornia
* opencv
* pytorch
* torchvision
* tb-nightly

### Usage

#### Training from scratch

`python scripts/train.py --mode train`

#### Self-supervised (rotation) training

`python scripts/train.py --mode self_train`

#### Auto-encoding training

`python scripts/train.py --mode auto_encoder`

#### Resuming training

Training data are saved by default to an `exps` directory by TensorBoard. Therein, a `ckpt` directory contains checkpoints with saved model parameters.

To resume a previous training, you can pass a specific checkpoint file or the `ckpt` directory as a `--resume` argument: in the latter case, the latest checkpoint from that experiment will be used.

For example:

`python scripts/train.py --mode train --resume exps/my_experiment/ckpt`

### Development

#### Adding a model

Models are located in the `models` directory, and must adhere to the following interface:
* The model's module contains a `Model` class.
* The `Model` class has a constructor that receives a dictionary, containing all command-line arguments.
  * In particular, all models should expect to receive a `data_size` argument (height/width of input images), a `data_channels` argument (number of channels of input images) and a `num_classes` argument (number of classes in the dataset).
* If training for auto-encoding, the model should accept a `return_features` argument, that -- when `True` -- should skip the fully-connected layers and return the last convolutional feature maps.

#### Adding a training procedure

The code contains training algorithms for classification (including self-training by rotation classification) and auto-encoding. If new training algorithms are needed, they should be added as a module to the `trainers` directory.
Each module should adhere to the following interface:
* The module contains a `Trainer` class.
* The `Trainer` class has a constructor that receives a dictionary containing all command-line arguments.
* The `Trainer` class has a `train` method that receives a dictionary of datasets on which a model should be trained.
  * The `train` method should create the model and the optimizer and carry out the actual training.

#### Adding a dataset

Datasets should follow the PyTorch `torch.utils.data.Dataset` interface. At the current state, the code is using CIFAR10. When data loaders for actual datasets are ready, the `scripts/train.py` script should be updated.
