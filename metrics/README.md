# Metrics to evaluate the performance of different models

Given that the focus is on detecting Covid-19 cases, the following evaluation
measures should be used to evaluate binary classification problems.

This class is only a wrapper that uses a limited number of metrics from
Scikit-learn and PyPRG.

## Virtual environment

```bash
# Use the module venv to create a virtual environment in a folder called venv
python3.6 -m venv venv
# Activate the virtual environment
source venv/bin/activate
# Upgrade package installer for Python
pip install --upgrade pip
# Install all the required dependencies indicated in the requirements.txt file
pip install -r requirements.txt
```

## Unittest

The current code uses docstrings that can be used as a test by just running the
code `python metrics.py`

## Code overview

The file metrics.py contains all the functions that compute scores given a
target and the output probabilities of a binary classification model.

The file results.py collects the outputs of a classifier together with the
targets and computes and exports all the results into the indicated folder.

A simple example of results can be run with

```
python results.py
```

which will generate a folder called results containing several csv files, one
per metric, and one for the model outputs and information about each sample.
