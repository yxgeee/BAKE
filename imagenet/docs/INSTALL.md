# Installation Instructions

This document covers how to install **pycls** and its dependencies.

## pycls

Clone the **pycls** repository:

```
# PYCLS=/path/to/clone/pycls
git clone https://github.com/facebookresearch/pycls $PYCLS
```

Install Python dependencies:

```
pip install -r $PYCLS/requirements.txt
```

Set up Python modules:

```
cd $PYCLS && python setup.py develop --user
```

## Datasets

**pycls** finds datasets via symlinks from `pycls/datasets/data` to the actual locations where the dataset images and annotations are stored. For instructions on how to create symlinks for CIFAR and ImageNet, please see [`DATA.md`](DATA.md).

## Getting Started

Please see [`GETTING_STARTED.md`](GETTING_STARTED.md) for basic instructions on training and evaluation with **pycls**.
