# BAKE on ImageNet

PyTorch implementation of [Self-distillation with Batch Knowledge Ensembling Improves ImageNet Classification](https://arxiv.org/abs/2104.13298) on ImageNet.


## Installation

- Install Python dependencies:

```
pip install -r requirements.txt
```

- Set up Python modules:

```
python setup.py develop --user
```

## Dataset

- Download ImageNet with an expected structure:

```
imagenet
|_ train
|  |_ n01440764
|  |_ ...
|  |_ n15075141
|_ val
|  |_ n01440764
|  |_ ...
|  |_ n15075141
|_ ...
```

- Create a directory containing symlinks:

```
mkdir -p pycls/datasets/data
```

- Symlink ImageNet:

```
ln -s /path/imagenet pycls/datasets/data/imagenet
```


## BAKE Training

```
sh tools/dist_train.sh <WORK_DIR> <CONFIG>
```

For example,
```
sh tools/dist_train.sh logs/resnet50_bake configs/resnet/BAKE-R-50-1x64d_dds_8gpu.yaml
```

**Note**: you could use `tools/slurm_train.sh` for distributed training with multiple machines.


## Validation

```
sh tools/dist_test.sh <CONFIG> <CKPT>
```

For example,
```
sh tools/dist_test.sh configs/resnet/BAKE-R-50-1x64d_dds_8gpu.yaml logs/resnet50_bake/checkpoints/model_epoch_0100.pyth
```

## Results

|architecture|ImageNet top-1 acc.|config|url|
|---|:--:|:--:|:--:|
|ResNet-50|78.0|[config](configs/resnet/BAKE-R-50-1x64d_dds_8gpu.yaml)||

## Thanks
The code is modified from [pycls](https://github.com/facebookresearch/pycls).
