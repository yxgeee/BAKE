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

|architecture|ImageNet top-1 acc.|config|download|
|---|:--:|:--:|:--:|
|ResNet-50|78.0|[config](configs/resnet/BAKE-R-50-1x64d_dds_8gpu.yaml)||
|ResNet-101|79.3|[config](configs/resnet/BAKE-R-101-1x64d_dds_8gpu.yaml)||
|ResNet-152|79.6|[config](configs/resnet/BAKE-R-152-1x64d_dds_8gpu.yaml)||
|ResNeSt-50|79.4|[config](configs/resnest/BAKE-S-50_dds_8gpu.yaml)||
|ResNeSt-101|80.4|[config](configs/resnest/BAKE-S-101_dds_8gpu.yaml)||
|ResNeXt-101(32x4d)|79.3|[config](configs/resnext/BAKE-X-101-32x4d_dds_8gpu.yaml)||
|ResNeXt-152(32x4d)|79.7|[config](configs/resnext/BAKE-X-152-32x4d_dds_8gpu.yaml)||
|MobileNet-V2|72.0|[config](configs/mobilenet/BAKE-M-V2-W1_dds_4gpu.yaml)||
|EfficientNet-B0|76.2|[config](configs/effnet/BAKE-EN-B0_dds_8gpu.yaml)||

## Thanks
The code is modified from [pycls](https://github.com/facebookresearch/pycls).
