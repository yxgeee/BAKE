# BAKE on Small-scale Datasets

PyTorch implementation of [Self-distillation with Batch Knowledge Ensembling Improves ImageNet Classification](https://arxiv.org/abs/2104.13298) on CIFAR-100, TinyImageNet, CUB-200-2011, Stanford Dogs, and MIT67.

## Requirements

- torch >= 1.2.0
- torchvision >= 0.4.0

## Datasets

Please download the raw datasets and re-organize them according to [DATA](data/). The folder tree should be like
```
├── data
│   ├── README.md
│   ├── cifar-100-python/
│   ├── CUB200/
│   ├── MIT67/
│   ├── STANFORD120/
│   ├── tinyimagenet/
│   ├── tools/
└── └── ...
```

## BAKE Training

```
sh scripts/train_bake.sh <GPU_ID> <DATASET> <ARCH> <BATCH_SIZE> <M> <OMEGA>
```

Specifically,
+ CIFAR-100

```
sh scripts/train_bake.sh 0 cifar100 CIFAR_ResNet18 128 3 0.5
sh scripts/train_bake.sh 0 cifar100 CIFAR_DenseNet121 128 3 0.5
```

+ TinyImageNet

```
sh scripts/train_bake.sh 0 tinyimagenet CIFAR_ResNet18 128 1 0.9
sh scripts/train_bake.sh 0 tinyimagenet CIFAR_DenseNet121 128 1 0.9
```

+ CUB-200-2011

```
sh scripts/train_bake.sh 0 CUB200 resnet18 32 3 0.5
sh scripts/train_bake.sh 0 CUB200 densenet121 32 3 0.5
```

+ Stanford Dogs

```
sh scripts/train_bake.sh 0 STANFORD120 resnet18 32 1 0.9
sh scripts/train_bake.sh 0 STANFORD120 densenet121 32 1 0.9
```

+ MIT67

```
sh scripts/train_bake.sh 0 MIT67 resnet18 32 1 0.9
sh scripts/train_bake.sh 0 MIT67 densenet121 32 1 0.9
```

## Baseline Training

```
sh scripts/train_baseline.sh <GPU_ID> <DATASET> <ARCH> <BATCH_SIZE>
```

## Validation

```
sh scripts/val.sh <GPU_ID> <DATASET> <ARCH> <CKPT>
```

## Results (top-1 error)

||CIFAR-100|TinyImageNet|CUB-200-2011|Stanford Dogs|MIT67|
|---|:--:|:--:|:--:|:--:|:--:|
|ResNet-18|21.28|41.71|29.74|30.20|39.95|
|DenseNet-121|20.74|37.07|28.79|27.66|39.15|

## Thanks
The code is modified from [CS-KD](https://github.com/alinlab/cs-kd).
