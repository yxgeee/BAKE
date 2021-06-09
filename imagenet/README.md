# BAKE on ImageNet

PyTorch implementation of [Self-distillation with Batch Knowledge Ensembling Improves ImageNet Classification](https://arxiv.org/abs/2104.13298) on ImageNet.


## Installation

Install Python dependencies:

```
pip install -r requirements.txt
```

Set up Python modules:

```
python setup.py develop --user
```


## Using pycls

Please see [`INSTALL.md`](docs/INSTALL.md) for brief installation instructions. After installation, please see [`GETTING_STARTED.md`](docs/GETTING_STARTED.md) for basic instructions and example commands on training and evaluation with **pycls**.

## Model Zoo

We provide a large set of baseline results and pretrained models available for download in the **pycls** [Model Zoo](MODEL_ZOO.md); including the simple, fast, and effective [RegNet](https://arxiv.org/abs/2003.13678) models that we hope can serve as solid baselines across a wide range of flop regimes.

## Projects

A number of projects at FAIR have been built on top of **pycls**:

- [On Network Design Spaces for Visual Recognition](https://arxiv.org/abs/1905.13214)
- [Exploring Randomly Wired Neural Networks for Image Recognition](https://arxiv.org/abs/1904.01569)
- [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)
- [Are Labels Necessary for Neural Architecture Search?](https://arxiv.org/abs/2003.12056)
- [PySlowFast Video Understanding Codebase](https://github.com/facebookresearch/SlowFast)

If you are using **pycls** in your research and would like us to include your project here, please let us know or send a PR.

## Citing pycls

If you find **pycls** helpful in your research or refer to the baseline results in the [Model Zoo](MODEL_ZOO.md), please consider citing:

```
@InProceedings{Radosavovic2019,
  title = {On Network Design Spaces for Visual Recognition},
  author = {Radosavovic, Ilija and Johnson, Justin and Xie, Saining and Lo, Wan-Yen and Doll{\'a}r, Piotr},
  booktitle = {ICCV},
  year = {2019}
}

@InProceedings{Radosavovic2020,
  title = {Designing Network Design Spaces},
  author = {Radosavovic, Ilija and Kosaraju, Raj Prateek and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle = {CVPR},
  year = {2020}
}
```

## License

**pycls** is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Contributing

We actively welcome your pull requests! Please see [`CONTRIBUTING.md`](docs/CONTRIBUTING.md) and [`CODE_OF_CONDUCT.md`](docs/CODE_OF_CONDUCT.md) for more info.
