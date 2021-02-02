# ADA-Net
This repo is a pytorch implementation of [*Semi-Supervised Learning by Augmented Distribution Alignment*](https://arxiv.org/abs/1905.08171)

## Prerequisite

| **Package**    | **version**  |
|----------------|--------------|
| python         |  >=3.5       |
| pytorch        |  1.2         |
| numpy          |  1.17.2      |
| tensorboardX   |  2.0         |

**Note:** For more detail, please look up `requirements.txt`

## Train the model

```
chmod u+x scripts/train_model.sh
./scripts/train_model.sh local
```
## Test the model

```
chmod u+x scripts/test_model.sh
./scripts/test_model.sh local
```

## New script: See `run.sh`

## Acknowlegement

This repository is based on:
1. Official Tensorflow code, [repo](https://github.com/qinenergy/adanet)
2. Unofficial Pytorch code, [repo](https://github.com/jizongFox/AdaNet-PyTorch)
