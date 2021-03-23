# Semi-Supervised Learning with Meta-Gradient

Official implementation of paper: **Semi-Supervised Learning with Meta-Gradient** (AISTATS 2021), by Xin-Yu Zhang, Taihong Xiao, Haolin Jia, Ming-Ming Cheng, and Ming-Hsuan Yang. [[paper](https://arxiv.org/abs/2007.03966), [poster](images/aistats-poster.pdf), [video](images/poster-video.mp4), [short slides](images/brief-slides.pdf), [full slides](images/full-slides.pdf)]

Under construction

## Introduction

This repository contains the official implementation of the `MetaSemi` algorithm for Semi-Supervised Learning (SSL). `MetaSemi` is a consistency-based SSL algorithm in which the **consistency loss** is guided by the **label information** of the specific task of interst. However, the consistency loss seems to have no relationship with the label information, so we borrow the idea of meta-learning to establish their relationship by differentiating through the gradient descent step.

## Algorithm

we formulate SSL as a bi-level optimization problem, as shown in the following image:
![Bi-level optimization.](images/formulation.png)

Solving the exact optimization problem is computationally prohibitive, so we adopt an online approximation approach. The `MetaSemi` algorithm is summarized below:
![Meta-learning algorithm.](images/algorithm.png)

Apart from meta-learning, we adopt several tricks to alleviate computation overhead and promote performance. Please refer to our [paper](https://arxiv.org/abs/2007.03966) for these details.

## Reproduce the experimental results

### Prerequisite

Please make sure the following packages are installed in your environment. Also, we provide the `Dockerfile` in the `docker` directory.

| **Package**    | **version**  |
|----------------|--------------|
| python         |  >=3.5       |
| pytorch        |  >=1.2       |
| tensorboardX   |  >=2.0       |


Our performance is favorable on SVHN and CIFAR datasets:


### Train the SemiMeta Algorithm on CIFAR/SVHN

```
CUDA_VISIBLE_DEVICES='0' python3 train_meta.py \
        --dataset "cifar100" \
        --num-label "10000" \
        -a "convlarge" \
        --mix-up \
        --alpha "1.0" \
        --save-path "/your/path/to/save/results" \
        --weight "1.0" \
        --total-steps "400000" \
        --milestones "[300000, 350000]" \
        --lr "0.1" \
        --seed "7788" \
        --warmup "4000" \
        --weight-decay "1e-4";
```

### Visualize features in 2D space

The following scripts reproduce Figure 3 in the paper.

```
CUDA_VISIBLE_DEVICES="0" python3 plot_features.py \
        --dataset "svhn" \
        --checkpoint-path "/path/to/model_best.pth" \
        --index-path "path/to/label_indices.txt" \
        --save-path "/your/save/path" \
        --num-point '5000';
```

## Citation

```
@inproceedings{zhang2020semisupervised,
  author    = {Xin-Yu Zhang and Taihong Xiao and Haolin Jia and Ming-Ming Cheng and Ming-Hsuan Yang},
  title     = {{Semi-Supervised Learning with Meta-Gradient}},
  booktitle = International Conference on Artificial Intelligence and Statistics (AISTATS),
  year      = {2021}
}
```
