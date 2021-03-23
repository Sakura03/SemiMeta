# Semi-Supervised Learning with Meta-Gradient

Official implementation of paper: **Semi-Supervised Learning with Meta-Gradient** (AISTATS 2021), by Xin-Yu Zhang, Taihong Xiao, Haolin Jia, Ming-Ming Cheng, and Ming-Hsuan Yang. [[paper](https://arxiv.org/abs/2007.03966), [poster](images/aistats-poster.pdf), [video](images/poster-video.mp4), [short slides](images/brief-slides.pdf), [full slides](images/full-slides.pdf)]

Under construction.

## Introduction

This repository contains the official implementation of the `MetaSemi` algorithm for Semi-Supervised Learning (SSL). `MetaSemi` is a consistency-based SSL algorithm in which the **consistency loss** is guided by the **label information** of the specific task of interst. However, the consistency loss seems to have no relationship with the label information, so we borrow the idea of meta-learning to establish their relationship by differentiating through the gradient descent step.

## Algorithm

we formulate SSL as a bi-level optimization problem, as shown in the following image:
![Bi-level optimization.](images/formulation.png)

Solving the exact optimization problem is computationally prohibitive, so we adopt an online approximation approach. The `MetaSemi` algorithm is summarized below:
![Meta-learning algorithm.](images/algorithm.png)

Apart from meta-learning, we adopt several tricks to alleviate computation overhead and promote performance. Please refer to our [paper](https://arxiv.org/abs/2007.03966) for these details.

## Reproduce the Experimental Results

### Prerequisite

Please make sure the following packages are installed in your environment:

| **Package**    | **Version**  |
|----------------|--------------|
| python         |  >=3.5       |
| pytorch        |  >=1.2       |
| tensorboardX   |  >=2.0       |

Also, we provide the [Dockerfile](Docker/Dockerfile) containing all necessary dependencies. You can simply run the following scripts to enter the docker environment:

```
cd Docker
sudo docker build .
sudo docker images
sudo docker run <the-image-id> --network=host        # Enter the image id shown in the last command
sudo docker ps
sudo docker exec -it <the-container-id> bash         # Enter the container id shown in the last command
```

### SVHN and CIFAR Datasets

Our performance on SVHN and CIFAR datasets is as follows:

|    **Dataset**    |   **SVHN**   | **CIFAR-10** | **CIFAR-100** |
|-------------------|--------------|--------------|---------------|
| **Num of Labels** |     1000     |     4000     |     10000     |
|   **Error Rate**  |     3.15%    |     7.78%    |     30.74%    |

To reproduce these results, run the following script:

```
bash run.sh
```

### ImageNet Dataset

TODO

### Visualize features in 2D space

To reproduce Fig. 3 in our [paper](https://arxiv.org/abs/2007.03966), run the following script:

```
bash visualization.sh
```

## Citation

If you find our work intersting or helpful to your research, please consider citing our paper.

```
@inproceedings{zhang2020semisupervised,
  author    = {Xin-Yu Zhang and Taihong Xiao and Haolin Jia and Ming-Ming Cheng and Ming-Hsuan Yang},
  title     = {{Semi-Supervised Learning with Meta-Gradient}},
  booktitle = International Conference on Artificial Intelligence and Statistics (AISTATS),
  year      = {2021}
}
```
