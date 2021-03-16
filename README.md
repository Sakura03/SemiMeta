# Semi-Supervised Learning with Meta-Gradient

By Xin-Yu Zhang, Taihong Xiao, Haolin Jia, Ming-Ming Cheng, and Ming-Hsuan Yang.

## Prerequisite

| **Package**    | **version**  |
|----------------|--------------|
| python         |  >=3.5       |
| pytorch        |  >=1.2       |
| numpy          |  >=1.17.2    |
| tensorboardX   |  >=2.0       |

## Train the SemiMeta Algorithm on CIFAR/SVHN

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
## Train the SemiMeta Algorithm on ImageNet

```
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -m torch.distributed.launch --nproc_per_node=8 train_meta_imagenet.py \
        -a "resnet18" \
        --data "path/to/rec" \
        --save-path "/your/path/to/save/imagenet/model" \
        --batch-size "64" \
        --wd "1e-4" \
        --epochs "600" \
        --lr "0.1" \
        --alpha "1.0" \
        --epsilon "0.01" \
        --warmup "5"
```

## Visualize features in 2D space

The following scripts reproduce Figure 3 in the paper.

```
CUDA_VISIBLE_DEVICES="0" python3 plot_features.py \
        --dataset "svhn" \
        --checkpoint-path "/path/to/model_best.pth" \
        --index-path "path/to/label_indices.txt" \
        --save-path "/your/save/path" \
        --num-point '5000';

CUDA_VISIBLE_DEVICES="0" python3 plot_features.py \
        --dataset "svhn" \
        --checkpoint-path "/path/to/model_best.pth" \
        --index-path "path/to/label_indices.txt" \
        --save-path "/your/save/path" \
        --num-point '5000';
```

## Citation

```
@InProceedings{zhang2020semisupervised,
      title={Semi-Supervised Learning with Meta-Gradient}, 
      author={Xin-Yu Zhang and Taihong Xiao and Haolin Jia and Ming-Ming Cheng and Ming-Hsuan Yang},
      year={2020},
      eprint={2007.03966},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
