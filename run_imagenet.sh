### Run baseline model
CUDA_VISIBLE_DEVICES='0,1,2,3' python -m torch.distributed.launch --nproc_per_node=4 train_imagenet.py \
    -a "resnet18" \
    --data "/media/ssd/imagenet/rec" \
    --tmp "results/imagenet-baseline-epoch600-bs512-cosine" \
    --batch-size "128" \
    --wd "1e-4" \
    --epochs "600" \
    --lr "0.1" \
    --warmup "5"

### Run meta-learning algorithm
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -m torch.distributed.launch --nproc_per_node=8 train_meta_imagenet.py \
    -a "resnet18" \
    --data "/media/ssd/imagenet/rec" \
    --tmp "results/imagenet-meta-learning-epoch600-bs512-cosine" \
    --batch-size "64" \
    --wd "1e-4" \
    --epochs "600" \
    --lr "0.1" \
    --alpha "1.0" \
    --epsilon "0.01" \
    --warmup "5"


