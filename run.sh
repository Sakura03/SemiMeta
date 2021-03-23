# Only label data (Baseline)
# CUDA_VISIBLE_DEVICES=0 python3 train_baseline.py --save_path "/your/path/to/save/baseline/results" --gpu;
#
# Meta-learning (Ours)
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
        --weight-decay "1e-4" \
        --gpu;
