# Only label data (Baseline)
# CUDA_VISIBLE_DEVICES=0 python3 train_baseline.py --save_path "results/baseline"
#
# Ada-Net
# CUDA_VISIBLE_DEVICES=0 python3 train_adanet.py --save_path "results/ada-Net"
#
# Meta-learning (Ours)
CUDA_VISIBLE_DEVICES='1' python3 train_meta.py \
        --dataset "cifar100" \
        --num-label "10000" \
        -a "wrn" \
	--mix-up \
	--alpha "1.0" \
        --save-path "results/cifar100-labels10000-wrn-mile30-35-wd1e-4-weight1.0" \
        --weight "1.0" \
	--total-steps "400000" \
        --milestones "[300000, 350000]" \
        --lr "0.1" \
	--seed "7788" \
        --warmup "4000" \
        --const-steps "0" \
        --weight-decay "1e-4";
