CUDA_VISIBLE_DEVICES="1" python3 plot_features.py \
        --dataset "svhn" \
        --checkpoint-path "results/svhn-labels1000-baseline-mile30-35-wd5e-5-balanced/model_best.pth" \
        --index-path "results/svhn-labels1000-baseline-mile30-35-wd5e-5-balanced/label_indices.txt" \
        --save-path "results/svhn-labels1000-baseline-mile30-35-wd5e-5-balanced/visualization" \
        --num-point '5000';

CUDA_VISIBLE_DEVICES="1" python3 plot_features.py \
        --dataset "svhn" \
        --checkpoint-path "results/svhn-labels1000-mile30-35-mixup0.1-wd5e-5-balanced-run2/model_best.pth" \
        --index-path "results/svhn-labels1000-mile30-35-mixup0.1-wd5e-5-balanced-run2/label_indices.txt" \
        --save-path "results/svhn-labels1000-mile30-35-mixup0.1-wd5e-5-balanced-run2/visualization" \
        --num-point '5000';

