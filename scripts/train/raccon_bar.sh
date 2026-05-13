# RaCCon: A Rank-Consistent Contrastive Learning Framework for Mitigating Spurious Correlations
# Written by Aiyang Han, PARNEC
# https://github.com/hannaiiyanggit/RaCCon

method_name=raccon
bias_id_epoch=1

PYTHONPATH=.:$PYTHONPATH python trainers/launcher.py \
    --method ${method_name} \
    --dataset bar \
    --amp \
    --num_worker 10 \
    --slurm_job_name ${method_name}_${bias_id_epoch} \
    --early_stop_metric_list normal \
    --num_seed 5 \
    --arch resnet18\
    --epoch 100 \
    --wandb \
    --beta_inverse 1.42 \
    --p_critical 0.75 \
    --lr 1e-4 \
    --momentum 0.8 \
    --weight_decay 0 \
    --gap 2 \
    --classifier_weight 0.5 \
    --batch_size 64 \
    --optimizer adam \
    --lr2 1e-4 \
    --weight_decay2 0 \
    --temperature 0.15