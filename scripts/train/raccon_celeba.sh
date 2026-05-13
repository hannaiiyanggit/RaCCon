# RaCCon: A Rank-Consistent Contrastive Learning Framework for Mitigating Spurious Correlations
# Written by Aiyang Han, PARNEC
# https://github.com/hannaiiyanggit/RaCCon

method_name=raccon
bias_id_epoch=1

PYTHONPATH=.:$PYTHONPATH python trainers/launcher.py \
    --method ${method_name} \
    --dataset celeba \
    --amp \
    --num_worker 10 \
    --slurm_job_name ${method_name}_${bias_id_epoch} \
    --early_stop_metric_list both \
    --num_seed 5 \
    --arch resnet18\
    --epoch 50 \
    --wandb \
    --beta_inverse 0.8 \
    --p_critical 0.7 \
    --lr 0.001 \
    --momentum 0.8 \
    --weight_decay 0.0001 \
    --gap 3 \
    --classifier_weight 100 \
    --temperature 0.05