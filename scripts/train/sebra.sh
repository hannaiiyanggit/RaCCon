# Play Hardball: Hard-to-Easy Curriculum Learning for Mitigating Spurious Correlations.
# Written by Aiyang Han, PARNEC
# https://github.com/hannaiiyanggit/HECL

method_name=sebra
bias_id_epoch=1

PYTHONPATH=.:$PYTHONPATH python trainers/launcher.py \
    --method ${method_name} \
    --dataset urbancars \
    --amp \
    --num_worker 10 \
    --slurm_job_name ${method_name}_${bias_id_epoch} \
    --early_stop_metric_list both \
    --num_seed 5 \
    --arch resnet18\
    --epoch 100 \
    --wandb \
    --momentum 0.1 \
    --weight_decay 1e-3 \
    --lr 1e-3 \
    --lr2 1e-3 \
    --weight_decay2 1e-3 \
    --batch_size 128 \
    --batch_size_stage2 64 \
    --beta_inverse 1.25