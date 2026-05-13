PYTHONPATH=.:$PYTHONPATH python create_datasets/urbancars/gen_urbancars.py --split test
PYTHONPATH=.:$PYTHONPATH python create_datasets/urbancars/urbancars_aug_gen_bg_only.py --data_root=data/urbancars/bg-0.95_co_occur_obj-0.95/train
