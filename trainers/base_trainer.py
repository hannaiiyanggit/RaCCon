"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import json
import os
import random

import submitit
import wandb
import torch
import copy
import torch.nn as nn


from tqdm import tqdm

from dataset.bar import BAR

from torchvision import transforms


from model.classifiers import (
    get_classifier,
    get_transforms,
)
from utils import (
    set_seed,
    MultiDimAverageMeter,
)

from local_record import local_log


class BaseTrainer:
    def __init__(self, args, log):
        self.args = args
        self._setup_method_name_and_default_name()

        self.log = log
        self.log.refresh()

        self.cur_epoch = 1

        if args.run_name is None:
            args.run_name = self.default_name
        else:
            args.run_name += f"_{self.default_name}"
        ckpt_dir = os.path.join(
            args.exp_root, args.run_name, f"seed_{args.seed}"
        )

        if args.resume is None:
            os.makedirs(ckpt_dir, exist_ok=True)

        self.ckpt_dir = ckpt_dir

        self.ckpt_fname = "ckpt"

        self.cond_best_acc = 0
        self.cond_on_best_val_log_dict = {}

    def _setup_all(self):
        args = self.args
        set_seed(args.seed)
        self.device = torch.device(0)

        self.scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

        self._setup_early_stop_metric()
        self._setup_dataset()
        self._setup_models()
        self._setup_criterion()
        self._setup_optimizers()
        self._method_specific_setups()

        if args.wandb:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                name=args.run_name,
                config=args,
                settings=wandb.Settings(start_method="fork"),
                mode="offline"
            )

        # loading checkpoint
        if args.resume:
            ckpt_fpath = args.resume
            assert os.path.exists(ckpt_fpath), f"{ckpt_fpath} does not exist"
            state_dict = torch.load(ckpt_fpath, map_location="cpu")
            self._load_state_dict(state_dict)
        else:
            self._before_train()

    def _get_train_collate_fn(self):
        return None

    def _get_train_loader(self, train_set):
        args = self.args
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
            collate_fn=self._get_train_collate_fn(),
        )
        return train_loader

    def _setup_early_stop_metric(self):
        args = self.args

        early_stop_metric_arg_to_real_metric = {
            "biasA": "val_biasA_worst_group_acc",
            "biasB": "val_biasB_worst_group_acc",
            "both": "val_both_worst_group_acc",
            "normal":"val_cue_obj_acc"
        }

        if args.method in [
            "groupdro",
            "di",
            "subg",
            "dfr",
        ]:
            args.early_stop_metric_real = early_stop_metric_arg_to_real_metric[
                args.group_label
            ]
        elif args.method in [
            "erm",
            "lff",
            "eiil",
            "sd",
            "jtt",
            "debian",
            "lle",
            "cf_f_aug",
            "augmix",
            "cutmix",
            "mixup",
            "cutout",
            "sebra",
            "reweighting",
            "resampling",
            "raccon"
        ]:
            # methods that do not use group labels
            args.early_stop_metric_real = early_stop_metric_arg_to_real_metric[
                args.early_stop_metric
            ]
        else:
            raise ValueError(f"unknown method: {args.method}")

    def _get_train_transform(self):
        args = self.args
        train_transform = get_transforms(args.arch, is_training=True, dataset=args.dataset)
        return train_transform

    def _setup_dataset(self):
        args = self.args

        self.has_group=False

        train_transform = self._get_train_transform()
        test_transform = get_transforms(args.arch, is_training=False, dataset=args.dataset)

        if args.dataset == "urbancars":
            from dataset.urbancars import UrbanCars
            self.has_group = True
            self.biasA_name = "Background"
            self.biasB_name = "Co-occurring Object"

            train_set = UrbanCars(
                args.data_dir,
                "train",
                group_label=args.group_label,
                transform=train_transform,
                return_group_index=args.method in ["groupdro", "eiil", "reweighting"],
                return_domain_label=args.method == "di",
                return_dist_shift=args.method == "lle",
            )
            val_set = UrbanCars(
                args.data_dir,
                "val",
                transform=test_transform,
            )
            test_set = UrbanCars(
                args.data_dir,
                "test",
                transform=test_transform,
            )
        elif args.dataset == "celeba":
            from dataset.multiceleba import BiasedCelebA

            self.has_group = True
            idx2attr = json.load(open("create_datasets/celeba/idx2attr.json", 'r'))
            idx2attr = {int(k): v for k, v in idx2attr.items()}
            target_name = idx2attr[31] #args.target_id
            biasA_name = idx2attr[20] #args.biasA_id: male
            biasB_name = idx2attr[39] #args.biasB_id: young
            self.biasA_name = biasA_name
            self.biasB_name = biasB_name
            biasA_ratio = args.biasA_ratio
            biasB_ratio = args.biasB_ratio
            root = args.data_dir
            train_set = BiasedCelebA(root=root, target_name=target_name, biasA_name=biasA_name,
                                   biasB_name=biasB_name,
                                   biasA_ratio=biasA_ratio, biasB_ratio=biasB_ratio, split="train",
                                   transform=train_transform)

            val_set = BiasedCelebA(root=root, target_name=target_name, biasA_name=biasA_name,
                                     biasB_name=biasB_name,
                                     biasA_ratio=biasA_ratio, biasB_ratio=biasB_ratio, split="val",
                                     transform=test_transform)
            test_set = BiasedCelebA(root=root, target_name=target_name, biasA_name=biasA_name,
                                     biasB_name=biasB_name,
                                     biasA_ratio=biasA_ratio, biasB_ratio=biasB_ratio, split="test",
                                     transform=test_transform)
        elif args.dataset == "bar":

            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

            rank_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ])

            indices = list(range(1941))
            random.shuffle(indices)
            split_point = int(0.9 * len(indices))
            train_index = indices[:split_point]
            valid_index = indices[split_point:]

            dset_dir = os.path.join(args.data_dir, "BAR")
            train_set = BAR(gap=args.gap, root=dset_dir, split='train', transform=train_transform,
                            indices=train_index)
            self.rank_set = BAR(gap=args.gap, root=dset_dir, split='train', transform=rank_transform,
                            indices=train_index)

            val_set = BAR(gap=args.gap, root=dset_dir, split='train', transform=train_transform,
                            indices=valid_index)

            test_set = BAR(split='test', root=dset_dir, transform=test_transform, indices=None)

        self.train_set = train_set
        self.obj_name_list = train_set.obj_name_list
        self.num_class = len(self.obj_name_list)

        if self.has_group:
            print("Using attribute labels: \n"
                  f"biasA: {args.biasA_ratio*100}% {self.biasA_name}\n"
                  f"biasB: {args.biasB_ratio*100}% {self.biasB_name}")
        else:
            print("Using no attribute labels")

        train_set = self._modify_train_set(train_set)
        train_loader = self._get_train_loader(train_set)

        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
        )
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def _method_specific_setups(self):
        pass

    def _setup_models(self):
        args = self.args
        self.classifier = get_classifier(
            args.arch,
            self.num_class,
        ).to(self.device)

    def _set_train(self):
        self.classifier.train()

    def _setup_criterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def _setup_optimizers(self):
        args = self.args
        parameters = [
            p for p in self.classifier.parameters() if p.requires_grad
        ]
        if args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                parameters,
                args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
        elif args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(parameters, self.args.lr,
                                              weight_decay=self.args.weight_decay)
        else:
            raise NotImplementedError

    def _setup_method_name_and_default_name(self):
        raise NotImplementedError

    def _modify_train_set(self, train_dataset):
        return train_dataset

    def _before_train(self):
        pass

    def __call__(self):
        torch.backends.cudnn.benchmark = True
        args = self.args
        self._setup_all()

        for _ in range(self.cur_epoch, args.epoch + 1):
            self.train()
            result = self.eval()
            is_best, val_log_dict, test_log_dict = result
            state_dict = self._state_dict_for_save()
            self._save_ckpt(state_dict, self.ckpt_fname)
            if is_best:
                self._save_ckpt(state_dict, "best")

            for split in ["val", "test", "cond_test"]:
                id_acc = self.log

            self.show_result(self.cond_on_best_val_log_dict, "cond_test")

            self.cur_epoch += 1

    def train(self):
        raise NotImplementedError

    def show_result(self, log_dict, split):

        if self.has_group:
            id_acc = log_dict[f"{split}_id_acc"] * 100
            biasA_gap = log_dict[f"{split}_biasA_gap"] * 100
            biasB_gap = log_dict[f"{split}_biasB_gap"] * 100
            both_gap = log_dict[f"{split}_both_gap"] * 100
            avg_gap = log_dict[f"{split}_avg_gap"] * 100
            both_worst_group_acc = log_dict[f"{split}_both_worst_group_acc"]*100

            print(
                f"[{self.cur_epoch}/{self.args.epoch}][{split}] "
                f"ID Acc: {id_acc:.2f} "
                f"biasA gap: {biasA_gap:.2f} "
                f"biasB gap: {biasB_gap:.2f} "
                f"Both gap: {both_gap:.2f} "
                f"AVG gap: {avg_gap:.2f} "
                f"WGA: {both_worst_group_acc:.2f}"
            )
        else:
            id_acc = log_dict[f"{split}_cue_obj_acc"] * 100
            print(
                f"[{self.cur_epoch}/{self.args.epoch}][{split}] "
                f"ID Acc: {id_acc:.2f} ")

    def eval(self):
        val_log_dict = self._eval_split(self.val_loader, "val")
        test_log_dict = self._eval_split(self.test_loader, "test")
        self.show_result(val_log_dict, "val")
        self.show_result(test_log_dict, "test")

        early_stop_metric_result = val_log_dict[
            self.args.early_stop_metric_real
        ]

        if (
            early_stop_metric_result <= self.cond_best_acc
            and self.cur_epoch > 1
            and len(self.cond_on_best_val_log_dict) > 0
        ):
            self.log_to_wandb(self.cond_on_best_val_log_dict)
            return (False, val_log_dict, test_log_dict)  # not best

        is_best = False

        if early_stop_metric_result > self.cond_best_acc or self.cond_best_acc==0:
            self.cond_best_acc = early_stop_metric_result
            is_best = True
            for key, value in test_log_dict.items():
                new_key = f"cond_{key}"
                self.cond_on_best_val_log_dict[new_key] = value

        self.log_to_wandb(self.cond_on_best_val_log_dict)
        return (is_best, val_log_dict, test_log_dict)

    def get_prediction(self, image):
        return self.classifier(image)

    @torch.no_grad()
    def _eval_split(self, loader, split):
        args = self.args

        meter = MultiDimAverageMeter(
            (self.num_class, self.num_class, self.num_class)
        )
        total_correct = []
        total_biasA_correct = []
        total_biasB_correct = []
        total_shortcut_conflict_mask = []

        self.classifier.eval()
        pbar = tqdm(loader, dynamic_ncols=True)
        for data_dict in pbar:
            image, target = data_dict["image"], data_dict["label"]
            image = image.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(enabled=args.amp):
                output = self.get_prediction(image)

            pred = output.argmax(dim=1)

            if self.has_group:
                obj_label = target[:, 0]
            else:
                obj_label = target

            correct = pred == obj_label
            total_correct.append(correct.cpu())

            if self.has_group:
                meter.add(correct.cpu(), target.cpu())

                biasA_label = target[:, 1]
                biasB_label = target[:, 2]

                shortcut_conflict_mask = biasA_label != biasB_label
                total_shortcut_conflict_mask.append(shortcut_conflict_mask.cpu())

                biasA_correct = pred == biasA_label
                total_biasA_correct.append(biasA_correct.cpu())

                biasB_correct = pred == biasB_label
                total_biasB_correct.append(biasB_correct.cpu())

        total_correct = torch.cat(total_correct, dim=0)
        log_dict = {}

        if self.has_group:
            num_correct = meter.cum.reshape(*meter.dims)
            cnt = meter.cnt.reshape(*meter.dims)
            multi_dim_color_acc = num_correct / cnt

            absent_present_str_list = ["absent", "present"]
            absent_present_biasA_ratio_list = [1 - args.biasA_ratio, args.biasA_ratio]
            absent_present_biasB_ratio_list = [
                1 - args.biasB_ratio,
                args.biasB_ratio,
            ]

            weighted_group_acc = 0
            for biasA_shortcut in range(len(absent_present_str_list)):
                for biasB_shortcut in range(len(absent_present_str_list)):
                    biasA_shortcut_mask = (meter.eye_tsr == biasA_shortcut).unsqueeze(2)
                    biasB_shortcut_mask = (
                        meter.eye_tsr == biasB_shortcut
                    ).unsqueeze(1)
                    mask = biasA_shortcut_mask * biasB_shortcut_mask
                    acc = multi_dim_color_acc[mask].mean().item()
                    biasA_shortcut_str = absent_present_str_list[biasA_shortcut]
                    biasB_shortcut_str = absent_present_str_list[
                        biasB_shortcut
                    ]
                    log_dict[
                        f"{split}_biasA_{biasA_shortcut_str}"
                        f"_biasB_{biasB_shortcut_str}_acc"
                    ] = acc
                    cur_group_biasA_ratio = absent_present_biasA_ratio_list[biasA_shortcut]
                    cur_group_biasB_ratio = (
                        absent_present_biasB_ratio_list[biasB_shortcut]
                    )
                    cur_group_ratio = (
                        cur_group_biasA_ratio * cur_group_biasB_ratio
                    )
                    weighted_group_acc += acc * cur_group_ratio

            worst_group_acc = min(log_dict[f"{split}_biasA_present_biasB_present_acc"],
                                  log_dict[f"{split}_biasA_absent_biasB_present_acc"],
                                  log_dict[f"{split}_biasA_present_biasB_absent_acc"],
                                  log_dict[f"{split}_biasA_absent_biasB_absent_acc"])

            biasA_gap = (
                log_dict[f"{split}_biasA_absent_biasB_present_acc"]
                - weighted_group_acc
            )
            biasB_gap = (
                log_dict[f"{split}_biasA_present_biasB_absent_acc"]
                - weighted_group_acc
            )
            both_gap = (
                log_dict[f"{split}_biasA_absent_biasB_absent_acc"]
                - weighted_group_acc
            )

            log_dict.update(
                {
                    f"{split}_id_acc": weighted_group_acc,
                    f"{split}_biasA_gap": biasA_gap,
                    f"{split}_biasB_gap": biasB_gap,
                    f"{split}_both_gap": both_gap,
                    f"{split}_avg_gap": (both_gap+biasA_gap+biasB_gap)/3,
                }
            )

            total_biasA_correct = torch.cat(total_biasA_correct, dim=0)
            total_biasB_correct = torch.cat(
                total_biasB_correct, dim=0
            )

            (
                biasA_worst_group_acc,
                biasB_worst_group_acc,
                both_worst_group_acc,
            ) = meter.get_worst_group_acc()

            log_dict.update(
                {
                    f"{split}_biasA_worst_group_acc": biasA_worst_group_acc,
                    f"{split}_biasB_worst_group_acc": biasB_worst_group_acc,
                    f"{split}_both_worst_group_acc": both_worst_group_acc,
                    #f"{split}_groupwise_worst_group_acc": worst_group_acc,
                }
            )

            biasA_acc = total_biasA_correct.float().mean().item()
            biasB_acc = total_biasB_correct.float().mean().item()
            log_dict.update(
                {
                    f"{split}_cue_biasA_acc": biasA_acc,
                    f"{split}_cue_biasB_acc": biasB_acc,
                }
            )

        #if args.method == "erm" or args.method == 'sebra':
        # evaluate cue preference for ERM

        obj_acc = total_correct.float().mean().item()
        log_dict.update({f"{split}_cue_obj_acc": obj_acc,})

        log_dict['epoch'] = self.cur_epoch

        self.log_to_wandb(log_dict)

        return log_dict

    def _state_dict_for_save(self):
        state_dict = {
            "classifier": self.classifier.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.cur_epoch,
            "cond_best_acc": self.cond_best_acc,
            "cond_on_best_val_log_dict": self.cond_on_best_val_log_dict,
        }
        return state_dict

    def _load_state_dict(self, state_dict):
        self.scaler.load_state_dict(state_dict["scaler"])
        self.cur_epoch = state_dict["epoch"] + 1
        self.classifier.load_state_dict(state_dict["classifier"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.cond_best_acc = state_dict["cond_best_acc"]
        self.cond_on_best_val_log_dict = state_dict["cond_on_best_val_log_dict"]

    def _save_ckpt(self, state_dict, name):
        ckpt_fpath = os.path.join(self.ckpt_dir, f"{name}.pth")
        torch.save(state_dict, ckpt_fpath)

    def _loss_backward(self, loss, retain_graph=False):
        if self.args.amp:
            self.scaler.scale(loss).backward(retain_graph=retain_graph)
        else:
            loss.backward(retain_graph=retain_graph)

    def _optimizer_step(self, optimizer):
        if self.args.amp:
            self.scaler.step(optimizer)
        else:
            optimizer.step()

    def _scaler_update(self):
        if self.args.amp:
            self.scaler.update()

    def checkpoint(self):
        new_args = copy.deepcopy(self.args)
        ckpt_fpath = os.path.join(self.ckpt_dir, f"{self.ckpt_fname}.pth")
        if os.path.exists(ckpt_fpath):
            new_args.resume = ckpt_fpath
        training_callable = self.__class__(new_args)
        # Resubmission to the queue is performed through the DelayedSubmission object
        return submitit.helpers.DelayedSubmission(training_callable)

    def log_to_wandb(self, log_dict, step=None):
        self.log.update(log_dict)
        if step is None:
            step = self.cur_epoch
        if self.args.wandb:
            wandb.log(log_dict,) #step=step)
