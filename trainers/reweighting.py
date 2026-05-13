import torch
import torch.nn as nn
import numpy as np
from .erm import ERMTrainer


class WeightedLoss(nn.Module):
    def __init__(self, base_criterion, group_weights):
        super(WeightedLoss, self).__init__()
        self.base_criterion = base_criterion  # 应当是 CrossEntropyLoss(reduction='none')
        self.register_buffer('group_weights', group_weights)

    def forward(self, logits, targets, group_indices):
        # 1. 计算每个样本的原始 Loss
        per_sample_losses = self.base_criterion(logits, targets)

        # 2. 提取对应组的权重
        # group_indices 形状 [batch_size]，从 buffer 中拿权重
        batch_weights = self.group_weights[group_indices]

        # 3. 加权平均
        weighted_loss = (per_sample_losses * batch_weights).mean()
        return weighted_loss


class ReweightingTrainer(ERMTrainer):
    def _setup_method_name_and_default_name(self):
        args = self.args
        args.method = "reweighting"
        # 这里的名字会影响保存路径
        self.default_name = f"{args.method}_es_{args.early_stop_metric}_{args.dataset}"

    def _modify_train_set(self, train_dataset):
        """
        核心修正：在 BaseTrainer 第 213 行被调用。
        强制修改数据集对象的属性，确保 __getitem__ 返回 group_idx。
        """
        if hasattr(train_dataset, "return_group_index"):
            train_dataset.return_group_index = True
            print(" -> [Reweighting] 强制开启 Dataset 的 return_group_index 成功")
        return train_dataset

    def _method_specific_setups(self):
        """
        在 BaseTrainer 第 66 行被调用，此时 dataset 已就绪。
        """
        args = self.args

        # 获取组分布信息
        if not hasattr(self.train_set, "group_array"):
            raise ValueError(f"数据集 {args.dataset} 缺少 group_array，无法计算 Reweighting 权重。")

        group_array = self.train_set.group_array
        groups, group_counts = np.unique(group_array, return_counts=True)

        # 计算归一化权重: W_g = N / (G * N_g)
        # 这样权重的期望值在 1 附近，不需要大幅度调整 Learning Rate
        weights = len(group_array) / (len(groups) * group_counts)
        weights_tensor = torch.from_numpy(weights).float().to(self.device)

        # 注入自定义的 WeightedLoss
        base_criterion = nn.CrossEntropyLoss(reduction='none')
        self.criterion = WeightedLoss(base_criterion, weights_tensor)

        print(f" -> [Reweighting] 权重初始化完成: {weights}")

    def train(self):
        args = self.args
        self._set_train()
        from utils import AverageMeter
        losses = AverageMeter("Loss", ":.4e")
        from tqdm import tqdm

        pbar = tqdm(self.train_loader, dynamic_ncols=True)
        for data_dict in pbar:
            image, target = data_dict["image"], data_dict["label"]

            # --- 修正点：兼容两种可能的命名 ---
            # 你的报错显示现在的键是 'group_index'
            group_idx = data_dict.get("group_index")
            if group_idx is None:
                group_idx = data_dict.get("group_idx")

            if group_idx is None:
                available_keys = list(data_dict.keys())
                raise KeyError(f"仍然找不到组索引。现有键: {available_keys}")
            # -------------------------------

            image = image.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            group_idx = group_idx.to(self.device, non_blocking=True)

            # UrbanCars 的目标标签
            obj_gt = target[:, 0] if self.has_group else target

            with torch.cuda.amp.autocast(enabled=args.amp):
                output = self.classifier(image)
                loss = self.criterion(output, obj_gt, group_idx)

            self._loss_backward(loss)
            self._optimizer_step(self.optimizer)
            self._scaler_update()
            self.optimizer.zero_grad(set_to_none=True)

            losses.update(loss.item(), image.size(0))
            pbar.set_description(f"[{self.cur_epoch}/{args.epoch}] RW-Loss: {losses.avg:.4f}")

        self.log_to_wandb({"train_loss": losses.avg})