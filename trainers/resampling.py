import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler, DataLoader
from .erm import ERMTrainer


class ResamplingTrainer(ERMTrainer):
    def _setup_method_name_and_default_name(self):
        args = self.args
        # 显式指定 method 名字，确保日志和路径正确
        args.method = "resampling"
        default_name = f"{args.method}_es_{args.early_stop_metric}_{args.dataset}"
        self.default_name = default_name

    def _get_train_loader(self, train_set):
        """
        覆盖 BaseTrainer 的方法，注入 WeightedRandomSampler
        """
        args = self.args

        # 1. 获取组信息
        # 根据 UrbanCars 和 BiasedCelebA 的常见实现，组信息存储在 group_array 中
        if hasattr(train_set, "group_array"):
            group_array = train_set.group_array
        else:
            # 兼容性处理：如果 dataset 没有直接暴露 group_array，
            # 尝试从 dataset 的属性中提取（某些版本可能叫 _group_array）
            group_array = getattr(train_set, "_group_array", None)

        if group_array is None:
            raise ValueError(
                f"Resampling 需要数据集包含 group_array 属性，但 {args.dataset} 未提供。"
            )

        # 2. 计算采样权重
        # group_array 是一个和数据集等长的数组，存储每个样本的 group id
        groups, group_counts = np.unique(group_array, return_counts=True)
        group_weights = 1.0 / group_counts

        # 映射到每个样本：每个样本的权重 = 1 / 该组的总样本数
        sample_weights = torch.from_numpy(group_weights[group_array]).float()

        # 3. 创建 Sampler
        # replacement=True 表示过采样（少数派会被多次抽到）
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        # 4. 构建 DataLoader
        # 注意：使用 sampler 时，shuffle 必须为 False
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
            sampler=sampler,
            collate_fn=self._get_train_collate_fn(),
        )

        print(f"成功注入 Resampling Sampler。组数: {len(groups)}, 最小组样本数: {min(group_counts)}")
        return train_loader