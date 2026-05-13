import random
import numpy as np
import torch
from PIL import Image
import glob
import os


class BAR:
    def __init__(self, gap=0, root="BAR/dataset0.01.torch",
                 split="train"
                 , transform=None, save_flag=False, second_image=False, transform_easy=None, indices=None):
        root = os.path.join(root, split)
        print(root)
        self.obj_name_list = ['climbing', 'diving', 'fishing', 'racing', 'throwing', 'pole vaulting']
        image_paths = glob.glob(os.path.join(root, "**", "*.*"), recursive=True)
        self.image_list = [
            p for p in image_paths
            if p.split('.')[-1].lower() in {'jpg', 'jpeg', 'png', 'bmp'}
        ]
        class_dic = {'climbing': 0, 'diving': 1, 'fishing': 2, 'racing': 3, 'throwing': 4, 'pole vaulting': 5}
        self.target = [class_dic[self.image_list[i].split("/")[-1].split("_")[0]] for i in range(len(self.image_list))]
        if indices is not None:
            self.image_list = [self.image_list[i] for i in indices]
            self.target = [self.target[i] for i in indices]
        self.gap = gap
        self.return_contrastive_pairs = False
        self.transform = transform
        image_path = [self.image_list[i] for i in range(len(self.image_list))]
        self.images = [Image.open(path).convert('RGB') for path in image_path]
        if save_flag:
            self.images = [None] * len(self.image_list)

    def __getitem__(self, i):
        img_origin = self.images[i]
        img = self.transform(img_origin)
        target_label = self.target[i]
        data_dict = {'image': img, 'label': target_label, 'index': i}
        if self.return_contrastive_pairs:
            rank = self.ranks[i]
            max_rank = int(self.max_rank[self.target[i]])
            pos_rank = int(min(rank + self.gap, max_rank))
            data_dict['pos_labels'] = self.target[i]
            data_dict['pos_rank'] = pos_rank
            neg_idx = np.random.choice(self.indices_label_rank[self.target[i]][rank])
            try:
                pos_idx = np.random.choice(self.indices_label_rank[self.target[i]][pos_rank])
            except:
                found = False
                for fallback_rank in range(pos_rank + 1, max_rank + 1):
                    indices = self.indices_label_rank[self.target[i]].get(fallback_rank, [])
                    if len(indices) > 0:
                        pos_idx = np.random.choice(indices)
                        found = True
                        data_dict['pos_rank'] = fallback_rank
                        break
                if not found:
                    pos_idx = -1
            if pos_idx != -1:
                x_pos = self.images[pos_idx]
                data_dict['positive'] = self.transform(x_pos)
            else:
                return None
            x_neg = self.images[neg_idx]
            data_dict['negative'] = self.transform(x_neg)
            data_dict['rank'] = rank
        return data_dict

    def __len__(self):
        return len(self.image_list)


def get_action_dataset_train_test(args, train_transform, test_transform, rank_transform, train_index, valid_index):
    train_set = BAR(gap=args.gap, root=args.dset_dir, split='train', transform=train_transform, indices=train_index)
    valid_set = BAR(gap=args.gap, root=args.dset_dir, split='train', transform=train_transform, indices=valid_index)
    test_set = BAR(split='test', root=args.dset_dir, transform=test_transform, indices=None)
    train_no_aug = BAR(split='train', root=args.dset_dir, transform=rank_transform, indices=train_index)
    return train_set, valid_set, test_set, train_no_aug
