import random
import numpy as np
import torch
from PIL import Image
import glob
import os

import torch
import json
import pickle
import numpy as np


from typing import Any
from torchvision.datasets.folder import ImageFolder

imagenet_200_wnids = [
        "n01443537",
        "n01484850",
        "n01494475",
        "n01498041",
        "n01514859",
        "n01518878",
        "n01531178",
        "n01534433",
        "n01614925",
        "n01616318",
        "n01630670",
        "n01632777",
        "n01644373",
        "n01677366",
        "n01694178",
        "n01748264",
        "n01770393",
        "n01774750",
        "n01784675",
        "n01806143",
        "n01820546",
        "n01833805",
        "n01843383",
        "n01847000",
        "n01855672",
        "n01860187",
        "n01882714",
        "n01910747",
        "n01944390",
        "n01983481",
        "n01986214",
        "n02007558",
        "n02009912",
        "n02051845",
        "n02056570",
        "n02066245",
        "n02071294",
        "n02077923",
        "n02085620",
        "n02086240",
        "n02088094",
        "n02088238",
        "n02088364",
        "n02088466",
        "n02091032",
        "n02091134",
        "n02092339",
        "n02094433",
        "n02096585",
        "n02097298",
        "n02098286",
        "n02099601",
        "n02099712",
        "n02102318",
        "n02106030",
        "n02106166",
        "n02106550",
        "n02106662",
        "n02108089",
        "n02108915",
        "n02109525",
        "n02110185",
        "n02110341",
        "n02110958",
        "n02112018",
        "n02112137",
        "n02113023",
        "n02113624",
        "n02113799",
        "n02114367",
        "n02117135",
        "n02119022",
        "n02123045",
        "n02128385",
        "n02128757",
        "n02129165",
        "n02129604",
        "n02130308",
        "n02134084",
        "n02138441",
        "n02165456",
        "n02190166",
        "n02206856",
        "n02219486",
        "n02226429",
        "n02233338",
        "n02236044",
        "n02268443",
        "n02279972",
        "n02317335",
        "n02325366",
        "n02346627",
        "n02356798",
        "n02363005",
        "n02364673",
        "n02391049",
        "n02395406",
        "n02398521",
        "n02410509",
        "n02423022",
        "n02437616",
        "n02445715",
        "n02447366",
        "n02480495",
        "n02480855",
        "n02481823",
        "n02483362",
        "n02486410",
        "n02510455",
        "n02526121",
        "n02607072",
        "n02655020",
        "n02672831",
        "n02701002",
        "n02749479",
        "n02769748",
        "n02793495",
        "n02797295",
        "n02802426",
        "n02808440",
        "n02814860",
        "n02823750",
        "n02841315",
        "n02843684",
        "n02883205",
        "n02906734",
        "n02909870",
        "n02939185",
        "n02948072",
        "n02950826",
        "n02951358",
        "n02966193",
        "n02980441",
        "n02992529",
        "n03124170",
        "n03272010",
        "n03345487",
        "n03372029",
        "n03424325",
        "n03452741",
        "n03467068",
        "n03481172",
        "n03494278",
        "n03495258",
        "n03498962",
        "n03594945",
        "n03602883",
        "n03630383",
        "n03649909",
        "n03676483",
        "n03710193",
        "n03773504",
        "n03775071",
        "n03888257",
        "n03930630",
        "n03947888",
        "n04086273",
        "n04118538",
        "n04133789",
        "n04141076",
        "n04146614",
        "n04147183",
        "n04192698",
        "n04254680",
        "n04266014",
        "n04275548",
        "n04310018",
        "n04325704",
        "n04347754",
        "n04389033",
        "n04409515",
        "n04465501",
        "n04487394",
        "n04522168",
        "n04536866",
        "n04552348",
        "n04591713",
        "n07614500",
        "n07693725",
        "n07695742",
        "n07697313",
        "n07697537",
        "n07714571",
        "n07714990",
        "n07718472",
        "n07720875",
        "n07734744",
        "n07742313",
        "n07745940",
        "n07749582",
        "n07753275",
        "n07753592",
        "n07768694",
        "n07873807",
        "n07880968",
        "n07920052",
        "n09472597",
        "n09835506",
        "n10565667",
        "n12267677",
    ]


def get_action_dataset_train_test_new(args, train_transform, test_transform, rank_transform, train_index, valid_index):
    train_set = ImageNet200new(root=args.dset_dir, split='train', transform=train_transform, indices=None)
    valid_set = ImageNet200new(root=args.dset_dir, split='val', transform=train_transform, indices=None)
    INR_set = ImageNet200new(split='imagenet-r', root=args.dset_dir, transform=test_transform, indices=None)
    test_set = ImageNet200new(split='test', root=args.dset_dir, transform=test_transform, indices=None)
    Sketch_set = ImageNet200new(split='imagenet-sketch', root=args.dset_dir, transform=test_transform, indices=None)

    train_no_aug = ImageNet200new(split='train', root=args.dset_dir, transform=rank_transform, indices=None)
    return train_set, valid_set, test_set, train_no_aug, INR_set, Sketch_set

class ImageNet200new(ImageFolder):

    def __init__(
        self,
        root: str,
        transform,
        gap=0,
        split: str = "train",
        return_group_index=False,
        return_file_path=False,
        return_dist_shift_index=False,
        return_image_size=False,
        dist_shift_index=0,
        **kwargs: Any
    ) -> None:
        assert (split in ["train", "val", "test"] or split.startswith("imagenet"))
        self.gap = gap
        self.root = root
        self.split = split
        wnid_to_classes = self.load_meta_file(self.root)

        super().__init__(self.split_folder, transform=transform)
        self.root = root
        self.return_group_index = return_group_index
        self.return_file_path = return_file_path
        self.return_dist_shift_index = return_dist_shift_index
        self.return_image_size = return_image_size
        self.dist_shift_index = dist_shift_index

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {
            cls: idx for idx, clss in enumerate(self.classes) for cls in clss
        }
        self.return_contrastive_pairs = False

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

    def load_meta_file(self, root):
        fpath = os.path.join(root, "labels.json")
        with open(fpath, 'r') as file:
            # Load the JSON data into a Python dictionary
            data = json.load(file)

        wnid_to_classes = {}

        for _, val in data.items():
            wn_id, cls_name = val[0], val[1]
            wnid_to_classes[wn_id] = cls_name

        return wnid_to_classes

    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        data_dict = {
            "image": image,
            "label": target,
            "index": index,
        }

        if self.return_contrastive_pairs:
            rank = self.ranks[index]
            max_rank = int(self.max_rank[self.targets[index]])
            pos_rank = int(min(rank + self.gap, max_rank))
            data_dict['pos_labels'] = self.targets[index]
            data_dict['pos_rank'] = pos_rank
            neg_idx = np.random.choice(self.indices_label_rank[self.targets[index]][rank])
            try:
                pos_idx = np.random.choice(self.indices_label_rank[self.targets[index]][pos_rank])
            except ValueError:
                found = False
                for fallback_rank in range(pos_rank + 1, max_rank + 1):
                    indices = self.indices_label_rank[self.targets[index]].get(fallback_rank, [])
                    if len(indices) > 0:
                        pos_idx = np.random.choice(indices)
                        found = True
                        data_dict['pos_rank'] = fallback_rank
                        break
                if not found:
                    pos_idx = -1
            if pos_idx != -1:
                image, _ = super().__getitem__(pos_idx)
                data_dict['positive'] = image
            else:
                return None
            image, _ = super().__getitem__(neg_idx)
            data_dict['negative'] = image
            data_dict['rank'] = rank
        return data_dict

    def set_num_group_and_group_array(self, num_shortcut_cat, shortcut_label):
        self.num_group = len(self.classes) * num_shortcut_cat
        self.group_array = (
            torch.tensor(self.targets, dtype=torch.long) * num_shortcut_cat
            + shortcut_label
        )

    def get_sampling_weights(self):
        group_counts = (
            (torch.arange(self.num_group).unsqueeze(1) == self.group_array)
            .sum(1)
            .float()
        )
        group_weights = len(self) / group_counts
        weights = group_weights[self.group_array]
        return weights

class ImageNetPrecomputed():
    base_folder = "imagenet"

    def __init__(
        self,
        root: str,
        split: str = "train",
    ) -> None:
        assert split in ["train", "val"]
        with open(root, 'rb') as file:
            data = pickle.load(file)
        self.features = data['ftrs']
        self.labels = data['labels']

    def __len__(self):
        return len(self.features)

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

    def load_meta_file(self, root):
        fpath = os.path.join(root, "labels.json")
        with open(fpath, 'r') as file:
            # Load the JSON data into a Python dictionary
            data = json.load(file)

        wnid_to_classes = {}

        for _, val in data.items():
            wn_id, cls_name = val[0], val[1]
            wnid_to_classes[wn_id] = cls_name

        return wnid_to_classes

    def __getitem__(self, index: int):
        features = self.features[index]
        target = self.labels[index]
        data_dict = {
            "image": features,
            "target": target,
            "index": index
        }
        return data_dict


def get_imagenet_class_name_list():
    with open("data/imagenet/labels.txt") as f:
        lines = f.readlines()

    prefix_len = len("n02892201,")
    class_name_list = [line[prefix_len:].strip() for line in lines]
    return class_name_list