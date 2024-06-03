# Copyright (c) Facebook, Inc. and its affiliates.
import json
import logging
import numpy as np
import os
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_sem_seg

# total image num: 6931
# categories num: 59
# number classes of frequent, common, rare (image frequency): 36, 12, 11
# number classes of frequent, common, rare (pixel frequency): 36, 12, 11
# ------------------------------------------------------------------------
#              |     min    |    rare    |    common    |    frequent    |
# ------------------------------------------------------------------------
#   image num  |      1     |     16     |      44      |      6931      |
# ------------------------------------------------------------------------
#   image freq | 0.00014427 | 0.00230846 |  0.00634829  |       1.0      |
# ------------------------------------------------------------------------
#   pixel num  | 0.00460480 | 0.23622672 |  0.6901806   |    4279.453    |
# ------------------------------------------------------------------------
#   pixel freq | 0.00000066 | 0.00003408 |  0.0000995  |   0.617436589  |
# ------------------------------------------------------------------------
# ps. freq = label num / total image num

mhpv2_image_num = 6931

MHPv2_CATEGORIES = [
    {'id': 0, 'name': 'Background', 'trainId': 0, 'image_count': 6931.0, 'pixel_count': 4279.453,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 1, 'name': 'Cap/Hat', 'trainId': 1, 'image_count': 740.0, 'pixel_count': 10.815641,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 2, 'name': 'Helmet', 'trainId': 2, 'image_count': 8.0, 'pixel_count': 0.13745834, 'image_frequency': 'rare',
     'pixel_frequency': 'rare'},
    {'id': 3, 'name': 'Face', 'trainId': 3, 'image_count': 6906.0, 'pixel_count': 242.69102,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 4, 'name': 'Hair', 'trainId': 4, 'image_count': 6894.0, 'pixel_count': 243.51534,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 5, 'name': 'Left-arm', 'trainId': 5, 'image_count': 3979.0, 'pixel_count': 59.379906,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 6, 'name': 'Right-arm', 'trainId': 6, 'image_count': 4031.0, 'pixel_count': 62.588593,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 7, 'name': 'Left-hand', 'trainId': 7, 'image_count': 5904.0, 'pixel_count': 41.769737,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 8, 'name': 'Right-hand', 'trainId': 8, 'image_count': 6024.0, 'pixel_count': 43.433514,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 9, 'name': 'Protector', 'trainId': 9, 'image_count': 11.0, 'pixel_count': 0.3038193,
     'image_frequency': 'rare', 'pixel_frequency': 'common'},
    {'id': 10, 'name': 'Bikini/bra', 'trainId': 10, 'image_count': 15.0, 'pixel_count': 0.46483794,
     'image_frequency': 'rare', 'pixel_frequency': 'common'},
    {'id': 11, 'name': 'Jacket/Windbreaker/Hoodie', 'trainId': 11, 'image_count': 2843.0, 'pixel_count': 452.9208,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 12, 'name': 'T-shirt', 'trainId': 12, 'image_count': 4934.0, 'pixel_count': 503.77356,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 13, 'name': 'Polo-shirt', 'trainId': 13, 'image_count': 38.0, 'pixel_count': 3.6304812,
     'image_frequency': 'common', 'pixel_frequency': 'frequent'},
    {'id': 14, 'name': 'Sweater', 'trainId': 14, 'image_count': 847.0, 'pixel_count': 68.62194,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 15, 'name': 'Singlet', 'trainId': 15, 'image_count': 2366.0, 'pixel_count': 82.274284,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 16, 'name': 'Torso-skin', 'trainId': 16, 'image_count': 6323.0, 'pixel_count': 86.1678,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 17, 'name': 'Pants', 'trainId': 17, 'image_count': 4822.0, 'pixel_count': 316.041,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 18, 'name': 'Shorts/Swim-shorts', 'trainId': 18, 'image_count': 656.0, 'pixel_count': 21.02937,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 19, 'name': 'Skirt', 'trainId': 19, 'image_count': 552.0, 'pixel_count': 22.358498,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 20, 'name': 'Stockings', 'trainId': 20, 'image_count': 34.0, 'pixel_count': 0.6901806,
     'image_frequency': 'common', 'pixel_frequency': 'frequent'},
    {'id': 21, 'name': 'Socks', 'trainId': 21, 'image_count': 47.0, 'pixel_count': 0.32260558,
     'image_frequency': 'frequent', 'pixel_frequency': 'common'},
    {'id': 22, 'name': 'Left-boot', 'trainId': 22, 'image_count': 32.0, 'pixel_count': 0.32287574,
     'image_frequency': 'common', 'pixel_frequency': 'common'},
    {'id': 23, 'name': 'Right-boot', 'trainId': 23, 'image_count': 36.0, 'pixel_count': 0.3393923,
     'image_frequency': 'common', 'pixel_frequency': 'common'},
    {'id': 24, 'name': 'Left-shoe', 'trainId': 24, 'image_count': 1543.0, 'pixel_count': 9.432695,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 25, 'name': 'Right-shoe', 'trainId': 25, 'image_count': 1546.0, 'pixel_count': 9.277901,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 26, 'name': 'Left-highheel', 'trainId': 26, 'image_count': 99.0, 'pixel_count': 0.5912717,
     'image_frequency': 'frequent', 'pixel_frequency': 'common'},
    {'id': 27, 'name': 'Right-highheel', 'trainId': 27, 'image_count': 64.0, 'pixel_count': 0.23622672,
     'image_frequency': 'frequent', 'pixel_frequency': 'common'},
    {'id': 28, 'name': 'Left-sandal', 'trainId': 28, 'image_count': 41.0, 'pixel_count': 0.121689096,
     'image_frequency': 'common', 'pixel_frequency': 'rare'},
    {'id': 29, 'name': 'Right-sandal', 'trainId': 29, 'image_count': 41.0, 'pixel_count': 0.11595454,
     'image_frequency': 'common', 'pixel_frequency': 'rare'},
    {'id': 30, 'name': 'Left-leg', 'trainId': 30, 'image_count': 1459.0, 'pixel_count': 16.763313,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 31, 'name': 'Right-leg', 'trainId': 31, 'image_count': 1483.0, 'pixel_count': 17.06767,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 32, 'name': 'Left-foot', 'trainId': 32, 'image_count': 668.0, 'pixel_count': 2.3844607,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 33, 'name': 'Right-foot', 'trainId': 33, 'image_count': 657.0, 'pixel_count': 2.3088884,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 34, 'name': 'Coat', 'trainId': 34, 'image_count': 1040.0, 'pixel_count': 110.82123,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 35, 'name': 'Dress', 'trainId': 35, 'image_count': 1342.0, 'pixel_count': 131.4145,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 36, 'name': 'Robe', 'trainId': 36, 'image_count': 44.0, 'pixel_count': 5.9627824,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 37, 'name': 'Jumpsuits', 'trainId': 37, 'image_count': 10.0, 'pixel_count': 0.53899044,
     'image_frequency': 'rare', 'pixel_frequency': 'common'},
    {'id': 38, 'name': 'Other-full-body-clothes', 'trainId': 38, 'image_count': 36.0, 'pixel_count': 3.445525,
     'image_frequency': 'common', 'pixel_frequency': 'frequent'},
    {'id': 39, 'name': 'Headwear', 'trainId': 39, 'image_count': 39.0, 'pixel_count': 0.43621355,
     'image_frequency': 'common', 'pixel_frequency': 'common'},
    {'id': 40, 'name': 'Backpack', 'trainId': 40, 'image_count': 16.0, 'pixel_count': 0.24105568,
     'image_frequency': 'common', 'pixel_frequency': 'common'},
    {'id': 41, 'name': 'Ball', 'trainId': 41, 'image_count': 9.0, 'pixel_count': 0.13079028, 'image_frequency': 'rare',
     'pixel_frequency': 'rare'},
    {'id': 42, 'name': 'Bats', 'trainId': 42, 'image_count': 3.0, 'pixel_count': 0.03406132, 'image_frequency': 'rare',
     'pixel_frequency': 'rare'},
    {'id': 43, 'name': 'Belt', 'trainId': 43, 'image_count': 992.0, 'pixel_count': 4.475071,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 44, 'name': 'Bottle', 'trainId': 44, 'image_count': 15.0, 'pixel_count': 0.115113564,
     'image_frequency': 'rare', 'pixel_frequency': 'rare'},
    {'id': 45, 'name': 'Carrybag', 'trainId': 45, 'image_count': 467.0, 'pixel_count': 10.121248,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 46, 'name': 'Cases', 'trainId': 46, 'image_count': 7.0, 'pixel_count': 0.117773265,
     'image_frequency': 'rare', 'pixel_frequency': 'rare'},
    {'id': 47, 'name': 'Sunglasses', 'trainId': 47, 'image_count': 63.0, 'pixel_count': 0.18657403,
     'image_frequency': 'frequent', 'pixel_frequency': 'rare'},
    {'id': 48, 'name': 'Eyewear', 'trainId': 48, 'image_count': 1082.0, 'pixel_count': 2.874794,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 49, 'name': 'Gloves', 'trainId': 49, 'image_count': 24.0, 'pixel_count': 0.41750056,
     'image_frequency': 'common', 'pixel_frequency': 'common'},
    {'id': 50, 'name': 'Scarf', 'trainId': 50, 'image_count': 40.0, 'pixel_count': 1.1167985,
     'image_frequency': 'common', 'pixel_frequency': 'frequent'},
    {'id': 51, 'name': 'Umbrella', 'trainId': 51, 'image_count': 1.0, 'pixel_count': 0.0046048085,
     'image_frequency': 'rare', 'pixel_frequency': 'rare'},
    {'id': 52, 'name': 'Wallet/Purse', 'trainId': 52, 'image_count': 10.0, 'pixel_count': 0.065467335,
     'image_frequency': 'rare', 'pixel_frequency': 'rare'},
    {'id': 53, 'name': 'Watch', 'trainId': 53, 'image_count': 1026.0, 'pixel_count': 1.0202947,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 54, 'name': 'Wristband', 'trainId': 54, 'image_count': 43.0, 'pixel_count': 0.050062217,
     'image_frequency': 'common', 'pixel_frequency': 'rare'},
    {'id': 55, 'name': 'Tie', 'trainId': 55, 'image_count': 1293.0, 'pixel_count': 10.199624,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 56, 'name': 'Other-accessaries', 'trainId': 56, 'image_count': 1492.0, 'pixel_count': 10.316427,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 57, 'name': 'Other-upper-body-clothes', 'trainId': 57, 'image_count': 561.0, 'pixel_count': 35.241367,
     'image_frequency': 'frequent', 'pixel_frequency': 'frequent'},
    {'id': 58, 'name': 'Other-lower-body-clothes', 'trainId': 58, 'image_count': 13.0, 'pixel_count': 0.3147636,
     'image_frequency': 'rare', 'pixel_frequency': 'common'},
]

MHPv2_FLIP_MAP = ((5, 6), (7, 8), (22, 23), (24, 25), (26, 27), (28, 29), (30, 31), (32, 33))


def _get_mhpv2_lt_7k_meta():
    stuff_ids = [k["id"] for k in MHPv2_CATEGORIES]
    assert len(stuff_ids) == 59, len(stuff_ids)

    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in MHPv2_CATEGORIES]
    image_count = [k["image_count"] for k in MHPv2_CATEGORIES]
    image_frequency = [k["image_frequency"] for k in MHPv2_CATEGORIES]
    pixel_count = [k["pixel_count"] for k in MHPv2_CATEGORIES]
    pixel_frequency = [k["pixel_frequency"] for k in MHPv2_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "image_count": image_count,
        "image_frequency": image_frequency,
        "pixel_count": pixel_count,
        "pixel_frequency": pixel_frequency,
    }
    return ret


def register_all_mhpv2_lt_7k(root):
    root = os.path.join(root, "mhpv2_lt_7k")
    meta = _get_mhpv2_lt_7k_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("train", "images/train", "annotations/train"),
        ("val", "images/val", "annotations/val"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"mhpv2_lt_7k_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            frequency=meta,
            stuff_classes=meta["stuff_classes"][:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="lt_sem_seg",
            ignore_label=255,
            flip_map=MHPv2_FLIP_MAP
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_mhpv2_lt_7k(_root)
