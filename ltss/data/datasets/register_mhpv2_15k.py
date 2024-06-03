# Copyright (c) Facebook, Inc. and its affiliates.
import json
import logging
import numpy as np
import os
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_sem_seg

MHPv2_CATEGORIES = [
    {"id": 0, "name": "Background", 'trainId': 0},
    {"id": 1, "name": "Cap/Hat", 'trainId': 1},
    {"id": 2,  "name": "Helmet", 'trainId': 2},
    {"id": 3, "name": "Face", 'trainId': 3},
    {"id": 4,  "name": "Hair", 'trainId': 4},
    {"id": 5, "name": "Left-arm", 'trainId': 5},
    {"id": 6,  "name": "Right-arm", 'trainId': 6},
    {"id": 7, "name": "Left-hand", 'trainId': 7},
    {"id": 8,  "name": "Right-hand", 'trainId': 8},
    {"id": 9, "name": "Protector", 'trainId': 9},
    {"id": 10, "name": "Bikini/bra", 'trainId': 10},
    {"id": 11, "name": "Jacket/Windbreaker/Hoodie", 'trainId': 11},
    {"id": 12, "name": "T-shirt", 'trainId': 12},
    {"id": 13, "name": "Polo-shirt", 'trainId': 13},
    {"id": 14, "name": "Sweater", 'trainId': 14},
    {"id": 15, "name": "Singlet", 'trainId': 15},
    {"id": 16, "name": "Torso-skin", 'trainId': 16},
    {"id": 17, "name": "Pants", 'trainId': 17},
    {"id": 18, "name": "Shorts/Swim-shorts", 'trainId': 18},
    {"id": 19, "name": "Skirt", 'trainId': 19},
    {"id": 20, "name": "Stockings", 'trainId': 20},
    {"id": 21, "name": "Socks", 'trainId': 21},
    {"id": 22, "name": "Left-boot", 'trainId': 22},
    {"id": 23, "name": "Right-boot", 'trainId': 23},
    {"id": 24, "name": "Left-shoe", 'trainId': 24},
    {"id": 25, "name": "Right-shoe", 'trainId': 25},
    {"id": 26, "name": "Left-highheel", 'trainId': 26},
    {"id": 27, "name": "Right-highheel", 'trainId': 27},
    {"id": 28, "name": "Left-sandal", 'trainId': 28},
    {"id": 29, "name": "Right-sandal", 'trainId': 29},
    {"id": 30, "name": "Left-leg", 'trainId': 30},
    {"id": 31, "name": "Right-leg", 'trainId': 31},
    {"id": 32, "name": "Left-foot", 'trainId': 32},
    {"id": 33, "name": "Right-foot", 'trainId': 33},
    {"id": 34, "name": "Coat", 'trainId': 34},
    {"id": 35, "name": "Dress", 'trainId': 35},
    {"id": 36, "name": "Robe", 'trainId': 36},
    {"id": 37, "name": "Jumpsuits", 'trainId': 37},
    {"id": 38, "name": "Other-full-body-clothes", 'trainId': 38},
    {"id": 39, "name": "Headwear", 'trainId': 39},
    {"id": 40, "name": "Backpack", 'trainId': 40},
    {"id": 41, "name": "Ball", 'trainId': 41},
    {"id": 42, "name": "Bats", 'trainId': 42},
    {"id": 43, "name": "Belt", 'trainId': 43},
    {"id": 44, "name": "Bottle", 'trainId': 44},
    {"id": 45, "name": "Carrybag", 'trainId': 45},
    {"id": 46, "name": "Cases", 'trainId': 46},
    {"id": 47, "name": "Sunglasses", 'trainId': 47},
    {"id": 48, "name": "Eyewear", 'trainId': 48},
    {"id": 49, "name": "Gloves", 'trainId': 49},
    {"id": 50, "name": "Scarf", 'trainId': 50},
    {"id": 51, "name": "Umbrella", 'trainId': 51},
    {"id": 52, "name": "Wallet/Purse", 'trainId': 52},
    {"id": 53, "name": "Watch", 'trainId': 53},
    {"id": 54, "name": "Wristband", 'trainId': 54},
    {"id": 55, "name": "Tie", 'trainId': 55},
    {"id": 56, "name": "Other-accessaries", 'trainId': 56},
    {"id": 57, "name": "Other-upper-body-clothes", 'trainId': 57},
    {"id": 58, "name": "Other-lower-body-clothes", 'trainId': 58},
]

MHPv2_FLIP_MAP = ((5, 6), (7, 8), (22, 23), (24, 25), (26, 27), (28, 29), (30, 31), (32, 33))


def _get_mhpv2_15k_meta():
    stuff_ids = [k["id"] for k in MHPv2_CATEGORIES]
    assert len(stuff_ids) == 59, len(stuff_ids)

    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in MHPv2_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret


def register_all_mhpv2_15k(root):
    root = os.path.join(root, "mhpv2_15k")
    meta = _get_mhpv2_15k_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("train", "images/train", "annotations/train"),
        ("val", "images/val", "annotations/val"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"mhpv2_15k_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=meta["stuff_classes"][:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            flip_map=MHPv2_FLIP_MAP
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_mhpv2_15k(_root)
