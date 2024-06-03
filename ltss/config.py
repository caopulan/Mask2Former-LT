# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


# from detectron2.config.config import CfgNode as CN


def add_ltss_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # WandB
    cfg.WANDB = CN({"ENABLED": False})
    cfg.WANDB.ENTITY = ""
    cfg.WANDB.NAME = ""
    cfg.WANDB.PROJECT = "Mask2Former-LT"

    # 1. RFS
    cfg.DATALOADER.FREQUENCY_TYPE = "image"  # image / pixel
    cfg.DATALOADER.ANN_TYPE = "seg"  # det: coco-json type annotation / seg: segmentation file.
    cfg.DATALOADER.REPEAT_FACTOR_FILE = ""  # Path to save repeat factor tensor.
    cfg.DATALOADER.REPEAT_FACTOR_MODE = "max"  # Mode to weight image with category. Support: max/mean/sum.

    # 2. Copy-Paste
    cfg.INPUT.COPY_PASTE = CN({"ENABLED": False})
    cfg.INPUT.COPY_PASTE.MODE = 'random'  # random / rare.
    cfg.INPUT.COPY_PASTE.MAX_NUM = 100
    cfg.INPUT.COPY_PASTE.RARE_FILE = ''

    # 3. Seesaw Loss
    cfg.LOSS = CN()
    cfg.LOSS.LOSS_TYPE = 'CE'  # CE / Seesaw
    cfg.LOSS.SEESAW_P = 1.2  # mitigation factor
    cfg.LOSS.SEESAW_Q = 1.0  # compensation factor

    # 4. Matcher
    cfg.MATCHER = CN()
    cfg.MATCHER.MODE = 'default'  # default / multi-match
    cfg.MATCHER.DATASET = ''  # mhpv2-lt / coco-lt / ade20k-full
    cfg.MATCHER.T = 0.
    cfg.MATCHER.S = 2.
