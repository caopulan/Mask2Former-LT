# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog # , build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.config import LazyCall as L
from detectron2.data.common import AspectRatioGroupedDataset, DatasetFromList, MapDataset, ToIterableDataset

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

# MaskFormer
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)

# LTSS
from ltss import (
    add_ltss_config,
    RepeatFactorTrainingSampler_Seg
)
from ltss.data.build import build_detection_train_loader
from ltss.data.dataset_mappers import MaskFormerSemanticDatasetMapperNoAug

from train_net import setup
from detectron2.data.build import (
    get_detection_dataset_dicts,
)


# def setup(args):
#     """
#     Create configs and perform basic setups.
#     """
#     cfg = get_cfg()
#     # for poly lr schedule
#     add_deeplab_config(cfg)
#     add_maskformer2_config(cfg)
#     add_ltss_config(cfg)
#     cfg.merge_from_file(args.config_file)
#     cfg.merge_from_list(args.opts)
#     cfg.freeze()
#     default_setup(cfg, args)
#     # Setup logger for "mask_former" module
#     setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
#     return cfg


def main(args):
    cfg = setup(args)

    if cfg.DATALOADER.REPEAT_FACTOR_MODE == 'max':
        mode = ''
    else:
        mode = '_' + cfg.DATALOADER.REPEAT_FACTOR_MODE
    rf_file = os.path.join(
        f'{cfg.DATALOADER.REPEAT_FACTOR_FILE}_{cfg.DATALOADER.FREQUENCY_TYPE}_{cfg.DATALOADER.REPEAT_THRESHOLD}{mode}.pt')

    # TODO: Add more dataset
    if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
        dataset = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )
        dataset = MapDataset(dataset, MaskFormerSemanticDatasetMapperNoAug(cfg, True))
    else:
        raise ValueError("TODO: Add more dataset")

    repeat_factors = RepeatFactorTrainingSampler_Seg.repeat_factors_from_category_frequency(
        dataset, cfg.DATALOADER.REPEAT_THRESHOLD, cfg.DATALOADER.FREQUENCY_TYPE, cfg.DATALOADER.ANN_TYPE, cfg.DATALOADER.REPEAT_FACTOR_MODE
    )
    torch.save(repeat_factors, rf_file)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    # print("Command Line Args:", args)
    main(args)
    # launch(
    #     main,
    #     args.num_gpus,
    #     num_machines=args.num_machines,
    #     machine_rank=args.machine_rank,
    #     dist_url=args.dist_url,
    #     args=(args,),
    # )
