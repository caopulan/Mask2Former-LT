import random
import numpy as np
from PIL import Image

from detectron2.utils.file_io import PathManager
from detectron2.data.detection_utils import _apply_exif_orientation

__all__ = [
    "read_semseg_gt",
    "flip_human_semantic_category",
]


def read_semseg_gt(file_name):
    with PathManager.open(file_name, "rb") as f:
        gt_pil = Image.open(f)
        # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
        gt_pil = _apply_exif_orientation(gt_pil)

        gt_array = np.asarray(gt_pil)
        if len(gt_array.shape) == 3:
            assert gt_array.shape[2] == 3
            gt_array = gt_array.transpose(2, 0, 1)[0, :, :]

        return gt_array


def flip_human_semantic_category(img, gt, flip_map, prob):
    do_hflip = random.random() < prob
    if do_hflip:
        img = np.flip(img, axis=1)
        gt = gt[:, ::-1]
        gt = np.ascontiguousarray(gt)
        for ori_label, new_label in flip_map:
            left = gt == ori_label
            right = gt == new_label
            gt[left] = new_label
            gt[right] = ori_label
    return img, gt
