import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
from typing import Optional, Union
import pycocotools.mask as mask_util
import torch
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager
from detectron2.evaluation.evaluator import DatasetEvaluator

from mask2former.evaluation.utils import load_image_into_numpy_array


class LTSemSegEvaluator(DatasetEvaluator):
    """
    Evaluate long-tailed semantic segmentation metrics.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        sem_seg_loading_fn=load_image_into_numpy_array,
        num_classes=None,
        ignore_label=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            sem_seg_loading_fn: function to read sem seg file and load into numpy array.
                Default provided, but projects can customize.
            num_classes, ignore_label: deprecated argument
        """
        self._logger = logging.getLogger("detectron2")
        if num_classes is not None:
            self._logger.warn(
                "LTSemSegEvaluator(num_classes) is deprecated! It should be obtained from metadata."
            )
        if ignore_label is not None:
            self._logger.warn(
                "LTSemSegEvaluator(ignore_label) is deprecated! It should be obtained from metadata."
            )
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        self.input_file_to_gt_file = {
            dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }

        meta = MetadataCatalog.get(dataset_name)
        self._image_frequency = meta.frequency["image_frequency"]
        self._pixel_frequency = meta.frequency["pixel_frequency"]

        # Dict that maps contiguous training ids to COCO category ids
        try:
            c2d = meta.stuff_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None
        self._class_names = meta.stuff_classes
        self.sem_seg_loading_fn = sem_seg_loading_fn
        self._num_classes = len(meta.stuff_classes)
        if num_classes is not None:
            assert self._num_classes == num_classes, f"{self._num_classes} != {num_classes}"
        self._ignore_label = ignore_label if ignore_label is not None else meta.ignore_label

    def reset(self):
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        for input, output in zip(inputs, outputs):
            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device)
            pred = np.array(output, dtype=np.int)
            gt_filename = self.input_file_to_gt_file[input["file_name"]]
            gt = self.sem_seg_loading_fn(gt_filename, dtype=np.int)

            gt[gt == self._ignore_label] = self._num_classes

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input["file_name"]))

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "lt_sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        acc = np.full(self._num_classes, np.nan, dtype=np.float)
        iou = np.full(self._num_classes, np.nan, dtype=np.float)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res["IoU-{}".format(name)] = 100 * iou[i]
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res["ACC-{}".format(name)] = 100 * acc[i]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "lt_sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)

        results = OrderedDict({"sem_seg": res, "LTSS": dict(), "Test":
            {"mIoU": res["mIoU"], "fwIoU": res["fwIoU"], "mACC": res["mACC"], "pACC": res["pACC"]}})

        self._logger.info(results)

        # compute image frequency iou
        im_f = [True if k == "frequent" else False for k in self._image_frequency]
        im_c = [True if k == "common" else False for k in self._image_frequency]
        im_r = [True if k == "rare" else False for k in self._image_frequency]

        im_f_acc_valid = [i & j for i, j in zip(acc_valid, im_f)]
        im_f_iou_valid = [i & j for i, j in zip(iou_valid, im_f)]
        im_f_miou = np.sum(iou[im_f_acc_valid]) / np.sum(im_f_iou_valid)

        im_c_acc_valid = [i & j for i, j in zip(acc_valid, im_c)]
        im_c_iou_valid = [i & j for i, j in zip(iou_valid, im_c)]
        im_c_miou = np.sum(iou[im_c_acc_valid]) / np.sum(im_c_iou_valid)

        im_r_acc_valid = [i & j for i, j in zip(acc_valid, im_r)]
        im_r_iou_valid = [i & j for i, j in zip(iou_valid, im_r)]
        im_r_miou = np.sum(iou[im_r_acc_valid]) / np.sum(im_r_iou_valid)
        self._logger.info("========================= image frequency iou =========================")
        self._logger.info({"mIoU@r": im_r_miou, "mIoU@c": im_c_miou, "mIoU@f": im_f_miou})

        results['LTSS']['mIoU-image@r'] = im_r_miou
        results['LTSS']['mIoU-image@c'] = im_c_miou
        results['LTSS']['mIoU-image@f'] = im_f_miou

        # compute pixel frequency iou
        px_f = [True if k == "frequent" else False for k in self._pixel_frequency]
        px_c = [True if k == "common" else False for k in self._pixel_frequency]
        px_r = [True if k == "rare" else False for k in self._pixel_frequency]

        px_f_acc_valid = [i & j for i, j in zip(acc_valid, px_f)]
        px_f_iou_valid = [i & j for i, j in zip(iou_valid, px_f)]
        px_f_miou = np.sum(iou[px_f_acc_valid]) / np.sum(px_f_iou_valid)

        px_c_acc_valid = [i & j for i, j in zip(acc_valid, px_c)]
        px_c_iou_valid = [i & j for i, j in zip(iou_valid, px_c)]
        px_c_miou = np.sum(iou[px_c_acc_valid]) / np.sum(px_c_iou_valid)

        px_r_acc_valid = [i & j for i, j in zip(acc_valid, px_r)]
        px_r_iou_valid = [i & j for i, j in zip(iou_valid, px_r)]
        px_r_miou = np.sum(iou[px_r_acc_valid]) / np.sum(px_r_iou_valid)
        self._logger.info("========================= pixel frequency iou =========================")
        self._logger.info({"mIoU@r": px_r_miou, "mIoU@c": px_c_miou, "mIoU@f": px_f_miou})

        results['LTSS']['mIoU-pixel@r'] = px_r_miou
        results['LTSS']['mIoU-pixel@c'] = px_c_miou
        results['LTSS']['mIoU-pixel@f'] = px_f_miou
        return results

    def encode_json_sem_seg(self, sem_seg, input_file_name):
        """
        Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
        See http://cocodataset.org/#format-results
        """
        json_list = []
        for label in np.unique(sem_seg):
            if self._contiguous_id_to_dataset_id is not None:
                assert (
                    label in self._contiguous_id_to_dataset_id
                ), "Label {} is not in the metadata info for {}".format(label, self._dataset_name)
                dataset_id = self._contiguous_id_to_dataset_id[label]
            else:
                dataset_id = int(label)
            mask = (sem_seg == label).astype(np.uint8)
            mask_rle = mask_util.encode(np.array(mask[:, :, None], order="F"))[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            json_list.append(
                {"file_name": input_file_name, "category_id": dataset_id, "segmentation": mask_rle}
            )
        return json_list
