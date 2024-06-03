import numpy as np
import random

import torch
import torch.utils.data as data

from detectron2.structures.instances import Instances

class CopyPaste(data.Dataset):
    """Simple Copy-Paste is a Strong Data Augmentation Method for Instance
    Segmentation The simple copy-paste transform steps are as follows:

    1. The destination image is already resized with aspect ratio kept,
       cropped and padded.
    2. Randomly select a source image, which is also already resized
       with aspect ratio kept, cropped and padded in a similar way
       as the destination image.
    3. Randomly select some objects from the source image.
    4. Paste these source objects to the destination image directly,
       due to the source and destination image have the same size.
    5. Update object masks of the destination image, for some origin objects
       may be occluded.
    6. Generate bboxes from the updated destination masks and
       filter some objects which are totally occluded, and adjust bboxes
       which are partly occluded.
    7. Append selected source bboxes, masks, and labels.

    Args:
        max_num_pasted (int): The maximum number of pasted objects.
            Default: 100.
        bbox_occluded_thr (int): The threshold of occluded bbox.
            Default: 10.
        mask_occluded_thr (int): The threshold of occluded mask.
            Default: 300.
        selected (bool): Whether select objects or not. If select is False,
            all objects of the source image will be pasted to the
            destination image.
            Default: True.
    """

    def __init__(
        self,
        mode,
        max_num_pasted=100,
        rare_classes='',
    ):
        self.mode = mode
        self.dataset = None
        self.max_num_pasted = max_num_pasted
        if self.mode == 'rare':
            self.rare_classes = np.load(rare_classes)
        else:
            self.rare_classes = None

    def set_dataset(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def get_index(self):
        """Call function to collect indexes.s.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.
        Returns:
            list: Indexes.
        """
        return random.randint(0, len(self.dataset) - 1)

    def __getitem__(self, index):
        """Call function to make a copy-paste of image.

        Args:
            results (dict): Result dict.
        Returns:
            dict: Result dict with copy-paste transformed.
        """
        src_data = self.dataset[self.get_index()]
        dst_data = self.dataset[index]

        selected_instances = self._select_object(src_data)
        if selected_instances is not None:
            return self._copy_paste(dst_data, src_data, selected_instances)
        else:
            return dst_data

    def _select_object(self, src_data):
        """Select some objects from the source results."""
        max_num_pasted = min(self.max_num_pasted, len(src_data['instances']))
        if max_num_pasted != 0:
            if self.mode == 'random':
                num_pasted = np.random.randint(0, max_num_pasted)
                selected_inds = np.random.choice(
                    len(src_data['instances']), size=num_pasted, replace=False)
            elif self.mode == 'rare':
                src_classes = src_data['instances'].get('gt_classes')
                selected_inds = [i for i, inds in enumerate(src_classes) if inds in self.rare_classes]
            selected_instances = src_data['instances'][selected_inds]
            return selected_instances
        else:
            print('src_data no instance.')
            return None

    def _copy_paste(self, dst_results, src_data, selected_instances):
        """CopyPaste transform function.

        Args:
            dst_results (dict): Result dict of the destination image.
            src_results (dict): Result dict of the source image.
        Returns:
            dict: Updated result dict.
        """
        selected_masks = selected_instances.get('gt_masks')
        selected_classes = selected_instances.get('gt_classes')
        update_mask = selected_masks.any(dim=0)

        # update annotation
        dst_classes = dst_results['instances'].get('gt_classes')
        dst_masks = dst_results['instances'].get('gt_masks')
        dst_classes_idx = {v.item(): idx for idx, v in enumerate(dst_classes)}
        for mask, class_id in zip(selected_masks, selected_classes):
            # updtae sem_seg
            dst_results['sem_seg'][mask] = class_id.item()
            # update instances
            dst_masks[:, mask] = False
            if class_id in dst_classes:
                idx = dst_classes_idx[class_id.item()]
                dst_masks[idx] = dst_masks[idx] | mask
            else:
                dst_masks = torch.cat([dst_masks, mask.unsqueeze(dim=0)], dim=0)
                dst_classes = torch.cat([dst_classes, class_id.unsqueeze(dim=0)], dim=0)

        # del empty mask
        idxs = []
        for idx in range(dst_masks.shape[0]):
            if dst_masks[idx].any():
                idxs.append(idx)
        dst_masks = dst_masks[idxs]
        dst_classes = dst_classes[idxs]

        # updtae image
        dst_results['image'][:, update_mask] = src_data['image'][:, update_mask]

        pasted_instances = Instances(image_size=dst_results['instances'].image_size)
        pasted_instances.set('gt_masks', dst_masks)
        pasted_instances.set('gt_classes', dst_classes)
        dst_results['instances'] = pasted_instances

        return dst_results
