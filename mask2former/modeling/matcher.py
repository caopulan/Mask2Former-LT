# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import numpy as np
import math
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from detectron2.projects.point_rend.point_features import point_sample
from ltss.data.datasets.register_mhpv2_lt_7k import MHPv2_CATEGORIES, mhpv2_image_num
from ltss.data.datasets.register_coco_stuff_lt_40k import COCO_CATEGORIES, coco_image_num
from ltss.data.datasets.register_ade20k_full import ADE20K_SEM_SEG_FULL_CATEGORIES, ade20k_full_image_num


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0, loss_type='CE',
            match_method='default', dataset_name='', t=None, s=None):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points
        self.loss_type = loss_type

        self.match_method = match_method
        if match_method == 'multi-match':
            dataset_dict = {
                'mhpv2-lt': {'categories': MHPv2_CATEGORIES, 'image_num': mhpv2_image_num},
                'coco-lt': {'categories': COCO_CATEGORIES, 'image_num': coco_image_num},
                'ade20k-full': {'categories': ADE20K_SEM_SEG_FULL_CATEGORIES, 'image_num': ade20k_full_image_num},
            }[dataset_name]
            self.class_query_num = self.get_category_query_num(dataset_dict, t, s)
        else:
            self.class_query_num = None

    def get_category_query_num(self, dataset_dict, t, s):
        # 1. Calculate categories frequency
        categories = dataset_dict['categories']
        image_num = dataset_dict['image_num']
        class_frequency = torch.zeros(len(categories))
        for cat_dict in categories:
            class_frequency[cat_dict['trainId']] = cat_dict['image_count'] / image_num

        # 2. Calculate query num
        # q = max(1, s*sqrt(t/c))
        query_num = (s * (t / class_frequency).sqrt()).clamp_min(1.)
        query_num = [math.ceil(i.item()) for i in query_num]

        return query_num

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):
            if self.loss_type == 'Seesaw':
                objectness = outputs["pred_logits"][b][:, -2:].softmax(dim=-1)
                out_prob = outputs["pred_logits"][b][:, :-2].softmax(-1)
                out_prob = torch.cat([out_prob * objectness[:, :1], objectness[:, 1:]], dim=-1)
            else:
                out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = targets[b]["labels"]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)

            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
            
            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()

            if self.match_method == 'rare_first':
                # match rare category first
                tgt_rare = [tgt_id.item() in self.rare_classes for tgt_id in tgt_ids]
                # if any(tgt_rare):
                #     print()
                rare_indices = linear_sum_assignment(C[:, tgt_rare])
                rare_cat_inds = np.arange(C.shape[1])[tgt_rare][rare_indices[1]]
                unmatch_query_id = torch.Tensor([i for i in range(C.shape[0]) if i not in rare_indices[0]]).to(torch.int64)
                unmatch_target_id = torch.Tensor([i for i in range(C.shape[1]) if i not in rare_cat_inds]).to(torch.int64)

                other_indices = linear_sum_assignment(C[unmatch_query_id][:, unmatch_target_id])

                indice = (
                    np.concatenate([rare_indices[0], unmatch_query_id[other_indices[0]].numpy()]),
                    np.concatenate([rare_cat_inds, unmatch_target_id[other_indices[1]].numpy()])
                )

                indices.append(indice)
            elif self.match_method == 'groups':
                tgt_rare = [tgt_id.item() in self.rare_classes for tgt_id in tgt_ids]
                tgt_common = [tgt_id.item() in self.common_classes for tgt_id in tgt_ids]
                tgt_frequent = [not (a | b) for a, b in zip(tgt_rare, tgt_common)]
                # if any(tgt_rare):
                #     print()
                frequent_indices = linear_sum_assignment(C[:self.groups[0], tgt_frequent])
                common_indices = linear_sum_assignment(C[self.groups[0]:self.groups[0]+self.groups[1], tgt_common])
                rare_indices = linear_sum_assignment(C[-self.groups[-1]:, tgt_rare])

                frequent_cat_inds = np.arange(C.shape[1])[tgt_frequent][frequent_indices[1]]
                common_cat_inds = np.arange(C.shape[1])[tgt_common][common_indices[1]]
                rare_cat_inds = np.arange(C.shape[1])[tgt_rare][rare_indices[1]]

                indice = (
                    np.concatenate([
                        frequent_indices[0],
                        self.groups[0] + common_indices[0],
                        self.groups[0] + self.groups[1] + rare_indices[0]
                    ]),
                    np.concatenate([frequent_cat_inds, common_cat_inds, rare_cat_inds])
                )
                indices.append(indice)
            elif self.match_method == 'rare-one-to-one':
                tgt_rare = [tgt_id.item() in self.rare_classes for tgt_id in tgt_ids]
                tgt_common = [tgt_id.item() in self.common_classes for tgt_id in tgt_ids]
                tgt_frequent = [not (a | b) for a, b in zip(tgt_rare, tgt_common)]
                # if any(tgt_rare):
                #     print()
                frequent_indices = linear_sum_assignment(C[:self.groups[0], tgt_frequent])
                common_indices = linear_sum_assignment(C[self.groups[0]:self.groups[0] + self.groups[1], tgt_common])
                # rare_indices = linear_sum_assignment(C[-self.groups[-1]:, tgt_rare])

                frequent_cat_inds = np.arange(C.shape[1])[tgt_frequent][frequent_indices[1]]
                common_cat_inds = np.arange(C.shape[1])[tgt_common][common_indices[1]]
                rare_cat_inds = np.arange(C.shape[1])[tgt_rare]

                rare_query_inds = [self.rare_classes.tolist().index(i) for i in tgt_ids[tgt_rare]]

                indice = (
                    np.concatenate([
                        frequent_indices[0],
                        self.groups[0] + common_indices[0],
                        self.groups[0] + self.groups[1] + np.array(rare_query_inds)
                    ]),
                    np.concatenate([frequent_cat_inds, common_cat_inds, rare_cat_inds])
                )
                indices.append(indice)
            elif self.match_method.startswith('rarex'):
                times = int(self.match_method[-1])
                ori_num = C.shape[1]
                aux_costs = []
                aux_idxs = []
                for i, idx in enumerate(tgt_ids):
                    if idx.item() in self.rare_classes:
                        for _ in range(times - 1):
                            aux_costs.append(C[:, i])
                            aux_idxs.append(i)
                if len(aux_costs) > 0:
                    C_ = torch.cat([C, torch.stack(aux_costs, dim=-1)], dim=-1)
                    indice = linear_sum_assignment(C_)
                    for i, idx in enumerate(indice[1]):
                        if idx >= ori_num:
                            indice[1][i] = aux_idxs[idx-ori_num]
                    indices.append(indice)
                else:
                    indices.append(linear_sum_assignment(C))
            elif self.match_method == 'multi-match':
                ori_num = C.shape[1]
                aux_costs = []
                aux_idxs = []
                for i, idx in enumerate(tgt_ids):
                    times = self.class_query_num[idx.item()]
                    for _ in range(times - 1):
                        aux_costs.append(C[:, i])
                        aux_idxs.append(i)
                if len(aux_costs) > 0:
                    C_ = torch.cat([C, torch.stack(aux_costs, dim=-1)], dim=-1)
                    indice = linear_sum_assignment(C_)
                    for i, idx in enumerate(indice[1]):
                        if idx >= ori_num:
                            indice[1][i] = aux_idxs[idx - ori_num]
                    indices.append(indice)
                else:
                    indices.append(linear_sum_assignment(C))
            else:
                indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
