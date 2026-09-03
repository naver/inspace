# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Loss functions for 3D bounding box estimation.

Includes:
- Hungarian matching (bipartite assignment)
- L1 regression loss for center and size
- Rotation loss (angular)
- Focal loss for confidence
- 3D GIoU / DIoU loss (axis-aligned approximation)
- Auxiliary decoder loss support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def bbox3d_iou(boxes1, boxes2):
    """
    Compute pairwise 3D IoU between two sets of axis-aligned bounding boxes.

    Args:
        boxes1: [N, 6] (cx, cy, cz, sx, sy, sz) - center + full extents
        boxes2: [M, 6] (cx, cy, cz, sx, sy, sz) - center + full extents

    Returns:
        iou: [N, M] IoU matrix
    """
    # Convert to min/max format
    min1 = boxes1[:, :3] - boxes1[:, 3:6] / 2  # [N, 3]
    max1 = boxes1[:, :3] + boxes1[:, 3:6] / 2  # [N, 3]
    min2 = boxes2[:, :3] - boxes2[:, 3:6] / 2  # [M, 3]
    max2 = boxes2[:, :3] + boxes2[:, 3:6] / 2  # [M, 3]

    # Intersection
    inter_min = torch.max(min1.unsqueeze(1), min2.unsqueeze(0))  # [N, M, 3]
    inter_max = torch.min(max1.unsqueeze(1), max2.unsqueeze(0))  # [N, M, 3]
    inter_size = (inter_max - inter_min).clamp(min=0)  # [N, M, 3]
    inter_vol = inter_size.prod(-1)  # [N, M]

    # Volumes
    vol1 = boxes1[:, 3:6].prod(-1)  # [N]
    vol2 = boxes2[:, 3:6].prod(-1)  # [M]

    # Union
    union_vol = vol1.unsqueeze(1) + vol2.unsqueeze(0) - inter_vol  # [N, M]

    iou = inter_vol / union_vol.clamp(min=1e-8)
    return iou


def bbox3d_giou(boxes1, boxes2):
    """
    Compute pairwise 3D Generalized IoU.

    Args:
        boxes1: [N, 6] (cx, cy, cz, sx, sy, sz)
        boxes2: [M, 6] (cx, cy, cz, sx, sy, sz)

    Returns:
        giou: [N, M] GIoU matrix in [-1, 1]
    """
    min1 = boxes1[:, :3] - boxes1[:, 3:6] / 2
    max1 = boxes1[:, :3] + boxes1[:, 3:6] / 2
    min2 = boxes2[:, :3] - boxes2[:, 3:6] / 2
    max2 = boxes2[:, :3] + boxes2[:, 3:6] / 2

    # Intersection
    inter_min = torch.max(min1.unsqueeze(1), min2.unsqueeze(0))
    inter_max = torch.min(max1.unsqueeze(1), max2.unsqueeze(0))
    inter_size = (inter_max - inter_min).clamp(min=0)
    inter_vol = inter_size.prod(-1)

    vol1 = boxes1[:, 3:6].prod(-1)
    vol2 = boxes2[:, 3:6].prod(-1)
    union_vol = vol1.unsqueeze(1) + vol2.unsqueeze(0) - inter_vol

    iou = inter_vol / union_vol.clamp(min=1e-8)

    # Enclosing box
    enclose_min = torch.min(min1.unsqueeze(1), min2.unsqueeze(0))
    enclose_max = torch.max(max1.unsqueeze(1), max2.unsqueeze(0))
    enclose_size = (enclose_max - enclose_min).clamp(min=0)
    enclose_vol = enclose_size.prod(-1)

    giou = iou - (enclose_vol - union_vol) / enclose_vol.clamp(min=1e-8)
    return giou


def bbox3d_diou(boxes1, boxes2):
    """
    Compute pairwise 3D Distance-IoU (DIoU).

    DIoU = IoU - (d^2 / c^2)
    where d = center distance, c = enclosing box diagonal.
    Explicitly penalizes center displacement for tighter bbox regression.

    Args:
        boxes1: [N, 6] (cx, cy, cz, sx, sy, sz)
        boxes2: [M, 6] (cx, cy, cz, sx, sy, sz)

    Returns:
        diou: [N, M] DIoU matrix in [-1, 1]
    """
    min1 = boxes1[:, :3] - boxes1[:, 3:6] / 2
    max1 = boxes1[:, :3] + boxes1[:, 3:6] / 2
    min2 = boxes2[:, :3] - boxes2[:, 3:6] / 2
    max2 = boxes2[:, :3] + boxes2[:, 3:6] / 2

    # Intersection
    inter_min = torch.max(min1.unsqueeze(1), min2.unsqueeze(0))
    inter_max = torch.min(max1.unsqueeze(1), max2.unsqueeze(0))
    inter_size = (inter_max - inter_min).clamp(min=0)
    inter_vol = inter_size.prod(-1)

    vol1 = boxes1[:, 3:6].prod(-1)
    vol2 = boxes2[:, 3:6].prod(-1)
    union_vol = vol1.unsqueeze(1) + vol2.unsqueeze(0) - inter_vol

    iou = inter_vol / union_vol.clamp(min=1e-8)

    # Center distance squared
    center1 = boxes1[:, :3]  # [N, 3]
    center2 = boxes2[:, :3]  # [M, 3]
    d_sq = ((center1.unsqueeze(1) - center2.unsqueeze(0)) ** 2).sum(-1)  # [N, M]

    # Enclosing box diagonal squared
    enclose_min = torch.min(min1.unsqueeze(1), min2.unsqueeze(0))
    enclose_max = torch.max(max1.unsqueeze(1), max2.unsqueeze(0))
    c_sq = ((enclose_max - enclose_min) ** 2).sum(-1)  # [N, M]

    diou = iou - d_sq / c_sq.clamp(min=1e-8)
    return diou


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """
    Focal loss for binary classification.

    Args:
        logits: [N] raw logits (before sigmoid)
        targets: [N] binary targets {0, 1}
        alpha: Balancing factor
        gamma: Focusing parameter

    Returns:
        loss: scalar
    """
    probs = logits.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
    p_t = probs * targets + (1 - probs) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t) ** gamma * ce_loss
    return loss.mean()


@torch.no_grad()
def hungarian_match(pred_bboxes, gt_bboxes, gt_mask, cost_weights=None):
    """
    Hungarian matching between predictions and ground truth.

    Args:
        pred_bboxes: dict with pred_centers [B, Q, 3], pred_sizes [B, Q, 3],
                     pred_confidences [B, Q, 1]
        gt_bboxes: [B, M, 7] ground truth (cx, cy, cz, sx, sy, sz, rot)
        gt_mask: [B, M] bool mask for valid GT boxes
        cost_weights: dict with center, size, confidence weights

    Returns:
        indices: list of (pred_indices, gt_indices) tuples per batch element
    """
    if cost_weights is None:
        cost_weights = {'center': 5.0, 'size': 5.0, 'confidence': 2.0, 'giou': 2.0}

    B = gt_bboxes.shape[0]
    indices = []

    pred_centers = pred_bboxes['pred_centers']  # [B, Q, 3]
    pred_sizes = pred_bboxes['pred_sizes']  # [B, Q, 3]
    pred_confs = pred_bboxes['pred_confidences']  # [B, Q, 1]

    for b in range(B):
        n_gt = gt_mask[b].sum().item()
        if n_gt == 0:
            indices.append((torch.tensor([], dtype=torch.long),
                           torch.tensor([], dtype=torch.long)))
            continue

        gt_b = gt_bboxes[b, :n_gt]  # [n_gt, 7]
        gt_centers = gt_b[:, :3]
        gt_sizes = gt_b[:, 3:6]

        # Cost components
        # L1 cost for center
        cost_center = torch.cdist(pred_centers[b], gt_centers, p=1)  # [Q, n_gt]

        # L1 cost for size
        cost_size = torch.cdist(pred_sizes[b], gt_sizes, p=1)  # [Q, n_gt]

        # Confidence cost: -log(confidence) for matched predictions
        cost_conf = -torch.log(pred_confs[b].squeeze(-1) + 1e-8).unsqueeze(1)  # [Q, 1]
        cost_conf = cost_conf.expand(-1, n_gt)  # [Q, n_gt]

        # GIoU cost (axis-aligned approximation, ignoring rotation)
        pred_boxes6 = torch.cat([pred_centers[b], pred_sizes[b]], dim=-1)  # [Q, 6]
        gt_boxes6 = torch.cat([gt_centers, gt_sizes], dim=-1)  # [n_gt, 6]
        cost_giou = -bbox3d_giou(pred_boxes6, gt_boxes6)  # [Q, n_gt], negate for cost

        # Total cost
        cost = (cost_weights['center'] * cost_center +
                cost_weights['size'] * cost_size +
                cost_weights['confidence'] * cost_conf +
                cost_weights['giou'] * cost_giou)

        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
        indices.append((torch.tensor(row_ind, dtype=torch.long),
                       torch.tensor(col_ind, dtype=torch.long)))

    return indices


class BBoxCriterion(nn.Module):
    """
    Loss computation for 3D bounding box estimation with Hungarian matching.

    Supports:
    - GIoU or DIoU loss (use_diou flag)
    - Auxiliary decoder losses (aux_loss_weight > 0)
    """
    def __init__(
        self,
        lambda_center=5.0,
        lambda_size=5.0,
        lambda_rotation=2.0,
        lambda_confidence=1.0,
        lambda_giou=2.0,
        cost_center=5.0,
        cost_size=5.0,
        cost_confidence=2.0,
        cost_giou=2.0,
        use_diou=False,
        aux_loss_weight=0.0,
    ):
        super().__init__()
        self.lambda_center = lambda_center
        self.lambda_size = lambda_size
        self.lambda_rotation = lambda_rotation
        self.lambda_confidence = lambda_confidence
        self.lambda_giou = lambda_giou
        self.use_diou = use_diou
        self.aux_loss_weight = aux_loss_weight

        self.cost_weights = {
            'center': cost_center,
            'size': cost_size,
            'confidence': cost_confidence,
            'giou': cost_giou,
        }

    def _compute_loss_for_layer(self, predictions, gt_bboxes, gt_mask, indices):
        """
        Compute regression + confidence losses for one decoder layer using pre-computed indices.

        Args:
            predictions: dict with pred_centers, pred_sizes, pred_rotations, pred_confidences
            gt_bboxes: [B, M, 7]
            gt_mask: [B, M]
            indices: list of (pred_idx, gt_idx) from Hungarian matching

        Returns:
            loss_dict with individual loss components and total 'loss'
        """
        B = gt_bboxes.shape[0]
        Q = predictions['pred_centers'].shape[1]
        device = predictions['pred_centers'].device

        iou_fn = bbox3d_diou if self.use_diou else bbox3d_giou

        loss_center = torch.tensor(0.0, device=device)
        loss_size = torch.tensor(0.0, device=device)
        loss_rotation = torch.tensor(0.0, device=device)
        loss_iou = torch.tensor(0.0, device=device)

        num_matched = 0
        for b in range(B):
            pred_idx, gt_idx = indices[b]
            if len(pred_idx) == 0:
                continue

            pred_idx = pred_idx.to(device)
            gt_idx = gt_idx.to(device)
            n = len(pred_idx)
            num_matched += n

            gt_b = gt_bboxes[b]

            pred_c = predictions['pred_centers'][b][pred_idx]
            gt_c = gt_b[gt_idx, :3]
            loss_center = loss_center + F.l1_loss(pred_c, gt_c, reduction='sum')

            pred_s = predictions['pred_sizes'][b][pred_idx]
            gt_s = gt_b[gt_idx, 3:6]
            loss_size = loss_size + F.l1_loss(pred_s, gt_s, reduction='sum')

            pred_r = predictions['pred_rotations'][b][pred_idx, 0]
            gt_r = gt_b[gt_idx, 6]
            loss_rotation = loss_rotation + (1 - torch.cos(pred_r - gt_r)).sum()

            pred_box6 = torch.cat([pred_c, pred_s], dim=-1)
            gt_box6 = torch.cat([gt_c, gt_s], dim=-1)
            iou_matrix = iou_fn(pred_box6, gt_box6)
            iou_diag = torch.diag(iou_matrix)
            loss_iou = loss_iou + (1 - iou_diag).sum()

        num_matched = max(num_matched, 1)
        loss_center = loss_center / num_matched
        loss_size = loss_size / num_matched
        loss_rotation = loss_rotation / num_matched
        loss_iou = loss_iou / num_matched

        # Confidence focal loss
        conf_targets = torch.zeros(B, Q, device=device)
        for b in range(B):
            pred_idx, _ = indices[b]
            if len(pred_idx) > 0:
                conf_targets[b, pred_idx.to(device)] = 1.0

        conf_logits = predictions['pred_confidences'].squeeze(-1)
        conf_logits_raw = torch.log(conf_logits / (1 - conf_logits + 1e-8) + 1e-8)
        loss_confidence = focal_loss(conf_logits_raw.reshape(-1),
                                     conf_targets.reshape(-1))

        total_loss = (
            self.lambda_center * loss_center +
            self.lambda_size * loss_size +
            self.lambda_rotation * loss_rotation +
            self.lambda_confidence * loss_confidence +
            self.lambda_giou * loss_iou
        )

        return {
            'loss': total_loss,
            'loss_center': loss_center,
            'loss_size': loss_size,
            'loss_rotation': loss_rotation,
            'loss_confidence': loss_confidence,
            'loss_iou': loss_iou,
            'num_matched': torch.tensor(num_matched, dtype=torch.float32, device=device),
        }

    def forward(self, predictions, gt_bboxes, gt_mask):
        """
        Compute all losses including auxiliary decoder losses.

        Args:
            predictions: dict from BBoxEstimator.forward()
                Must have top-level keys (final layer predictions).
                May have 'aux_outputs': list of dicts (intermediate layer predictions).
            gt_bboxes: [B, M, 7] ground truth boxes
            gt_mask: [B, M] boolean mask

        Returns:
            loss_dict: dict with 'loss' (total) and individual loss components
        """
        # Hungarian matching on final layer predictions only
        indices = hungarian_match(predictions, gt_bboxes, gt_mask, self.cost_weights)

        # Final layer loss
        loss_dict = self._compute_loss_for_layer(predictions, gt_bboxes, gt_mask, indices)

        # Auxiliary decoder losses (reuse matching indices from final layer)
        aux_outputs = predictions.get('aux_outputs', [])
        if aux_outputs and self.aux_loss_weight > 0:
            aux_loss_sum = torch.tensor(0.0, device=loss_dict['loss'].device)
            for aux_pred in aux_outputs:
                aux_loss = self._compute_loss_for_layer(aux_pred, gt_bboxes, gt_mask, indices)
                aux_loss_sum = aux_loss_sum + aux_loss['loss']
            aux_loss_avg = aux_loss_sum / len(aux_outputs)
            loss_dict['loss'] = loss_dict['loss'] + self.aux_loss_weight * aux_loss_avg
            loss_dict['loss_aux'] = aux_loss_avg

        return loss_dict
