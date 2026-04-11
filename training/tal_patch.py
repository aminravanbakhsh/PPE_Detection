"""Patch Ultralytics ``TaskAlignedAssigner`` before training (Apple / PyTorch boolean-index bugs).

``get_box_metrics`` uses 4D boolean indexing that can disagree with ``overlaps[mask]`` assignment
length on macOS (see https://github.com/ultralytics/ultralytics/issues/22971).  We gather/scatter
via a flattened mask.  ``forward`` runs on CPU when inputs are on MPS to avoid bad advanced indexing.

Only the **class** on ``ultralytics.utils.tal`` is patched.  Instance-level binding via
``types.MethodType`` is intentionally avoided because it pollutes the instance ``__dict__``,
which gets pickled into checkpoints and then fails to unpickle (``nn.Module.__getattr__``
cannot resolve the bound-method reference).
"""

from __future__ import annotations

import torch
from ultralytics.utils.tal import TaskAlignedAssigner

_original_tal_forward = TaskAlignedAssigner.forward


def _fixed_get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
    na = pd_bboxes.shape[-2]
    mask_gt = mask_gt.bool()
    overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
    bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

    dev = pd_scores.device
    ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long, device=dev)
    ind[0] = torch.arange(end=self.bs, device=dev).view(-1, 1).expand(-1, self.n_max_boxes)
    ind[1] = gt_labels.squeeze(-1).long()
    flat_m = mask_gt.reshape(-1)
    gathered = pd_scores[ind[0], :, ind[1]]
    bbox_scores.reshape(-1)[flat_m] = gathered.reshape(-1)[flat_m]

    exp_pd = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1).reshape(-1, 4)
    exp_gt = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1).reshape(-1, 4)
    pd_boxes = exp_pd[flat_m]
    gt_boxes = exp_gt[flat_m]
    iou = self.iou_calculation(gt_boxes, pd_boxes).reshape(-1)
    overlaps_flat = overlaps.reshape(-1)
    overlaps_flat[flat_m] = iou.to(overlaps_flat.dtype)

    align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
    return align_metric, overlaps


def _mps_safe_tal_forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
    if gt_bboxes.device.type != "mps":
        return _original_tal_forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
    device = gt_bboxes.device
    cpu = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]
    result = _original_tal_forward(self, *cpu)
    return tuple(t.to(device) for t in result)


def apply_tal_patches() -> None:
    """Re-bind patches on ``ultralytics.utils.tal`` (idempotent)."""
    import ultralytics.utils.tal as ut

    ut.TaskAlignedAssigner.get_box_metrics = _fixed_get_box_metrics
    ut.TaskAlignedAssigner.forward = _mps_safe_tal_forward


apply_tal_patches()
