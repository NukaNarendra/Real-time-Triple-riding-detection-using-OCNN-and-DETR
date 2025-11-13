import torch
import numpy as np

def compute_map(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_threshold=0.5):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0
    ious = box_iou(torch.tensor(pred_boxes), torch.tensor(gt_boxes))
    correct = (ious > iou_threshold).float()
    tp = (correct.sum(1) > 0).float().sum()
    precision = tp / max(len(pred_boxes), 1)
    recall = tp / max(len(gt_boxes), 1)
    map_val = (precision + recall) / 2.0
    return map_val.item()

def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, None] + area2 - inter + 1e-6)
    return iou

def mean_absolute_error(pred_counts, true_counts):
    pred = np.array(pred_counts)
    true = np.array(true_counts)
    return np.mean(np.abs(pred - true))

def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return precision, recall, f1
