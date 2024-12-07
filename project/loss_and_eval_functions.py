import torch
import torch.nn as nn

def dice_score(pred, target, smooth=1e-4):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice

def iou_score(pred, target, smooth=1e-4):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def dice_loss(pred, target, smooth=1e-4):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = 1 - (2 * intersection + smooth) / (union + smooth)
    return dice

def combined_loss(pred, target, alpha=0.5):
    bce = nn.BCEWithLogitsLoss()(pred, target)
    dice = dice_loss(pred, target)
    return alpha * bce + (1 - alpha) * dice