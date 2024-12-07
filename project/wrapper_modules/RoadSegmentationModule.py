import pytorch_lightning as pl
import torchmetrics
import torch

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

class RoadSegmentationModule(pl.LightningModule):
    def __init__(self, model, loss_fn, optimizer, only_last_layer=False, weight_decay=0.01, lr=1e-3):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.precision_fn = torchmetrics.Precision(task="binary")
        self.optimizer = optimizer
        self.lr = lr
        self.only_last_layer = only_last_layer
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        if y.shape != pred.shape:
            pred = pred.squeeze(1)
        loss = self.loss_fn(pred, y)
        precision = self.precision_fn(pred, y)
        dice = dice_score(pred, y)
        iou = iou_score(pred, y)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_precision", precision, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_dice", dice, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_iou", iou, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        if y.shape != pred.shape:
            pred = pred.squeeze(1)
        loss = self.loss_fn(pred, y)
        precision = self.precision_fn(pred, y)
        dice = dice_score(pred, y)
        iou = iou_score(pred, y)

        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_precision", precision, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_dice", dice, prog_bar=True, on_step=True, on_epoch=True)
        self.log("val_iou", iou, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        if self.optimizer == "Adam":
            if self.only_last_layer:
                opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
            else:
                opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer == "AdamW":
            if self.only_last_layer:
                opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, weight_decay=self.weight_decay)
            else:
                opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError("Optimizer not implemented in RoadSegmentationModule")
        return opt