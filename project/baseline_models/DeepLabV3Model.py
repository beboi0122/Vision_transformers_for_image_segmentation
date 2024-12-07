import torch
import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet50

class DeepLabV3Model(torch.nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.model = deeplabv3_resnet50(pretrained=pretrained)

        self.model.classifier[4] = nn.Conv2d(
            in_channels=self.model.classifier[4].in_channels,
            out_channels=1,
            kernel_size=self.model.classifier[4].kernel_size,
            stride=self.model.classifier[4].stride,
            padding=self.model.classifier[4].padding
        )
        if pretrained:
            for param in self.model.parameters():
                param.requires_grad = False

            # Csak az utolsó réteg paramétereinek engedélyezése a tanításra
            for param in self.model.classifier[4].parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)["out"]