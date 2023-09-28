import torch
from torch import nn

class FCNHead(nn.Module):
    def __init__(self, dim_in, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(dim_in, num_classes, kernel_size=1, bias=True)
        )

    def forward(self, x):
        return self.classifier(x)