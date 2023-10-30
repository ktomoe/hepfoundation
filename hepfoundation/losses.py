import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MultiCrossEntropyLoss, self).__init__()

    def forward(self, outputs, targets):

        outputs = outputs.view(-1, outputs.shape[-1])
        targets = targets.view(-1)

        return F.cross_entropy(outputs, targets)
