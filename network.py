import torch
import torch.nn as nn

from globals import MODELS, DATASETS, EVALUATIONS

@MODELS.register
class YOLOv5(nn.Module):
    def __init__(self, **kwargs):
        super(YOLOv5, self).__init__(**kwargs)
        raise NotImplementedError('Must be implemented')
    def forward(self, inputs, **kwargs):
        raise NotImplementedError('Must be implemented')


@MODELS.register
class PDLPR(nn.Module):
    def __init__(self, **kwargs):
        super(PDLPR, self).__init__(**kwargs)
        raise NotImplementedError('Must be implemented')
    def forward(self, inputs, **kwargs):
        raise NotImplementedError('Must be implemented')



