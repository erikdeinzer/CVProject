import torch
import torch.nn as nn

from globals import MODELS, DATASETS, EVALUATIONS


@MODELS.register
class YOLOv5(nn.Module):
    def __init__(self, model_path='yolov5s.pt', device='cuda'):
        super().__init__()
        from ultralytics import YOLO
        self.yolo = YOLO(model_path)
        self.device = device
        self.yolo.model.to(device)

    def forward(self, x, targets=None):
        if self.training and targets is not None:
            return self.yolo.model(x, targets)  # loss
        else:
            return self.yolo(x)  # predictions

    def loss(self, x, targets):
        return self.forward(x, targets)
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)


@MODELS.register
class PDLPR(nn.Module):
    def __init__(self, **kwargs):
        super(PDLPR, self).__init__(**kwargs)
        raise NotImplementedError('Must be implemented')
    def forward(self, x, **kwargs):
        raise NotImplementedError('Must be implemented')



