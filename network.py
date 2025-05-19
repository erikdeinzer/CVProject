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
        self.yolo.to(device)

    def forward(self, x, targets=None):
        # If targets is not None, we are in training mode
        if self.training and targets is not None:
            loss = self.yolo(x, mode='train', batch=targets) # Compute loss
            # Switch to no_grad for prediction output
            self.eval() # Switch to eval mode
            with torch.no_grad():
                preds = self.yolo(x)
            self.train() # Switch back to train mode
            return {'loss': loss, 'preds': preds}
        else:
            return {'preds': self.yolo(x)}

    def loss(self, x, targets):
        self.train()
        out = self.forward(x, targets)

        return out

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



