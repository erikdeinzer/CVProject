import torch
import torch.nn as nn

import cv2
import numpy as np

from globals import DATASETS


@DATASETS.register
class CCPD():
    def __init__(self, **kwargs):
        raise NotImplementedError('Must be implemented')
    def transform(self, **kwargs):
        raise NotImplementedError('Must be implemented')
    def get_sample(self, **kwargs):
        raise NotImplementedError('Must be implemented')
    

