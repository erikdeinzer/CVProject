import torch
import torch.nn as nn

from globals import DATASETS, MODELS, EVALUATIONS
from globals import Builder

from network import *
from globals import *
from utils import *
from data import *

dataset_builder = Builder(DATASETS)
model_builder = Builder(MODELS)
eval_builder = Builder(EVALUATIONS)
transforms_builder = Builder(TRANSFORMS)


def main(dataset_cfg: dict, 
         model_cfg: dict | list[dict],
         eval_cfg: dict,
         train_cfg,
         optim_cfg,):
    
    dataset = dataset_builder.build_module(**dataset_cfg)

    if isinstance(model_cfg, dict):
        model = model_builder.build_module(**model_cfg)
    elif isinstance(model_cfg, list):
        model = nn.Sequential([
            model_builder.build_module(**cfg)
            for cfg in model_cfg
        ])
    else:
        raise TypeError('Model config must either be a dict or a list of dicts')
    
    








