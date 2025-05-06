import torch
import torch.nn as nn

import time
import os

from globals import DATASETS, MODELS, EVALUATIONS, TRANSFORMS
from globals import Builder

import network
import globals
import utils
import data

dataset_builder = Builder(DATASETS)
model_builder = Builder(MODELS)
eval_builder = Builder(EVALUATIONS)
transforms_builder = Builder(TRANSFORMS)
