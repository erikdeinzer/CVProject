# License Plate Detector - a CV Project



## Basic setup

This Project tries to allow straightforward, registry-based CV model development. The idea of this design comes from the `mmdetection3d` framework and was implemented in a more lightweight manner.

### Model development
In the model design process, basic `torch.nn.Modules` build the core building blocks. To allow access to those developed blocks, a function decorator is applied. 

```python
import torch.nn as nn

@MODELS.register_module
class FooModel(nn.Module):
    def __init__(self, *kwargs):
        ...
    def forward(self, *kwargs):
        ...
```

## Registries
There are four categories of registries

|Registry|Usecase|
|---|---|
|`MODELS` | The model to be trained|
|`DATASETS` | The Dataset class - as this project is only designed for CCPV, only this Dataset will be implemented |
|`TRANSFORMS`| The transforms applied by the Dataloader - most of them will be the standard transforms |
|`EVALUATIONS`| The Evaluation procedures to assess model performance |


## Model building
To use a specific, registered module, it has to be defined in the config file

```python
model_cfg = dict(type='YOLOv5', in_features=128,)
```

The implemented builders are capable of either `dict` inputs (for only one module to be loaded) or `list[dict]` inputs for sequential model development.

## Builders

The builders build the configurations defined in the config file used. Note that they will be build sequentially, thus modules outputs will be passed as input to their subsequent module.

```python
transforms = [
        dict(type='ToTensor'),
        dict(type='Augment1', param1=0.5, param2=0.5),
        dict(type='Augment2', paramA=0.5, paramB=0.5),
        dict(type='PackInputs')
    ]
```

This will execute
ToTensor $\rightarrow$ Augment1 $\rightarrow$ Augment2 $\rightarrow$ PackInputs


## Configuration files

The configuration files are demanded to have the variables:

|Variable|Description|
|---|---|
|`model_cfg` | Description and configuration for the models to be loaded | 
|`dataset_cfg` | Description and configuration of the datasets to be loaded |
|`train_dataloader` | Configuration of the data loader for the training phase |
|`eval_dataloader` | Configuration of the data loader for the evaluation phase |





