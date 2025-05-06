import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os

from PIL import Image

from globals import Builder

import numpy as np

from globals import DATASETS, TRANSFORMS, Builder


TRANSFORMS.register(T.ToTensor, type='ToTensor')
TRANSFORMS.register(T.Normalize, type='Normalize')
TRANSFORMS.register(T.Resize, type='Resize')
TRANSFORMS.register(T.ColorJitter, type='ColorJitter')
TRANSFORMS.register(T.RandomHorizontalFlip, type='RandomHorizontalFlip')
TRANSFORMS.register(T.RandomRotation, type='RandomRotation')
TRANSFORMS.register(T.RandomAffine, type='RandomAffine')
TRANSFORMS.register(T.RandomCrop, type='RandomCrop')
TRANSFORMS.register(T.RandomErasing, type='RandomErasing')
TRANSFORMS.register(T.RandomPerspective, type='RandomPerspective')
TRANSFORMS.register(T.RandomResizedCrop, type='RandomResizedCrop')
TRANSFORMS.register(T.RandomVerticalFlip, type='RandomVerticalFlip')
TRANSFORMS.register(T.RandomAdjustSharpness, type='RandomAdjustSharpness')
TRANSFORMS.register(T.RandomAutocontrast, type='RandomAutocontrast')
TRANSFORMS.register(T.RandomEqualize, type='RandomEqualize')
TRANSFORMS.register(T.RandomInvert, type='RandomInvert')
TRANSFORMS.register(T.RandomSolarize, type='RandomSolarize')
TRANSFORMS.register(T.AugMix, type='AugMix')
TRANSFORMS.register(T.AutoAugment, type='AutoAugment')
TRANSFORMS.register(T.RandAugment, type='RandAugment')


@DATASETS.register(type='CCPD')
class CCPD(Dataset):
    def __init__(self, data_root, split='train', pipeline=None):
        """
        Args:
            data_root (str): Root directory containing CCPD image data.
            split (str): Dataset split - typically 'train', 'val', or 'test'.
            transforms (list[dict]): Transform pipeline defined in config.
        """
        self.data_root = os.path.join(data_root, split)
        self.image_files = [f for f in os.listdir(self.data_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.transforms = self._build_transforms(pipeline)

    def _build_transforms(self, transform_cfgs):
        transforms_builder = Builder(TRANSFORMS)
        if transform_cfgs is None:
            return None
        transform_list = []
        for cfg in transform_cfgs:
            transform = transforms_builder.build_module(**cfg)
            transform_list.append(transform)
        return T.Compose(transform_list)

    def __len__(self):
        return len(self.image_files)

    def _parse_label_from_filename(self, filename):
        # Example filename: 080811_136_275&404_249&434-94&164_263&164_263&191_94&191-0_0_7_5_2_3_4-28-26.jpg
        # Last part before extension is plate characters
        try:
            label_str = filename.split('-')[-3]
            label = [int(char) for char in label_str.split('_')]  # Convert to int if using digits as class IDs
            return label
        except Exception as e:
            raise ValueError(f"Could not parse label from filename: {filename}") from e

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.data_root, filename)
        image = Image.open(img_path).convert('RGB')
        label = self._parse_label_from_filename(filename)

        if self.transforms:
            image = self.transforms(image)

        return {'img': image, 'label': torch.tensor(label, dtype=torch.long)}
    

@DATASETS.register(type='RepeatDataset')
class RepeatDataset(Dataset):
    def __init__(self, dataset_cfg, times=1):
        """
        Args:
            dataset_cfg (dict): Config dict for the base dataset.
            times (int): Number of repetitions.
        """
        dataset_builder = Builder(DATASETS)
        self.dataset = dataset_builder.build_module(**dataset_cfg)
        self.times = times

    def __len__(self):
        return len(self.dataset) * self.times

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]
    

    

def build_dataloader_from_cfg(cfg):
    """
    Builds a PyTorch DataLoader from a config dictionary.

    Args:
        cfg (dict): Config with `dataset` and DataLoader options.

    Returns:
        DataLoader
    """
    cfg = cfg.copy()
    dataset_cfg = cfg['dataset']
    dataset_builder = Builder(DATASETS)
    cfg['dataset'] = dataset_builder.build_module(**dataset_cfg)

    dataloader = DataLoader(
        **cfg
    )
    return dataloader