# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from monai.data import CacheDataset, DataLoader, DistributedSampler, SmartCacheDataset, load_decathlon_datalist
from torch.utils.data import Dataset as _TorchDataset
from typing import Callable, Optional
import random
import torch
import numpy as np
import torch.nn.functional as F
from monai.transforms import (
    apply_transform,
    Compose,
    LoadImaged,
    ToTensord,
)

class UniViT_Dataset(_TorchDataset):
    """
    A generic dataset with a length property and an optional callable data transform
    when fetching a data sample.
    For example, typical input data can be a list of dictionaries::

        [{                            {                            {
             'img': 'image1.nii.gz',      'img': 'image2.nii.gz',      'img': 'image3.nii.gz',
             'seg': 'label1.nii.gz',      'seg': 'label2.nii.gz',      'seg': 'label3.nii.gz',
             'extra': 123                 'extra': 456                 'extra': 789
         },                           },                           }]
    """

    def __init__(self, data, config):
        """
        Args:
            data (Iterable): input data to load and transform to generate dataset for model.
            transform: a callable data transform on input data.
        """
        self.data = data
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        image_path, _, _, _, _ = random.choice(self.data[index])
        img = torch.tensor(np.load(image_path), dtype=torch.float)
        if len(img.shape) == 4:
            img = img[:, :, :, 0]
        img = img.permute(2, 0, 1).unsqueeze(0)
        img = F.interpolate(
            img.unsqueeze(0),
            size=(self.config.max_depth, self.config.max_height, self.config.max_width),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0).numpy()
        data = {"image": img}
        return data

def get_loader(args, datalist, config):
    num_workers = config.num_workers
    # def addChannel(image):
    #     image['image'] = image['image'].reshape(1, *image['image'].shape)
    #     print('addChannel', image['image'].shape)
    #     return image
    
    # def reorient(image):
    #     image['image'] = image['image'].permute_dims((0, 3, 1, 2))
    #     print('reorient', image['image'].shape)
    #     return image

    # def resize(image):
    #     image['image'] = F.interpolate(
    #         torch.from_numpy(image['image']).unsqueeze(0),
    #         size=(config.max_depth, config.max_height, config.max_width),
    #         mode="trilinear",
    #         align_corners=False,
    #     ).squeeze(0).numpy()
    #     print('resize', image['image'].shape)
    #     return image

    # train_transforms = Compose(
    #     [
    #         LoadImaged(keys=["image"]),
    #         addChannel,
    #         reorient,
    #         resize,
    #         # RandSpatialCropSamplesd(
    #         #     keys=["image"],
    #         #     roi_size=(10,224,224),#(config.max_depth, config.max_height, config.max_width),
    #         #     num_samples=2,#args.sw_batch_size,
    #         #     random_center=True,
    #         #     random_size=False,
    #         # ),
    #         ToTensord(keys=["image"]),
    #     ]
    # )
    train_ds = UniViT_Dataset(datalist, config) #, transform=train_transforms)

    if args.distributed:
        train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True)
    else:
        train_sampler = None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=num_workers, sampler=train_sampler, drop_last=True
    )

    return train_loader
