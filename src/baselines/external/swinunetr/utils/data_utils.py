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
from numpy.random import randint

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
    
    def rotate(self, x):
        x1 = x.clone()
        x2 = x.clone()
        orientation1 = np.random.randint(0, 4)
        orientation2 = np.random.randint(0, 4)
        
        if orientation1 == 0:
            pass
        elif orientation1 == 1:
            x1 = x1.rot90(1, (2, 3))
        elif orientation1 == 2:
            x1 = x1.rot90(2, (2, 3))
        elif orientation1 == 3:
            x1 = x1.rot90(3, (2, 3))
            
        if orientation2 == 0:
            pass
        elif orientation2 == 1:
            x2 = x2.rot90(1, (2, 3))
        elif orientation2 == 2:
            x2 = x2.rot90(2, (2, 3))
        elif orientation2 == 3:
            x2 = x2.rot90(3, (2, 3))
            
        rot1 = torch.tensor(orientation1, dtype=torch.long)
        rot2 = torch.tensor(orientation2, dtype=torch.long)
        return x1, x2, rot1, rot2

    def patch_rand_drop(self, x, x_rep=None, max_drop=0.3, max_block_sz=0.33, tolr=0.1):
        c, d, h, w = x.size()
        n_drop_pix = np.random.uniform(0, max_drop) * d * h * w
        mx_blk_slices = int(d * max_block_sz)
        mx_blk_height = int(h * max_block_sz)
        mx_blk_width = int(w * max_block_sz)
        tolr = (int(tolr * d), int(tolr * h), int(tolr * w))
        total_pix = 0
        while total_pix < n_drop_pix:
            rnd_s = randint(0, d - tolr[0])
            rnd_r = randint(0, h - tolr[1])
            rnd_c = randint(0, w - tolr[2])
            rnd_d = min(randint(tolr[0], mx_blk_slices) + rnd_s, d)
            rnd_h = min(randint(tolr[1], mx_blk_height) + rnd_r, h)
            rnd_w = min(randint(tolr[2], mx_blk_width) + rnd_c, w)
            if x_rep is None:
                x_uninitialized = torch.empty(
                    (c, rnd_d - rnd_s, rnd_h - rnd_r, rnd_w - rnd_c), dtype=x.dtype, device=x.device
                ).normal_()
                x_uninitialized = (x_uninitialized - torch.min(x_uninitialized)) / (
                    torch.max(x_uninitialized) - torch.min(x_uninitialized)
                )
                x[:, rnd_s:rnd_d, rnd_r:rnd_h, rnd_c:rnd_w] = x_uninitialized
            else:
                x[:, rnd_s:rnd_d, rnd_r:rnd_h, rnd_c:rnd_w] = x_rep[:, rnd_s:rnd_d, rnd_r:rnd_h, rnd_c:rnd_w]
            total_pix = total_pix + (rnd_d - rnd_s) * (rnd_h - rnd_r) * (rnd_w - rnd_c)
        return x

    def augment_pre(self, x_1, x_2):
        x_aug1 = x_1.clone()
        x_aug2 = x_2.clone()
        x_aug1 = self.patch_rand_drop(x_aug1)
        x_aug2 = self.patch_rand_drop(x_aug2)
        return x_aug1, x_aug2

    def augment_post(self, x_1, x_2):
        img_n = x_1.size()[0]
        x_aug1 = x_1.clone()
        x_aug2 = x_2.clone()
        for i in range(img_n):
            x_aug1[i] = self.patch_rand_drop(x_aug1[i])
            x_aug2[i] = self.patch_rand_drop(x_aug2[i])
            idx_rnd1 = randint(0, img_n)
            idx_rnd2 = randint(0, img_n)
            if idx_rnd1 != i:
                x_aug1[i] = self.patch_rand_drop(x_aug1[i], x_aug1[idx_rnd1])
            if idx_rnd2 != i:
                x_aug2[i] = self.patch_rand_drop(x_aug2[i], x_aug2[idx_rnd2])
        return x_aug1, x_aug2

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
        ).squeeze(0)
        
        x1, x2, rot1, rot2 = self.rotate(img)
        x1_augment, x2_augment = self.augment_pre(x1, x2)
        return x1, x2, rot1, rot2, x1_augment, x2_augment

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
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=num_workers, drop_last=True,
    )

    return train_loader
