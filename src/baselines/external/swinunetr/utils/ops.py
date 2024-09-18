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

import numpy as np
import torch
from numpy.random import randint


def patch_rand_drop(args, x, x_rep=None, max_drop=0.3, max_block_sz=0.33, tolr=0.1):
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
                (c, rnd_d - rnd_s, rnd_h - rnd_r, rnd_w - rnd_c), dtype=x.dtype, device=args.local_rank
            ).normal_()
            x_uninitialized = (x_uninitialized - torch.min(x_uninitialized)) / (
                torch.max(x_uninitialized) - torch.min(x_uninitialized)
            )
            x[:, rnd_s:rnd_d, rnd_r:rnd_h, rnd_c:rnd_w] = x_uninitialized
        else:
            x[:, rnd_s:rnd_d, rnd_r:rnd_h, rnd_c:rnd_w] = x_rep[:, rnd_s:rnd_d, rnd_r:rnd_h, rnd_c:rnd_w]
        total_pix = total_pix + (rnd_d - rnd_s) * (rnd_h - rnd_r) * (rnd_w - rnd_c)
    return x


def rot_rand(args, x_s):
    img_n = x_s.size()[0]
    x_aug = x_s.detach().clone()
    x_rot = torch.zeros(img_n).long().to(x_s.device)
    for i in range(img_n):
        x = x_s[i]
        orientation = np.random.randint(0, 4)
        if orientation == 0:
            pass
        elif orientation == 1:
            x = x.rot90(1, (2, 3))
        elif orientation == 2:
            x = x.rot90(2, (2, 3))
        elif orientation == 3:
            x = x.rot90(3, (2, 3))
        x_aug[i] = x
        x_rot[i] = orientation
    return x_aug, x_rot


def aug_rand(args, samples):
    img_n = samples.size()[0]
    x_aug = samples.detach().clone()
    for i in range(img_n):
        x_aug[i] = patch_rand_drop(args, x_aug[i])
        idx_rnd = randint(0, img_n)
        if idx_rnd != i:
            x_aug[i] = patch_rand_drop(args, x_aug[i], x_aug[idx_rnd])
    return x_aug
