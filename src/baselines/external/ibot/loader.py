# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import pickle
import math
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder

class PseudoImageFolder(ImageFolder):
    def __init__(self, data_list, transform=None):
        """
        Initialize with a data list, where each item is a list of tuples: 
        (image_path, dimensions, _, label, _)
        and an optional transform function.
        """
        self.data_list = pickle.load(open(data_list, "rb"))
        self.transform = transform
        self.label_to_index = self.create_label_mapping()

    def __len__(self):
        # Return the total number of patients (each patient has multiple images)
        return len(self.data_list)

    def __getitem__(self, index):
        # Randomly pick an image tuple from the data list of the selected patient
        patient_data = self.data_list[index]
        image_tuple = random.choice(patient_data)  # Randomly pick an image tuple
        image_path, _, _, label, _ = image_tuple
        label_index = self.label_to_index[label]
        
        # Load the image (assumes image_path is a valid file path to a 3D or 2D image)
        image = self.load_image(image_path)
        
        # Apply transformations (if any)
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label_index
    
    def create_label_mapping(self):
        """
        Creates a mapping from string labels to integer indices.
        """
        labels = set([label for patient in self.data_list for (_, _, _, label, _) in patient])
        return {label: idx for idx, label in enumerate(sorted(labels))}
    
    def load_image(self, image_path):
        if image_path.endswith(".npy"):
            img = np.load(image_path)
            if len(img.shape) == 4:
                img = img[:, :, :, 0]
            if True: # For now take middle slice
                img = img[:, :, img.shape[2] // 2]
            else: # Possibly take random slice
                img = img[:, :, random.randint(0, img.shape[2] - 1)]
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img).convert("RGB")
        elif (
            image_path.endswith(".jpg")
            or image_path.endswith(".png")
            or image_path.endswith(".tif")
        ):
            img = Image.open(image_path).convert("RGB")
        else:
            raise ValueError("Invalid image format")
        return img
    

class ImageFolderInstance(PseudoImageFolder):
    def __getitem__(self, index):
        img, target = super(ImageFolderInstance, self).__getitem__(index)
        return img, target, index

class ImageFolderMask(PseudoImageFolder):
    def __init__(self, *args, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio, 
                 pred_shape='block', pred_start_epoch=0, **kwargs):
        super(ImageFolderMask, self).__init__(*args, **kwargs)
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio
        
        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        output = super(ImageFolderMask, self).__getitem__(index)
                
        masks = []
        for img in output[0]:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except:
                # skip non-image
                continue
            
            high = self.get_pred_ratio() * H * W
            
            if self.pred_shape == 'block':
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2 
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta
            
            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)

            else:
                # no implementation
                assert False

            masks.append(mask)

        return output + (masks,)