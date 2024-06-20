import torch
import random
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

MIN_CROP = 0.8
MIN_RESOLUTION = 32
SLICE_AUGMENTATION_PROB = 1 / 2
IMAGE_AUGMENTATION_PROB = 3 / 4

class ImageDataset(Dataset):
    def __init__(self, dataset, config, device, augment=False, downstream=False, multiclass=False):
        self.dataset = dataset
        self.config = config
        self.device = device
        self.init_augment = augment
        self.downstream = downstream
        self.multiclass = multiclass
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)
      
    def load_image(self, image_path, chosenDim):
        if image_path.endswith('.npy'):
            img = torch.tensor(np.load(image_path), dtype=torch.float)
            if len(img.shape) == 4:
                img = img[:,:,:,0]
            img = img.permute(2, 0, 1).unsqueeze(1).repeat(1, 3, 1, 1)
        elif image_path.endswith('.jpg') or image_path.endswith('.png') or image_path.endswith('.tif'):
            img = Image.open(image_path).convert('RGB')
            img = self.transform(img).unsqueeze(0)
        else:
            raise ValueError('Invalid image format')
        
        currDim = tuple(img[:, 0, :, :].shape)
        if chosenDim is not None and chosenDim != currDim:
            if currDim[1] == chosenDim[2] and currDim[2] == chosenDim[1]:
                img = img.permute(0, 1, 3, 2)
            elif (currDim[1] > chosenDim[1]) == (chosenDim[2] > currDim[2]) and (currDim[1] != chosenDim[1] and currDim[2] != chosenDim[2]):
                img = img.permute(0, 1, 3, 2)
                img = img.permute(1, 0, 2, 3).unsqueeze(0)
                img = F.interpolate(img, size=(chosenDim[0], chosenDim[1], chosenDim[2]), mode='trilinear', align_corners=True)
                img = img.squeeze(0).permute(1, 0, 2, 3)
            else:
                img = img.permute(1, 0, 2, 3).unsqueeze(0)
                img = F.interpolate(img, size=(chosenDim[0], chosenDim[1], chosenDim[2]), mode='trilinear', align_corners=True)
                img = img.squeeze(0).permute(1, 0, 2, 3)
        
        if img.shape[0] > self.config.max_slice:
            scale_factor_slices = self.config.max_slice / img.shape[0]
        else:
            scale_factor_slices = 1
        if img.shape[2] > self.config.max_height:
            scale_factor_height = self.config.max_height / img.shape[2]
        else:
            scale_factor_height = 1
        if img.shape[3] > self.config.max_width:
            scale_factor_width = self.config.max_width / img.shape[3]
        else:
            scale_factor_width = 1
        
        scale_factor_image = min(scale_factor_height, scale_factor_width)
        if min(scale_factor_image, scale_factor_slices) < 1:
            img = img.permute(1, 0, 2, 3).unsqueeze(0)
            img = F.interpolate(img, scale_factor=(scale_factor_slices, scale_factor_image, scale_factor_image), mode='trilinear', align_corners=True)
            img = img.squeeze(0).permute(1, 0, 2, 3)

        return img
    
    # dim is (slices, height, width)
    
    def random_crop(self, img, dim):
        crop_height = random.randint(int(dim[2] * MIN_CROP), img.shape[2])  # Adjust range as needed
        crop_width = random.randint(int(dim[3] * MIN_CROP), img.shape[3])  # Adjust range as needed
        i, j, h, w = transforms.RandomCrop.get_params(img[0, 0, :, :], output_size=(crop_height, crop_width))
        img[:, :, :crop_height, :crop_width] = img[:, :, i:i+h, j:j+w]
        img[:, :, crop_height:, :] = 0
        img[:, :, :, crop_width:] = 0
        dim[1] = crop_height
        dim[2] = crop_width
        return img, dim

    def random_resize(self, img, dim):
        min_scale = max(MIN_RESOLUTION / dim[2], MIN_RESOLUTION / dim[3])
        max_scale = min(self.config.max_height / dim[2], self.config.max_width / dim[3])
        scale = random.uniform(min_scale, max_scale)
        height = min(max(int(dim[2] * scale), MIN_RESOLUTION), self.config.max_height)
        width = min(max(int(dim[3] * scale), MIN_RESOLUTION), self.config.max_width)
        img[:, :, :height, :width] = F.interpolate(img[:, :, :dim[2], :dim[3]], size=(height, width), mode='bilinear', align_corners=False)
        img[:, :, height:, :] = 0
        img[:, :, :, width:] = 0
        dim[1] = height
        dim[2] = width
        return img, dim

    def slice_interpolation(self, img, dim):
        # Interpolating slices if there are multiple, adjusting to a specific number of slices
        if dim[0] > 2:
            slices = random.randint(2, dim[0])  # Target number of slices, adjust as needed
            img = img.permute(1, 0, 2, 3)
            img[:, :slices, :dim[1], :dim[2]] = F.interpolate(img[:, :dim[0], :dim[1], :dim[2]].unsqueeze(0), size=(slices, dim[1], dim[2]), mode='trilinear', align_corners=False).squeeze(0)
            img = img.permute(1, 0, 2, 3)
            img[slices:, :, :, :] = 0
            dim[0] = slices
        return img, dim

    def select_slice(self, img, dim):
        if dim[1] > 1:
            idx = random.randint(0, dim[0] - 1)
            img[0, :, :, :] = img[idx, :, :, :]
            img[1:, :, :, :] = 0
            dim[0] = 1
        return img, dim
    
    def augment(self, img, dim):
        hasAugmented = False
        if dim[0] > 1 and random.random() < SLICE_AUGMENTATION_PROB:    
            hasAugmented = True
            if dim[0] > 2 and random.random() < 0.5:
                img, dim = self.slice_interpolation(img, dim)
            else:
                img, dim = self.select_slice(img, dim)
            
        if not hasAugmented or random.random() < IMAGE_AUGMENTATION_PROB:
            if random.random() < 0.5:
                img, dim = self.random_resize(img, dim)
            else:
                img, dim = self.random_crop(img, dim)
                
        return img, dim
        
    def augment_batch(self, img, dim):
        for i in range(img.shape[0]):
            img[i], dim[i] = self.augment(img[i], dim[i])
            
        return img, dim

    def __getitem__(self, idx):
        path, dim, _, _, labels = self.dataset[idx]
        image_tensor = torch.zeros(self.config.max_slice, self.config.num_channels, self.config.max_height, self.config.max_width, dtype=torch.float, device=self.device)
        dimension_tensor = torch.ones(3, dtype=torch.long)
        img = self.load_image(path, dim)
        image_tensor[:img.shape[0], :, :img.shape[2], :img.shape[3]] = img
        dimension_tensor[0] = img.shape[0]
        dimension_tensor[1] = img.shape[2]
        dimension_tensor[2] = img.shape[3]
            
        if self.init_augment:
            image_tensor, dimension_tensor = self.augment(image_tensor, dimension_tensor)
                    
        if self.downstream:
            label_tensor = torch.tensor(labels, dtype=torch.long if self.multiclass else torch.float, device=self.device)
            return image_tensor, dimension_tensor, label_tensor
        
        return image_tensor, dimension_tensor