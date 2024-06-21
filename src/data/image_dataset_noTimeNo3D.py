import torch
import random
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

MIN_CROP = 0.8
MIN_RESOLUTION = 32

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
            img = img[:, :, img.shape[2] // 2] # Take middle slice for 3D images
            img = img.unsqueeze(0).repeat(3, 1, 1)
        elif image_path.endswith('.jpg') or image_path.endswith('.png') or image_path.endswith('.tif'):
            img = Image.open(image_path).convert('RGB')
            img = self.transform(img)
        else:
            raise ValueError('Invalid image format')
        
        if img.shape[1] > self.config.max_height:
            scale_factor_height = self.config.max_height / img.shape[1]
        else:
            scale_factor_height = 1
        if img.shape[2] > self.config.max_width:
            scale_factor_width = self.config.max_width / img.shape[2]
        else:
            scale_factor_width = 1
        
        scale_factor_image = min(scale_factor_height, scale_factor_width)
        if scale_factor_image < 1:
            img = img.unsqueeze(0)
            img = F.interpolate(img, scale_factor=(scale_factor_image, scale_factor_image), mode='bilinear', align_corners=True)
            img = img.squeeze(0)

        return img
    
    def random_crop(self, img, dim):
        crop_height = random.randint(int(dim[0] * MIN_CROP), img.shape[1])  # Adjust range as needed
        crop_width = random.randint(int(dim[1] * MIN_CROP), img.shape[2])  # Adjust range as needed
        i, j, h, w = transforms.RandomCrop.get_params(img[0, :, :], output_size=(crop_height, crop_width))
        img[:, :crop_height, :crop_width] = img[:, i:i+h, j:j+w]
        img[:, crop_height:, :] = 0
        img[:, :, crop_width:] = 0
        dim[0] = crop_height
        dim[1] = crop_width
        return img, dim

    def random_resize(self, img, dim):
        min_scale = max(MIN_RESOLUTION / dim[0], MIN_RESOLUTION / dim[1])
        max_scale = min(self.config.max_height / dim[0], self.config.max_width / dim[1])
        scale = random.uniform(min_scale, max_scale)
        height = min(max(int(dim[0] * scale), MIN_RESOLUTION), self.config.max_height)
        width = min(max(int(dim[1] * scale), MIN_RESOLUTION), self.config.max_width)
        img[:, :height, :width] = F.interpolate(img[:, :dim[0], :dim[1]].unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False).squeeze(0)
        img[:, height:, :] = 0
        img[:, :, width:] = 0
        dim[0] = height
        dim[1] = width
        return img, dim
    
    def augment(self, img, dim):
        if random.random() < 0.5:
            img, dim = self.random_resize(img, dim)
        else:
            img, dim = self.random_crop(img, dim)
                
        return img, dim
        
    def augment_batch(self, img, dim):
        img = img.clone()
        dim = dim.clone()
        for i in range(img.shape[0]):
            img[i], dim[i] = self.augment(img[i], dim[i])
            
        return img, dim

    def __getitem__(self, idx):
        path, dim, _, _, labels = self.dataset[idx]
        image_tensor = torch.zeros(self.config.num_channels, self.config.max_height, self.config.max_width, dtype=torch.float, device=self.device)
        dimension_tensor = torch.ones(2, dtype=torch.long)
        img = self.load_image(path, dim)
        image_tensor[:, :img.shape[1], :img.shape[2]] = img
        dimension_tensor[0] = img.shape[1]
        dimension_tensor[1] = img.shape[2]
            
        if self.init_augment:
            image_tensor, dimension_tensor = self.augment(image_tensor, dimension_tensor)
                    
        if self.downstream:
            label_tensor = torch.tensor(labels, dtype=torch.long if self.multiclass else torch.float, device=self.device)
            return image_tensor, dimension_tensor, label_tensor
        
        return image_tensor, dimension_tensor