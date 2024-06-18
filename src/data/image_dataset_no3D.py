import torch
import random
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

MIN_CROP = 0.8
MIN_RESOLUTION = 32
TIME_AUGMENTATION_PROB = 1 / 2
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
            img = img[:, :, img.shape[2] // 2] # Take middle slice for 3D images
            img = img.unsqueeze(0).repeat(3, 1, 1)
        elif image_path.endswith('.jpg'):
            img = Image.open(image_path).convert('RGB')
            img = self.transform(img)
        else:
            raise ValueError('Invalid image format')
        
        currDim = tuple(img[0, :, :].shape)
        if chosenDim is not None and chosenDim != currDim:
            if currDim[0] == chosenDim[1] and currDim[1] == chosenDim[0]:
                img = img.permute(0, 2, 1)
            elif (currDim[0] > chosenDim[0]) == (chosenDim[1] > currDim[1]) and (currDim[0] != chosenDim[0] and currDim[1] != chosenDim[1]):
                img = img.permute(0, 2, 1).unsqueeze(0)
                img = F.interpolate(img, size=(chosenDim[0], chosenDim[1]), mode='bilinear', align_corners=True)
                img = img.squeeze(0)
            else:
                img = img.unsqueeze(0)
                img = F.interpolate(img, size=(chosenDim[0], chosenDim[1]), mode='bilinear', align_corners=True)
                img = img.squeeze(0)
        
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
    
    # dim is (time_steps, height, width)
    
    def random_crop(self, img, dim):
        crop_height = random.randint(int(dim[1] * MIN_CROP), img.shape[2])  # Adjust range as needed
        crop_width = random.randint(int(dim[2] * MIN_CROP), img.shape[3])  # Adjust range as needed
        i, j, h, w = transforms.RandomCrop.get_params(img[0, 0, :, :], output_size=(crop_height, crop_width))
        img[:, :, :crop_height, :crop_width] = img[:, :, i:i+h, j:j+w]
        img[:, :, crop_height:, :] = 0
        img[:, :, :, crop_width:] = 0
        dim[1] = crop_height
        dim[2] = crop_width
        return img, dim

    def random_resize(self, img, dim):
        min_scale = max(MIN_RESOLUTION / dim[1], MIN_RESOLUTION / dim[2])
        max_scale = min(self.config.max_height / dim[1], self.config.max_width / dim[2])
        scale = random.uniform(min_scale, max_scale)
        height = min(max(int(dim[1] * scale), MIN_RESOLUTION), self.config.max_height)
        width = min(max(int(dim[2] * scale), MIN_RESOLUTION), self.config.max_width)
        img[:, :, :height, :width] = F.interpolate(img[:, :, :dim[1], :dim[2]].view(-1, self.config.num_channels, dim[1], dim[2]), size=(height, width), mode='bilinear', align_corners=False).view(img.shape[0], self.config.num_channels, height, width)
        img[:, :, height:, :] = 0
        img[:, :, :, width:] = 0
        dim[1] = height
        dim[2] = width
        return img, dim

    def remove_timestep(self, img, dim):
        if dim[0] > 2:  # Check if there are multiple time steps
            idx_to_remove = random.randint(0, dim[0] - 1)
            img[idx_to_remove:-1, :, :, :] = img[idx_to_remove+1:, :, :, :].clone()
            img[-1, :, :, :] = 0
            dim[0] = dim[0] - 1
            
        return img, dim

    def select_timestep(self, img, dim):
        if dim[0] > 1:
            idx_to_select = random.randint(0, dim[0] - 1)
            img[0, :, :, :] = img[idx_to_select, :, :, :]
            img[1:, :, :, :] = 0
            dim[0] = 1
        return img, dim
    
    def augment(self, img, dim):
        hasAugmented = False
        if dim[0] > 1 and random.random() < TIME_AUGMENTATION_PROB:
            hasAugmented = True
            if dim[0] > 2 and random.random() < 0.5:
                img, dim = self.remove_timestep(img, dim)
            else:
                img, dim = self.select_timestep(img, dim)
        
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
        p = self.dataset[idx]
        image_tensor = torch.zeros(self.config.max_time, self.config.num_channels, self.config.max_height, self.config.max_width, dtype=torch.float, device=self.device)
        dimension_tensor = torch.ones(3, dtype=torch.long)
        if len(p) > self.config.max_time:
            indices = list(range(len(p)))
            random.shuffle(indices)
            p = [p[i] for i in sorted(indices[:self.config.max_time])]
            
        chosenDim = random.choice([dim for _, dim, _, _, _ in p])
        chosenDim = (chosenDim[1], chosenDim[2])
        dimension_tensor[0] = len(p)
        for j, (path, _, _, _, labels) in enumerate(p):
            img = self.load_image(path, chosenDim)
            image_tensor[j, :, :img.shape[2], :img.shape[3]] = img
            dimension_tensor[1] = img.shape[1]
            dimension_tensor[2] = img.shape[2]
            
        if self.init_augment:
            image_tensor, dimension_tensor = self.augment(image_tensor, dimension_tensor)

        if self.downstream:
            label_tensor = torch.tensor(labels, dtype=torch.long if self.multiclass else torch.float, device=self.device)
            return image_tensor, dimension_tensor, label_tensor
        
        return image_tensor, dimension_tensor