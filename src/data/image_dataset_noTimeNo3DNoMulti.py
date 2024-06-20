import torch
import random
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

MIN_CROP = 0.5

class ImageDataset(Dataset):
    def __init__(self, dataset, config, device, augment=False, downstream=False, multiclass=False):
        self.dataset = dataset
        self.config = config
        self.device = device
        self.init_augment = augment
        self.downstream = downstream
        self.multiclass = multiclass
        self.image_size = config.max_height
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)
      
    def load_image(self, image_path):
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

        return img
    
    def random_crop(self, img):
        crop_height = random.randint(int(self.image_size * MIN_CROP), img.shape[1])  # Adjust range as needed
        crop_width = random.randint(int(self.image_size * MIN_CROP), img.shape[2])  # Adjust range as needed
        i, j, h, w = transforms.RandomCrop.get_params(img[0, :, :], output_size=(crop_height, crop_width))
        img = img[:, i:i+h, j:j+w]
        img = F.interpolate(img.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear', align_corners=True).squeeze(0)
        return img
    
    def augment(self, img):
        img = self.random_crop(img)  
        return img
        
    def augment_batch(self, img):
        for i in range(img.shape[0]):
            img[i] = self.augment(img[i])
            
        return img

    def __getitem__(self, idx):
        path, _, _, _, labels = self.dataset[idx]
        image_tensor = self.load_image(path)
            
        if self.init_augment:
            image_tensor = self.augment(image_tensor)
                    
        if self.downstream:
            label_tensor = torch.tensor(labels, dtype=torch.long if self.multiclass else torch.float, device=self.device)
            return image_tensor, label_tensor
        
        return image_tensor