import torch
import random
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, dataset, config, device, patch_size=None, processing_fn=None, image_size=None, augment=False, downstream=False, multiclass=False):
        assert patch_size is not None or processing_fn is not None, 'Either patch_size or processing_fn must be provided'
        self.dataset = dataset
        self.config = config
        self.device = device
        self.patch_size = patch_size
        self.image_size = image_size
        self.init_augment = augment
        self.downstream = downstream
        self.multiclass = multiclass
        if self.patch_size is None:
            self.transform = processing_fn
            self.toImg = transforms.ToPILImage()
        elif self.image_size is not None:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.dataset)
      
    def load_image(self, image_path):
        if image_path.endswith('.npy'):
            img = torch.tensor(np.load(image_path), dtype=torch.float)
            if len(img.shape) == 4:
                img = img[:,:,:,0]
            img = img[:,:,img.shape[2] // 2] # Take middle depth for 3D images
            img = img.unsqueeze(0).repeat(3, 1, 1) # Repeat for 3 channels
            if self.patch_size is None:
                img = self.transform(self.toImg(img))
            elif self.image_size is not None:
                img = F.interpolate(img.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False).squeeze(0)
        elif image_path.endswith('.jpg') or image_path.endswith('.png') or image_path.endswith('.tif'):
            img = Image.open(image_path).convert('RGB')
            img = self.transform(img)
        else: 
            raise ValueError('Invalid image format')

        if self.patch_size is not None and img.shape[1] // self.patch_size != 0:
            img = F.interpolate(img.unsqueeze(0), size=(img.shape[1] // self.patch_size * self.patch_size, img.shape[2] // self.patch_size * self.patch_size), mode='bilinear', align_corners=False).squeeze(0)
            
        return img
    
    def augment(self, img):
        if random.random() < 0.5:
            img = self.random_resize(img)
        else:
            img = self.random_crop(img)
                
        return img
        
    def augment_batch(self, img):
        img = img.clone()
        for i in range(img.shape[0]):
            img[i] = self.augment(img[i])
            
        return img

    def __getitem__(self, idx):
        p = self.dataset[idx]
        _, _, _, _, labels = p[-1]
        path, _, _, _, _ = p[-2]
        image_tensor = self.load_image(path)
        if self.init_augment:
            image_tensor = self.augment(image_tensor)
                    
        if self.downstream:
            label_tensor = torch.tensor(labels, dtype=torch.long if self.multiclass else torch.float, device=self.device)
            return image_tensor, label_tensor
        
        return image_tensor