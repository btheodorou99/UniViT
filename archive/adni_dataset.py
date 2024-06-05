import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, dataset, config, device, shape, label_idx):
        self.dataset = dataset
        self.config = config
        self.device = device
        self.resolution = shape[0]
        self.channels = shape[1]
        self.label_idx = label_idx
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
            img = img.permute(2, 0, 1)
            img = F.interpolate(img.unsqueeze(0).unsqueeze(0), size=(self.channels, self.resolution, self.resolution), mode='trilinear', align_corners=True).squeeze(0).squeeze(0)
            
        else:
            raise ValueError('Invalid image format')
        
        return img

    def __getitem__(self, idx):
        path, labels = self.dataset[idx]
        image_tensor = self.load_image(path)
        label_tensor = torch.tensor([labels[i] for i in range(len(labels)) if i in self.label_idx], dtype=torch.float, device=self.device)
        ehr_tensor = torch.tensor([labels[i] for i in range(len(labels)) if i not in self.label_idx and i > 50], dtype=torch.float, device=self.device)
        return image_tensor, ehr_tensor, label_tensor