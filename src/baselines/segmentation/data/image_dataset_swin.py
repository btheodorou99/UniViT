import torch
import random
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

SEGMENTATION_THRESHOLD = 0.05

class ImageDataset(Dataset):
    def __init__(self, dataset, config, device):
        self.dataset = dataset
        self.config = config
        self.device = device
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def load_image(self, image_path, chosenDim, channels=True):
        if image_path.endswith(".npy"):
            img = torch.tensor(np.load(image_path), dtype=torch.float)
            if len(img.shape) == 4:
                img = img[:, :, :, 0]
            img = img.permute(2, 0, 1)
            img = img.unsqueeze(1).repeat(1, 3, 1, 1)
        else:
            raise ValueError("Invalid image format")

        currDim = tuple(img[:, 0, :, :].shape)
        if currDim != chosenDim:
            if (
                (currDim[1] == chosenDim[2]
                and currDim[2] == chosenDim[1])
                or (currDim[1] > chosenDim[1]) == (chosenDim[2] > currDim[2])
                and (currDim[1] != chosenDim[1] and currDim[2] != chosenDim[2])
            ):
                img = img.permute(1, 0, 3, 2).unsqueeze(0)
                img = F.interpolate(
                    img,
                    size=(chosenDim[0], chosenDim[1], chosenDim[2]),
                    mode="trilinear",
                    align_corners=True,
                )
                img = img.squeeze(0).permute(1, 0, 2, 3)
            else:
                img = img.permute(1, 0, 2, 3).unsqueeze(0)
                img = F.interpolate(
                    img,
                    size=(chosenDim[0], chosenDim[1], chosenDim[2]),
                    mode="trilinear",
                    align_corners=True,
                )
                img = img.squeeze(0).permute(1, 0, 2, 3)
                
        if not channels:
            img = img[:, 0, :, :]
            
        return img

    def __getitem__(self, idx):
        p = self.dataset[idx]
        chosenDim = (self.config.segmentation_depth, self.config.max_height, self.config.max_width)
        path, _, _, _, labels = p[0]
        
        guide_img = self.load_image(path, chosenDim)
        img = F.pad(guide_img[:,0,:,:], (0, 0, 0, 0, self.config.max_depth // 2, self.config.max_depth // 2))
        img = img.unfold(dimension=0, size=self.config.max_depth, step=1).permute(0, 3, 1, 2).unsqueeze(1)

        label_tensor = (self.load_image(labels, chosenDim, channels=False) > SEGMENTATION_THRESHOLD).float()
        return img, label_tensor, guide_img