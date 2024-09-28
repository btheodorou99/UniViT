import torch
import random
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

SEGMENTATION_THRESHOLD = 0.05

class ImageDataset(Dataset):
    def __init__(
        self,
        dataset,
        config,
        device,
        transform=None,
        image_depth=False,
    ):
        self.dataset = dataset
        self.config = config
        self.device = device
        self.transform = transform
        self.image_depth = image_depth

    def __len__(self):
        return len(self.dataset)

    def load_image_3D(self, image_path):
        if image_path.endswith(".npy"):
            img = np.load(image_path)
            if len(img.shape) == 4:
                img = img[:, :, :, 0]
            img = img.transpose(2, 0, 1)
            img = (img * 255).astype(np.uint8)
            tensor = []
            for i in range(img.shape[0]):
                tensor.append(self.transform(Image.fromarray(img[i]).convert("RGB")))
            img = torch.stack(tensor)
            img = F.interpolate(
                img.permute(1,0,2,3).unsqueeze(0),
                size=(self.image_depth, img.shape[2], img.shape[3]),
                mode="trilinear",
                align_corners=False,
            ).squeeze(0).permute(1,0,2,3)
        else:
            raise ValueError("Invalid 3D image format")

        return img

    def load_image(self, image_path):
        if image_path.endswith(".npy"):
            img = np.load(image_path)
            if len(img.shape) == 4:
                img = img[:, :, :, 0]
            img = img[:, :, img.shape[2] // 2]  # Take middle depth for 3D images
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

        img = self.transform(img)
        return img
    
    def load_image_labels(self, image_path, chosenDim, channels=True):
        if image_path.endswith(".npy"):
            img = torch.tensor(np.load(image_path), dtype=torch.float)
            if len(img.shape) == 4:
                img = img[:, :, :, 0]
            img = img.permute(2, 0, 1)
            img = img.unsqueeze(1).repeat(1, 3, 1, 1)
        elif (
            image_path.endswith(".jpg")
            or image_path.endswith(".png")
            or image_path.endswith(".tif")
        ):
            img = Image.open(image_path)
            if channels:
                img = img.convert("RGB")
            img = self.transform(img).unsqueeze(0)
        else:
            raise ValueError("Invalid image format")

        currDim = tuple(img[:, 0, :, :].shape)
        if currDim != chosenDim:
            if (
                currDim[1] == chosenDim[2]
                and currDim[2] == chosenDim[1]
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
        path, _, _, _, labels = self.dataset[idx]
        image_tensor = (
            self.load_image_3D(path)
            if self.image_depth
            else self.load_image(path)
        )
        
        chosenDim = (self.config.max_depth, self.config.max_height, self.config.max_width)
        label_tensor = (self.load_image_labels(labels, chosenDim, channels=False) > SEGMENTATION_THRESHOLD).float()
        return image_tensor, label_tensor
