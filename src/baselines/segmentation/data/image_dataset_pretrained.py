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
        image_depth=None,
    ):
        self.dataset = dataset
        self.config = config
        self.device = device
        self.transform = transform
        self.image_depth = image_depth

    def __len__(self):
        return len(self.dataset)

    def load_image_3D(self, image_path, channels=True):
        if image_path.endswith(".npy"):
            img = np.load(image_path)
            if len(img.shape) == 4:
                img = img[:, :, :, 0]
            img = img.permute(2, 0, 1)
            tensor = []
            for i in range(img.shape[0]):
                tensor.append(self.transform(Image.fromarray(img[i])))
            img = torch.stack(tensor)
            img = img.unsqueeze(0)  # Only 1 channel
            if self.image_depth is not None:
                img = F.interpolate(
                    img.unsqueeze(0),
                    size=(self.image_depth, img.shape[2], img.shape[3]),
                    mode="trilinear",
                    align_corners=False,
                ).squeeze(0)
            if not channels:
                img = img.squeeze(0)
        else:
            raise ValueError("Invalid 3D image format")

        return img

    def load_image(self, image_path):
        if image_path.endswith(".npy"):
            img = np.load(image_path)
            if len(img.shape) == 4:
                img = img[:, :, :, 0]
            img = img[:, :, img.shape[2] // 2]  # Take middle depth for 3D images
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

    def __getitem__(self, idx):
        path, _, _, _, labels = self.dataset[idx]
        image_tensor = (
            self.load_image(path)
            if self.image_depth is None
            else self.load_image_3D(path)
        )
        label_tensor = (self.load_image_3D(labels, channels=False) > SEGMENTATION_THRESHOLD).float()
        return image_tensor, label_tensor
