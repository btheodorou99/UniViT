import torch
import random
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

SEGMENTATION_THRESHOLD = 0.1

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
            if channels:
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

        return img

    def __getitem__(self, idx):
        path, dim, _, _, labels = self.dataset[idx]
        image_tensor = torch.zeros(
            self.config.max_depth,
            self.config.num_channels,
            self.config.max_height,
            self.config.max_width,
            dtype=torch.float,
            device=self.device,
        )
        dimension_tensor = torch.ones(3, dtype=torch.long)
        chosenDim = (self.config.max_depth, self.config.max_height, self.config.max_width)
        img = self.load_image(path, chosenDim)
        image_tensor[: img.shape[0], :, : img.shape[2], : img.shape[3]] = img
        dimension_tensor[1] = img.shape[0]
        dimension_tensor[2] = img.shape[2]
        dimension_tensor[3] = img.shape[3]

        label_tensor = (self.load_image(labels, chosenDim, channels=False) > SEGMENTATION_THRESHOLD).float()
        return image_tensor, dimension_tensor, label_tensor