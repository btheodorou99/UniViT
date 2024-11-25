import torch
import random
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from skimage.filters import threshold_otsu

MIN_RESOLUTION = 32

class ImageDataset(Dataset):
    def __init__(
        self, dataset, config, device, temporal=True, multiclass=False
    ):
        self.dataset = dataset
        self.config = config
        self.config.max_depth = self.config.max_height
        self.device = device
        self.temporal = temporal
        self.multiclass = multiclass
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def adjust_size(self, origDim):
        if origDim[0] > self.config.max_depth:
            newSlice = self.config.max_depth
        else:
            newSlice = origDim[0]
        if origDim[1] > self.config.max_height:
            scale_factor_height = self.config.max_height / origDim[1]
        else:
            scale_factor_height = 1
        if origDim[2] > self.config.max_width:
            scale_factor_width = self.config.max_width / origDim[2]
        else:
            scale_factor_width = 1

        scale_factor_image = min(scale_factor_height, scale_factor_width)
        newHeight = min(
            max(int(origDim[1] * scale_factor_image), MIN_RESOLUTION),
            self.config.max_height,
        )
        newWidth = min(
            max(int(origDim[2] * scale_factor_image), MIN_RESOLUTION),
            self.config.max_width,
        )
        return (newSlice, newHeight, newWidth)

    def load_3D(self, image_path, buffer=1):
        img = np.load(image_path)
        if len(img.shape) == 4:
            img = img[:, :, :, 0]
        img = img.transpose(2, 0, 1)
        threshold = threshold_otsu(img)
        mask = img >= threshold
        coords = np.argwhere(mask)
        min_coords = np.maximum(np.min(coords, axis=0) - buffer, 0)
        max_coords = np.minimum(np.max(coords, axis=0) + buffer + 1, img.shape)
        img = img[
                    min_coords[0]:max_coords[0], 
                    min_coords[1]:max_coords[1], 
                    min_coords[2]:max_coords[2]
                ]
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        return img

    def load_image(self, image_path, chosenDim):
        if image_path.endswith(".npy"):
            img = self.load_3D(image_path)
        elif (
            image_path.endswith(".jpg")
            or image_path.endswith(".png")
            or image_path.endswith(".tif")
        ):
            img = Image.open(image_path).convert("RGB")
            img = self.transform(img).unsqueeze(1)
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
                img = F.interpolate(
                    img.permute(0, 1, 3, 2).unsqueeze(0),
                    size=(chosenDim[0], chosenDim[1], chosenDim[2]),
                    mode="trilinear",
                    align_corners=True,
                ).squeeze(0)
            else:
                img = F.interpolate(
                    img.unsqueeze(0),
                    size=(chosenDim[0], chosenDim[1], chosenDim[2]),
                    mode="trilinear",
                    align_corners=True,
                ).squeeze(0)

        return img

    def __getitem__(self, idx):
        p = self.dataset[idx]
        _, _, _, _, labels = p[-1]
        label_tensor = torch.tensor(
            labels,
            dtype=torch.long if self.multiclass else torch.float,
            device=self.device,
        )
        
        chosenDim = (128, 128, 128)
        if self.temporal:         
            path1 = p[-2][0]
            path2 = p[-1][0]
            image1 = self.load_image(path1, chosenDim)[:, 0].unsqueeze(0)
            image2 = self.load_image(path2, chosenDim)[:, 0].unsqueeze(0)
            return image1, image2, label_tensor
        else:
            path = p[-1][0]
            image = self.load_image(path, chosenDim)
            return image, label_tensor
