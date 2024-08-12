import torch
import random
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

SEGMENTATION_THRESHOLD = 0.1

class ImageDataset(Dataset):
    def __init__(
        self,
        dataset,
        config,
        device,
        patch_size=None,
        processing_fn=None,
        image_size=None,
        image_depth=None,
    ):
        assert (
            patch_size is not None or processing_fn is not None
        ), "Either patch_size or processing_fn must be provided"
        self.dataset = dataset
        self.config = config
        self.device = device
        self.patch_size = patch_size
        self.image_size = image_size
        self.image_depth = image_depth
        if self.patch_size is None:
            self.transform = processing_fn
            self.toImg = transforms.ToPILImage()
        elif self.image_size is not None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )

    def __len__(self):
        return len(self.dataset)

    def load_image_3D(self, image_path, channels=True):
        if image_path.endswith(".npy"):
            img = torch.tensor(np.load(image_path), dtype=torch.float)
            if len(img.shape) == 4:
                img = img[:, :, :, 0]
            img = img.permute(2, 0, 1)
            if channels:
                img = img.unsqueeze(0)  # Only 1 channel
            if self.patch_size is None:
                img = self.transform(self.toImg(img))
            elif self.image_size is not None:
                img = F.interpolate(
                    img.unsqueeze(0),
                    size=(self.image_depth, self.image_size, self.image_size),
                    mode="trilinear",
                    align_corners=False,
                ).squeeze(0)
        else:
            raise ValueError("Invalid 3D image format")

        return img

    def load_image(self, image_path):
        if image_path.endswith(".npy"):
            img = torch.tensor(np.load(image_path), dtype=torch.float)
            if len(img.shape) == 4:
                img = img[:, :, :, 0]
            img = img[:, :, img.shape[2] // 2]  # Take middle depth for 3D images
            img = img.unsqueeze(0).repeat(3, 1, 1)  # Repeat for 3 channels
            if self.patch_size is None:
                img = self.transform(self.toImg(img))
            elif self.image_size is not None:
                img = F.interpolate(
                    img.unsqueeze(0),
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
        elif (
            image_path.endswith(".jpg")
            or image_path.endswith(".png")
            or image_path.endswith(".tif")
        ):
            img = Image.open(image_path).convert("RGB")
            img = self.transform(img)
        else:
            raise ValueError("Invalid image format")

        if self.patch_size is not None and img.shape[1] // self.patch_size != 0:
            img = F.interpolate(
                img.unsqueeze(0),
                size=(
                    img.shape[1] // self.patch_size * self.patch_size,
                    img.shape[2] // self.patch_size * self.patch_size,
                ),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

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
