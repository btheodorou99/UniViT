import time
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(
        self,
        dataset,
        config,
        device,
        transform=None,
        image_depth=False,
        multiclass=False,
    ):
        self.dataset = dataset
        self.config = config
        self.device = device
        self.transform = transform
        self.image_depth = image_depth
        self.multiclass = multiclass

    def __len__(self):
        return len(self.dataset)

    def load_image_3D(self, image_path):
        if image_path.endswith(".npy"):
            img = torch.tensor(np.load(image_path), dtype=torch.float)
            if len(img.shape) == 4:
                img = img[:, :, :, 0]
            img = img.permute(2, 0, 1).unsqueeze(0)  # Only 1 channel
            img = F.interpolate(
                img.unsqueeze(0),
                size=(self.config.max_depth, self.config.max_height, self.config.max_width),
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
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
    
    def load_image_with_retries(self, path, max_attempts=5, delay=1):
        attempts = 0
        errMessage = ""
        while attempts < max_attempts:
            try:
                with open(path, 'rb') as f:
                    image_tensor = (
                        self.load_image(path)
                        if not self.image_depth
                        else self.load_image_3D(path)
                    )
                return image_tensor  # Return the loaded image if successful
            except IOError as e:
                errMessage = str(e)
                time.sleep(delay * (2 ** attempts))
        raise Exception(f"Failed to load image after {max_attempts} attempts: {errMessage}")

    def __getitem__(self, idx):
        path, _, _, _, labels = self.dataset[idx]
        image_tensor = self.load_image_with_retries(path)
        label_tensor = torch.tensor(
            labels,
            dtype=torch.long if self.multiclass else torch.float,
            device=self.device,
        )
        return image_tensor, label_tensor
