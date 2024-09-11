import time
import torch
import random
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

MIN_RESOLUTION = 32

class ImageDataset(Dataset):
    def __init__(self, dataset, config, device, multiclass=False):
        self.dataset = dataset
        self.config = config
        self.device = device
        self.multiclass = multiclass
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def adjust_size(self, origDim):
        if origDim[0] > self.config.max_height:
            scale_factor_height = self.config.max_height / origDim[1]
        else:
            scale_factor_height = 1
        if origDim[1] > self.config.max_width:
            scale_factor_width = self.config.max_width / origDim[2]
        else:
            scale_factor_width = 1

        scale_factor_image = min(scale_factor_height, scale_factor_width)
        newHeight = min(max(int(origDim[0] * scale_factor_image), MIN_RESOLUTION), self.config.max_height)
        newWidth = min(max(int(origDim[1] * scale_factor_image), MIN_RESOLUTION), self.config.max_width)
        return (newHeight, newWidth)

    def load_image(self, image_path, chosenDim):
        if image_path.endswith(".npy"):
            img = torch.tensor(np.load(image_path), dtype=torch.float)
            if len(img.shape) == 4:
                img = img[:,:,:,0]
            img = img[:, :, img.shape[2] // 2] # Take middle depth for 3D images
            img = img.unsqueeze(0).repeat(3, 1, 1)
        elif (
            image_path.endswith('.jpg') 
            or image_path.endswith('.png') 
            or image_path.endswith('.tif')
        ):
            img = Image.open(image_path).convert("RGB")
            transform = transforms.Compose(
                [
                    transforms.Resize((chosenDim[1], chosenDim[2])),
                    transforms.ToTensor(),
                ]
            )
            img = transform(img)
        else:
            raise ValueError("Invalid image format")

        currDim = tuple(img[0, :, :].shape)
        if currDim != chosenDim:
            if currDim[0] == chosenDim[1] and currDim[1] == chosenDim[0]:
                img = img.permute(0, 2, 1)
            elif (currDim[0] > chosenDim[0]) == (chosenDim[1] > currDim[1]) and (currDim[0] != chosenDim[0] and currDim[1] != chosenDim[1]):
                img = img.permute(0, 2, 1).unsqueeze(0)
                img = F.interpolate(img, size=(chosenDim[0], chosenDim[1]), mode='bilinear', align_corners=True)
                img = img.squeeze(0)
            else:
                img = img.unsqueeze(0)
                img = F.interpolate(img, size=(chosenDim[0], chosenDim[1]), mode='bilinear', align_corners=True)
                img = img.squeeze(0)

        return img

    def load_image_with_retries(self, path, chosenDim, max_attempts=5, delay=1):
        attempts = 0
        errMessage = ""
        while attempts < max_attempts:
            try:
                image_tensor = self.load_image(path, chosenDim)
                return image_tensor  # Return the loaded image if successful
            except IOError as e:
                errMessage = str(e)
                time.sleep(delay * (2 ** attempts))
        raise Exception(f"Failed to load image after {max_attempts} attempts: {errMessage}")

    def __getitem__(self, idx):
        p = self.dataset[idx]
        path, chosenDim, _, _, labels = p[-1]
        image_tensor = torch.zeros(self.config.num_channels, self.config.max_height, self.config.max_width, dtype=torch.float, device=self.device)
        dimension_tensor = torch.ones(2, dtype=torch.long)
        
        chosenDim = self.adjust_size(chosenDim)
        dimension_tensor[0] = chosenDim[0]
        dimension_tensor[1] = chosenDim[1]
        img = self.load_image_with_retries(path, chosenDim)
        image_tensor[:, :img.shape[1], :img.shape[2]] = img
            
        label_tensor = torch.tensor(labels, dtype=torch.long if self.multiclass else torch.float, device=self.device)
        return image_tensor, dimension_tensor, label_tensor