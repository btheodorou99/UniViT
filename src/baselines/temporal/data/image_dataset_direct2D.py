import torch
import random
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

MIN_CROP = 0.8
MIN_RESOLUTION = 32
TIME_AUGMENTATION_PROB = 1 / 2
SLICE_AUGMENTATION_PROB = 1 / 2
IMAGE_AUGMENTATION_PROB = 3 / 4


class ImageDataset(Dataset):
    def __init__(
        self, dataset, config, device, augment=False, downstream=False, multiclass=False
    ):
        self.dataset = dataset
        self.config = config
        self.device = device
        self.init_augment = augment
        self.downstream = downstream
        self.multiclass = multiclass
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def load_image(self, image_path, chosenDim):
        if (
            image_path.endswith(".jpg")
            or image_path.endswith(".png")
            or image_path.endswith(".tif")
        ):
            img = Image.open(image_path).convert("RGB")
            img = self.transform(img)
        else:
            raise ValueError("Invalid image format")

        currDim = tuple(img[0, :, :].shape)
        if currDim != chosenDim:
            if (
                currDim[0] == chosenDim[1]
                and currDim[1] == chosenDim[0]
                or (currDim[0] > chosenDim[0]) == (chosenDim[1] > currDim[1])
                and (currDim[0] != chosenDim[0] and currDim[1] != chosenDim[1])
            ):
                img = img.permute(0, 2, 1)
                img = F.interpolate(
                    img.unsqueeze(0),
                    size=(chosenDim[0], chosenDim[1]),
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(0)
            else:
                img = F.interpolate(
                    img.unsqueeze(0),
                    size=(chosenDim[0], chosenDim[1]),
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(0)

        return img

    def random_crop(self, img, dim):
        crop_height = random.randint(
            int(dim[2] * MIN_CROP), img.shape[3]
        )  # Adjust range as needed
        crop_width = random.randint(
            int(dim[3] * MIN_CROP), img.shape[4]
        )  # Adjust range as needed
        i, j, h, w = transforms.RandomCrop.get_params(
            img[0, 0, 0, :, :], output_size=(crop_height, crop_width)
        )
        img[:, :, :, :crop_height, :crop_width] = img[:, :, :, i : i + h, j : j + w]
        img[:, :, :, crop_height:, :] = 0
        img[:, :, :, :, crop_width:] = 0
        dim[2] = crop_height
        dim[3] = crop_width
        return img, dim

    def random_resize(self, img, dim):
        min_scale = max(MIN_RESOLUTION / dim[2], MIN_RESOLUTION / dim[3])
        max_scale = min(self.config.max_height / dim[2], self.config.max_width / dim[3])
        scale = random.uniform(min_scale, max_scale)
        height = min(max(int(dim[2] * scale), MIN_RESOLUTION), self.config.max_height)
        width = min(max(int(dim[3] * scale), MIN_RESOLUTION), self.config.max_width)
        img[:, :, :, :height, :width] = F.interpolate(
            img[:, :, :, : dim[2], : dim[3]].view(
                -1, self.config.num_channels, dim[2], dim[3]
            ),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).view(img.shape[0], img.shape[1], self.config.num_channels, height, width)
        img[:, :, :, height:, :] = 0
        img[:, :, :, :, width:] = 0
        dim[2] = height
        dim[3] = width
        return img, dim

    def remove_timestep(self, img, dim):
        if dim[0] > 2:  # Check if there are multiple time steps
            idx_to_remove = random.randint(0, dim[0] - 1)
            img[idx_to_remove:-1, :, :, :, :] = img[
                idx_to_remove + 1 :, :, :, :, :
            ].clone()
            img[-1, :, :, :, :] = 0
            dim[0] = dim[0] - 1

        return img, dim

    def select_timestep(self, img, dim):
        if dim[0] > 1:
            idx_to_select = random.randint(0, dim[0] - 1)
            img[0, :, :, :, :] = img[idx_to_select, :, :, :, :]
            img[1:, :, :, :, :] = 0
            dim[0] = 1
        return img, dim

    def depth_interpolation(self, img, dim):
        # Interpolating depths if there are multiple, adjusting to a specific number of depths
        if dim[1] > 2:
            depths = random.randint(
                2, dim[1]
            )  # Target number of depths, adjust as needed
            img = img.permute(0, 2, 1, 3, 4)
            img[:, :, :depths, : dim[2], : dim[3]] = F.interpolate(
                img[:, :, : dim[1], : dim[2], : dim[3]],
                size=(depths, dim[2], dim[3]),
                mode="trilinear",
                align_corners=False,
            )
            img = img.permute(0, 2, 1, 3, 4)
            img[:, depths:, :, :, :] = 0
            dim[1] = depths
        return img, dim

    def select_depth(self, img, dim):
        if dim[1] > 1:
            idx = random.randint(0, dim[1] - 1)
            img[:, 0, :, :, :] = img[:, idx, :, :, :]
            img[:, 1:, :, :, :] = 0
            dim[1] = 1
        return img, dim

    def augment(self, img, dim):
        hasAugmented = False
        if dim[0] > 1 and random.random() < TIME_AUGMENTATION_PROB:
            hasAugmented = True
            if dim[0] > 2 and random.random() < 0.5:
                img, dim = self.remove_timestep(img, dim)
            else:
                img, dim = self.select_timestep(img, dim)

        if dim[1] > 1 and random.random() < SLICE_AUGMENTATION_PROB:
            hasAugmented = True
            if dim[1] > 2 and random.random() < 0.5:
                img, dim = self.depth_interpolation(img, dim)
            else:
                img, dim = self.select_depth(img, dim)

        if not hasAugmented or random.random() < IMAGE_AUGMENTATION_PROB:
            if random.random() < 0.5:
                img, dim = self.random_resize(img, dim)
            else:
                img, dim = self.random_crop(img, dim)

        return img, dim

    def augment_batch(self, img, dim):
        img = img.clone()
        dim = dim.clone()
        for i in range(img.shape[0]):
            img[i], dim[i] = self.augment(img[i], dim[i])

        return img, dim

    def __getitem__(self, idx):
        p = self.dataset[idx]
        _, _, _, _, labels = p[-1]
        path1 = p[-2][0]
        path2 = p[-1][0]
        chosenDim = (224, 224)
        image1 = self.load_image(path1, chosenDim)
        image2 = self.load_image(path2, chosenDim)
        if self.downstream:
            label_tensor = torch.tensor(
                labels,
                dtype=torch.long if self.multiclass else torch.float,
                device=self.device,
            )
            return image1, image2, label_tensor

        return image1, image2
