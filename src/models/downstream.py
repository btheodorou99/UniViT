import torch.nn as nn
import torch
from collections import OrderedDict

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

class DownstreamModel(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(DownstreamModel, self).__init__()
    self.fc1 = nn.Linear(input_dim, 64)
    self.fc2 = nn.Linear(64, 32)
    self.fc3 = nn.Linear(32, output_dim)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x

class SegmentationModel(nn.Module):

    def __init__(self, representation_dim, patch_size, depth):
        super(SegmentationModel, self).__init__()
        self.representation_dim = representation_dim
        self.hidden_dim = 128
        self.patch_size = patch_size
        self.depth = depth
        self.deconv_schedule = self.create_deconv_schedule(self.patch_size)

        self.fc = nn.Linear(self.representation_dim, self.hidden_dim)
        self.deconv1 = nn.ConvTranspose3d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=(1, self.deconv_schedule[0], self.deconv_schedule[0]),
            stride=(1, self.deconv_schedule[0], self.deconv_schedule[0]),
            padding=0,
        )
        self.norm1 = nn.BatchNorm3d(self.hidden_dim)
        self.deconv2 = nn.ConvTranspose3d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=(1, self.deconv_schedule[1], self.deconv_schedule[1]),
            stride=(1, self.deconv_schedule[1], self.deconv_schedule[1]),
            padding=0,
        )
        self.norm2 = nn.BatchNorm3d(self.hidden_dim)
        self.conv1 = nn.Conv3d(
            self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1
        )
        self.norm3 = nn.BatchNorm3d(self.hidden_dim)
        self.conv2 = nn.Conv3d(
            self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1
        )
        self.norm4 = nn.BatchNorm3d(self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, 1)

    def create_deconv_schedule(self, patch_size):
        if patch_size == 16:
            return [4, 4]
        elif patch_size == 14:
            return [7,2]

    def forward(self, x):
        bs, depth, seq_len, _ = x.shape
        sqrt_seq_len = int(seq_len ** 0.5)
        x = torch.relu(self.fc(x))
        x = x.reshape(bs, depth, sqrt_seq_len, sqrt_seq_len, self.hidden_dim)
        x = x.permute(0, 4, 1, 2, 3)

        x = torch.relu(self.norm1(self.deconv1(x)))
        x = torch.relu(self.norm2(self.deconv2(x)))
        x = torch.relu(self.norm3(self.conv1(x)))
        x = torch.relu(self.norm4(self.conv2(x)))
        x = x.permute(0, 2, 3, 4, 1)

        x = torch.sigmoid(self.out(x).squeeze(-1))
        return x


class SegmentationModelWImage(nn.Module):
    def __init__(self, representation_dim, patch_size, depth, image_channels=3):
        super(SegmentationModelWImage, self).__init__()
        self.representation_dim = representation_dim
        self.hidden_dim = 128
        self.patch_size = patch_size
        self.depth = depth
        self.image_channels = image_channels
        self.deconv_schedule = self.create_deconv_schedule(self.patch_size)

        # Layers for patch embeddings
        self.fc = nn.Linear(self.representation_dim, self.hidden_dim)

        # Layers for full image
        self.image_conv = nn.Conv3d(
            self.image_channels, self.hidden_dim, kernel_size=3, stride=1, padding=1
        )
        self.image_norm = nn.BatchNorm3d(self.hidden_dim)

        # Shared layers after fusion
        self.deconv1 = nn.ConvTranspose3d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=(1, self.deconv_schedule[0], self.deconv_schedule[0]),
            stride=(1, self.deconv_schedule[0], self.deconv_schedule[0]),
            padding=0,
        )
        self.norm1 = nn.BatchNorm3d(self.hidden_dim)
        self.deconv2 = nn.ConvTranspose3d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=(1, self.deconv_schedule[1], self.deconv_schedule[1]),
            stride=(1, self.deconv_schedule[1], self.deconv_schedule[1]),
            padding=0,
        )
        self.norm2 = nn.BatchNorm3d(self.hidden_dim)
        self.conv1 = nn.Conv3d(
            self.hidden_dim * 2, self.hidden_dim, kernel_size=3, stride=1, padding=1
        )
        self.norm3 = nn.BatchNorm3d(self.hidden_dim)
        self.conv2 = nn.Conv3d(
            self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1
        )
        self.norm4 = nn.BatchNorm3d(self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, 1)

    def create_deconv_schedule(self, patch_size):
        if patch_size == 16:
            return [4, 4]
        elif patch_size == 14:
            return [7, 2]
        elif patch_size == 32:
            return [8, 4]

    def forward(self, x, full_image):
        """
        x: Patch embeddings (B, depth, seq_len, representation_dim)
        full_image: Full input image (B, C, depth, H, W)
        """
        # Process patch embeddings
        bs, depth, seq_len, _ = x.shape
        sqrt_seq_len = int(seq_len**0.5)
        x = torch.relu(self.fc(x))
        x = x.reshape(bs, depth, sqrt_seq_len, sqrt_seq_len, self.hidden_dim)
        x = x.permute(
            0, 4, 1, 2, 3
        )  # (B, hidden_dim, depth, sqrt_seq_len, sqrt_seq_len)
        full_image = full_image.permute(0, 2, 1, 3, 4)

        # Process full image
        full_image = torch.relu(self.image_norm(self.image_conv(full_image)))

        # Deconvolve the patch embeddings
        x = torch.relu(self.norm1(self.deconv1(x)))
        x = torch.relu(self.norm2(self.deconv2(x)))

        # Concatenate along the channel dimension
        x = torch.cat((x, full_image), dim=1)  # (B, hidden_dim * 2, depth, H, W)

        # Process the concatenated tensor
        x = torch.relu(self.norm3(self.conv1(x)))
        x = torch.relu(self.norm4(self.conv2(x)))
        x = x.permute(0, 2, 3, 4, 1)  # (B, depth, H, W, hidden_dim)

        x = torch.sigmoid(self.out(x).squeeze(-1))  # (B, depth, H, W)
        return x
