import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding3D(nn.Module):
    def __init__(self, out_dim, temporal=False):
        super(Embedding3D, self).__init__()
        self.slice = 16
        self.image_dim = 128

        self.conv1 = nn.Conv3d(
            2 if temporal else 1, 16, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.flat_dim = (
            64 * (self.slice // 8) * (self.image_dim // 8) * (self.image_dim // 8)
        )
        self.fc1 = nn.Linear(self.flat_dim, 768)
        self.fc2 = nn.Linear(768, 768)
        self.out = nn.Linear(768, out_dim)

    def forward(self, x):
        # Convolution + ReLU + MaxPooling
        x = F.relu(self.conv1(x))
        x = F.max_pool3d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool3d(x, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool3d(x, 2)

        # Flattening the output
        x = x.view(-1, self.flat_dim)

        # Passing through the fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


class Embedding2D(nn.Module):
    def __init__(self, out_dim, temporal=False):
        super(Embedding2D, self).__init__()
        self.image_dim = 128

        self.conv1 = nn.Conv2d(
            2 if temporal else 1, 16, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.flat_dim = 64 * (self.image_dim // 8) * (self.image_dim // 8)
        self.fc1 = nn.Linear(self.flat_dim, 768)
        self.fc2 = nn.Linear(768, 768)
        self.out = nn.Linear(768, out_dim)

    def forward(self, x):
        # Convolution + ReLU + MaxPooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        # Flattening the output
        x = x.view(-1, self.flat_dim)

        # Passing through the fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
