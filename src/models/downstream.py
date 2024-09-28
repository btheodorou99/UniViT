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
    def __init__(self, representation_dim, patch_size, depth=1, full_slice=False, slices=1):
        super(SegmentationModel, self).__init__()
        self.representation_dim = representation_dim
        self.hidden_dim = 128
        self.patch_size = patch_size
        self.depth = depth
        self.full_slice = full_slice
        self.slices = slices
        self.deconv_schedule = self.create_deconv_schedule(self.patch_size)
        
        self.fc = nn.Linear(self.representation_dim, self.hidden_dim)
        if self.depth > 1:
            self.deconv1 = nn.ConvTranspose3d(
                            self.hidden_dim, self.hidden_dim, 
                            kernel_size=(1 if self.full_slice else self.depth // self.slices, self.deconv_schedule[0], self.deconv_schedule[0]), 
                            stride=(1 if self.full_slice else self.depth // self.slices, self.deconv_schedule[0], self.deconv_schedule[0]), 
                            padding=0
                        )
            self.norm1 = nn.BatchNorm3d(self.hidden_dim)
            self.deconv2 = nn.ConvTranspose3d(
                            self.hidden_dim, self.hidden_dim, 
                            kernel_size=(1, self.deconv_schedule[1], self.deconv_schedule[1]), 
                            stride=(1, self.deconv_schedule[1], self.deconv_schedule[1]), 
                            padding=0
                        )
            self.norm2 = nn.BatchNorm3d(self.hidden_dim)
            self.conv1 = nn.Conv3d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1)
            self.norm3 = nn.BatchNorm3d(self.hidden_dim)
            self.conv2 = nn.Conv3d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1)
            self.norm4 = nn.BatchNorm3d(self.hidden_dim)
        else:
            self.deconv1 = nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim, kernel_size=self.deconv_schedule[0], stride=self.deconv_schedule[0], padding=0)
            self.norm1 = nn.BatchNorm2d(self.hidden_dim)
            self.deconv2 = nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim, kernel_size=self.deconv_schedule[1], stride=self.deconv_schedule[1], padding=0)
            self.norm2 = nn.BatchNorm2d(self.hidden_dim)
            self.conv1 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1)
            self.norm3 = nn.BatchNorm2d(self.hidden_dim)
            self.conv2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1)
            self.norm4 = nn.BatchNorm2d(self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, 1)

    def create_deconv_schedule(self, patch_size):
        if patch_size == 16:
            return [4, 4]
        elif patch_size == 14:
            return [7,2]

    def forward(self, x):
        if self.depth > 1 and self.full_slice:
            bs, depth, seq_len, _ = x.shape
        else:
            bs, seq_len, _ = x.shape
            
        sqrt_seq_len = int(seq_len ** 0.5)
        x = torch.relu(self.fc(x))
        
        if self.depth == 1:
            x = x.reshape(bs, sqrt_seq_len, sqrt_seq_len, self.hidden_dim)
            x = x.permute(0, 3, 1, 2)
        elif self.full_slice:
            x = x.reshape(bs, depth, sqrt_seq_len, sqrt_seq_len, self.hidden_dim)
            x = x.permute(0, 4, 1, 2, 3)
        elif self.slices != 1:
            sqrt_seq_len = int((seq_len / self.slices) ** 0.5)
            x = x.reshape(bs, self.slices, sqrt_seq_len, sqrt_seq_len, self.hidden_dim)
            x = x.permute(0, 4, 1, 2, 3)
        else:
            x = x.reshape(bs, sqrt_seq_len, sqrt_seq_len, self.hidden_dim)
            x = x.permute(0, 3, 1, 2)
            x = x.unsqueeze(2)
            
        x = torch.relu(self.norm1(self.deconv1(x)))
        x = torch.relu(self.norm2(self.deconv2(x)))
        x = torch.relu(self.norm3(self.conv1(x)))
        x = torch.relu(self.norm4(self.conv2(x)))
        
        if self.depth > 1:
            x = x.permute(0, 2, 3, 4, 1)
        else:
            x = x.permute(0, 2, 3, 1)
            
        x = torch.sigmoid(self.out(x).squeeze(-1))
        return x