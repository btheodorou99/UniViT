import torch.nn as nn
import torch

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

class SegmentationModel(nn.Module):
    def __init__(self, representation_dim, patch_size, slice=1, full_slice=False, slice_factor=1):
        super(SegmentationModel, self).__init__()
        self.representation_dim = representation_dim
        self.hidden_dim = 64
        self.patch_size = patch_size
        self.slice = slice
        self.full_slice = full_slice
        self.slice_factor = slice_factor
        self.deconv_schedule = self.create_deconv_schedule(self.patch_size)
        
        self.fc = nn.Linear(self.representation_dim, self.hidden_dim)
        if self.slice > 1:
            self.deconv1 = nn.ConvTranspose3d(self.hidden_dim, self.hidden_dim, kernel_size=(1 if self.full_slice else (self.slice // self.slice_factor) if self.slice_factor != 1 else self.slice, self.deconv_schedule[0], self.deconv_schedule[0]), stride=(1 if self.full_slice else (self.slice // self.slice_factor) if self.slice_factor != 1 else self.slice, self.deconv_schedule[0], self.deconv_schedule[0]), padding=0)
            self.deconv2 = nn.ConvTranspose3d(self.hidden_dim, self.hidden_dim, kernel_size=(1, self.deconv_schedule[1], self.deconv_schedule[1]), stride=(1, self.deconv_schedule[1], self.deconv_schedule[1]), padding=0)
            self.conv1 = nn.Conv3d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv3d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1)
        else:
            self.deconv1 = nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim, kernel_size=self.deconv_schedule[0], stride=self.deconv_schedule[0], padding=0)
            self.deconv2 = nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim, kernel_size=self.deconv_schedule[1], stride=self.deconv_schedule[1], padding=0)
            self.conv1 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1)
        self.out = nn.Linear(self.hidden_dim, 1)

    def create_deconv_schedule(self, patch_size):
        if patch_size == 16:
            return [4, 4]
        elif patch_size == 14:
            return [7,2]

    def forward(self, x):
        if self.slice > 1 and self.full_slice:
            bs, depth, seq_len, rep_dim = x.shape
        else:
            bs, seq_len, rep_dim = x.shape
        sqrt_seq_len = int(seq_len ** 0.5)
        x = self.fc(x)
        if self.slice == 1:
            x = x.reshape(bs, sqrt_seq_len, sqrt_seq_len, self.hidden_dim)
            x = x.permute(0, 3, 1, 2)
        elif self.full_slice:
            x = x.reshape(bs, depth, sqrt_seq_len, sqrt_seq_len, self.hidden_dim)
            x = x.permute(0, 4, 1, 2, 3)
        elif self.slice_factor != 1:
            sqrt_seq_len = int((seq_len / self.slice_factor) ** 0.5)
            x = x.reshape(bs, self.slice_factor, sqrt_seq_len, sqrt_seq_len, self.hidden_dim)
            x = x.permute(0, 4, 1, 2, 3)
        else:
            x = x.reshape(bs, sqrt_seq_len, sqrt_seq_len, self.hidden_dim)
            x = x.permute(0, 3, 1, 2)
            x = x.unsqueeze(2)
            
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.conv1(x)
        x = self.conv2(x)
        if self.slice > 1:
            x = x.permute(0, 2, 3, 4, 1)
        else:
            x = x.permute(0, 2, 3, 1)
        x = self.out(x)
        return torch.sigmoid(x.squeeze(-1))
            