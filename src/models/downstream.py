import torch
import torch.nn as nn

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

# TODO: Linear Probing
# class DownstreamModel(nn.Module):
#   def __init__(self, input_dim, output_dim):
#     super(DownstreamModel, self).__init__()
#     self.fc = nn.Linear(input_dim, output_dim)

#   def forward(self, x):
#     return self.fc(x)
