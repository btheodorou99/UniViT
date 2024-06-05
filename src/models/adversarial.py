import torch
import torch.nn as nn

class Permutation(nn.Module):
  def __init__(self, embd_dim):
    super(Permutation, self).__init__()
    self.linear1 = nn.Linear(embd_dim, embd_dim)
    self.linear2 = nn.Linear(embd_dim, embd_dim)

  def forward(self, x):
    x = self.linear1(x)
    x = torch.relu(x)
    x = self.linear2(x)
    return x