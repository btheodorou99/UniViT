import torch
import torch
import torch.nn as nn

class CNNModel(nn.Module):
  def __init__(self, input_channels, input_resolution, num_labels, num_cnn_layers):
    super(CNNModel, self).__init__()
    
    self.cnn_layers = nn.Sequential()
    in_channels = input_channels
    out_channels = 64
    
    # Add CNN layers
    for i in range(num_cnn_layers):
      self.cnn_layers.add_module(f"conv{i+1}", nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
      self.cnn_layers.add_module(f"relu{i+1}", nn.ReLU(inplace=True))
      self.cnn_layers.add_module(f"pool{i+1}", nn.MaxPool2d(kernel_size=2, stride=2))
      in_channels = out_channels
      out_channels *= 2
    
    # Calculate the output size after CNN layers
    cnn_output_size = input_resolution // (2 ** num_cnn_layers)
    cnn_output_channels = out_channels // 2
    
    self.fc = nn.Linear(cnn_output_size * cnn_output_size * cnn_output_channels, 128)
    self.output = nn.Linear(128, num_labels)
  
  def forward(self, x):
    x = self.cnn_layers(x)
    x = x.view(x.size(0), -1)
    x = self.output(torch.relu(self.fc(x)))
    return x
  

class LinearModel(nn.Module):
  def __init__(self, input_dim, output_dim, num_layers):
    super(LinearModel, self).__init__()
    
    layers = []
    in_dim = input_dim
    out_dim = 128
    
    # Add linear layers
    for i in range(num_layers):
      layers.append(nn.Linear(in_dim, out_dim))
      layers.append(nn.ReLU(inplace=True))
      in_dim = out_dim
    
    layers.append(nn.Linear(in_dim, output_dim))
    
    self.model = nn.Sequential(*layers)
  
  def forward(self, x):
    return self.model(x)
