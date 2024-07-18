
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from src.config import Config
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from src.models.univit_simple import UniViT
from src.data.image_dataset import ImageDataset

model_key = 'univit_simple'

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

config = Config()
cuda_num = 0
device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)

def labelData(p):
    labels = []
    labels.append(p[0][3])
    if len(p) > 1:
        labels.append('Temporal')
    else:
        labels.append('Static')
    if p[0][1][0] > 1:
        labels.append('3D')
    else:
        labels.append('2D')
    return labels

data_dir = '/shared/eng/bpt3/data/UniViT/data'
save_dir = '/shared/eng/bpt3/data/UniViT/save'
visualization_data = pickle.load(open(f'{data_dir}/visualizationDataset.pkl', 'rb'))
labels = [labelData(p) for p in visualization_data]

model = UniViT(config.max_height, 
               config.max_width, 
               config.max_time, 
               config.max_slice, 
               config.num_channels, 
               config.patch_size, 
               config.representation_size, 
               config.num_layers, 
               config.num_heads, 
               config.projection_size, 
               config.mlp_dim, 
               config.dropout, 
               config.attention_dropout,
               config.mask_prob).to(device)
print("Loading previous model")
model.load_state_dict(torch.load(f'{save_dir}/{model_key}.pt', map_location='cpu')['model'], strict=False)
model.eval()
model.requires_grad_(False)

embeddings = []
visualization_data = ImageDataset(visualization_data, config, 'cpu', augment=False)
visualization_loader = DataLoader(visualization_data, batch_size=config.downstream_batch_size, shuffle=False, num_workers=config.num_workers)
for batch_images, batch_dimensions in tqdm(visualization_loader, desc=f'Generating Embeddings', leave=False):
    batch_images = batch_images.to(device)
    batch_labels = batch_labels.to(device)
    with torch.no_grad():
        representations = model.embed(batch_images, batch_dimensions)
    
    for i in range(len(representations)):
        embeddings.append(representations[i].cpu().clone().numpy())


tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
reduced_embeddings = tsne.fit_transform(embeddings)
for i in range(len(labels[0])):
    sublabels = [l[i] for l in labels]
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=sublabels, cmap='viridis')
    plt.colorbar()
    plt.savefig(f'{save_dir}/{model_key}_visualization_{i}.png')