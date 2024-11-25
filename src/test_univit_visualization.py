import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from src.config import Config
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from src.models.univit import UniViT
from src.data.image_dataset import ImageDataset

import matplotlib
matplotlib.use('Agg')

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

model_key = "univit"

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

config = Config()
cuda_num = 0
device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

modalityMap = {
    'ADNI PET': 'PET',
    'BraTS-GLI': 'Brain MRI',
    'BraTS-GoAT': 'Brain MRI',
    'BraTS-MEN-RT': 'Brain MRI',
    'BraTS-MET': 'Brain MRI',
    'BraTS-PED': 'Brain MRI',
    'BraTS-Path': 'Tissue Sample',
    'CRC-HE': 'Tissue Sample',
    'CheXpert': 'Chest X-ray',
    'DeepLesion': 'CT',
    'MIMIC-CXR': 'Chest X-ray',
}

def labelDatasetData(p):
    label = p[0][3]
    return label

def labelModalityData(p):
    label = p[0][3]
    return modalityMap[label]

def labelDepthData(p):
    if p[0][1][0] > 1:
        return "3D"
    else:
        return "2D"
    
def labelTimeData(p):
    if len(p) > 1:
        return "Temporal"
    else:
        return "Static"
    
def labelPreciseDepthData(p):
    numSlices = min(config.max_depth, p[-1][1][0])
    return f'{numSlices} Slice' + ('s' if numSlices > 1 else '')
    
def labelPreciseTimeData(p):
    numSteps = min(config.max_time, len(p))
    return f'{numSteps} Image' + ('s' if numSteps > 1 else '')

def labelDepthTimeData(p):
    labels = []
    if len(p) > 1:
        labels.append("Temporal")
    else:
        labels.append("Static")
    if p[0][1][0] > 1:
        labels.append("3D")
    else:
        labels.append("2D")
    labels = f'{labels[0]} {labels[1]}'
    return labels

def labelAllData(p):
    labels = []
    labels.append(p[0][3])
    if len(p) > 1:
        labels.append("Temporal")
    else:
        labels.append("Static")
    if p[0][1][0] > 1:
        labels.append("3D")
    else:
        labels.append("2D")
    labels = f'{labels[0]} {labels[1]} {labels[2]}'
    return labels

def labelsToColors(labels):
    labelSet = list(set(labels))
    colors = [labelSet.index(label) for label in labels]
    return colors

def plot_tsne(embeddings, labels, title, path):    
    # Create a scatter plot of the t-SNE embeddings
    plt.figure(figsize=(10, 8))

    # Get unique labels for coloring
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    
    # Plot each class with a different color
    for label in unique_labels:
        indices = np.where(labels == label)
        plt.scatter(embeddings[indices, 0], embeddings[indices, 1], label=label, marker='o', s=10)

    plt.xlim([embeddings[:, 0].min() - 1, embeddings[:, 0].max() + 1])
    plt.ylim([embeddings[:, 1].min() - 1, embeddings[:, 1].max() + 1])
    plt.legend(loc='best')
    # plt.title(title)
    # plt.xlabel("Dimension 1")
    # plt.ylabel("Dimension 2")
    plt.savefig(path)
    plt.clf()

data_dir = "/shared/eng/bpt3/data/UniViT/data"
save_dir = "/shared/eng/bpt3/data/UniViT/save"
visualization_data = pickle.load(open(f"{data_dir}/trainingDataset.pkl", "rb"))
visualization_data = random.choices(visualization_data, k=10000)
# temporal_3d_data = random.choice([p for p in visualization_data if len(p) > 1 and p[0][1][0] > 1])
# temporal_2d_data = [(v[0], (1, v[1][1], v[1][2]), v[2], v[3], v[4]) for v in temporal_3d_data]
# static_3d_data = random.choice([p for p in visualization_data if len(p) == 1 and p[0][1][0] > 1])
# static_2d_data = [(v[0], (1, v[1][1], v[1][2]), v[2], v[3], v[4]) for v in static_3d_data]
# visualization_data += [temporal_3d_data, temporal_2d_data, static_3d_data, static_2d_data]
dataset_labels = [labelDatasetData(p) for p in visualization_data]
modality_labels = [labelModalityData(p) for p in visualization_data]
depth_labels = [labelDepthData(p) for p in visualization_data]
depth_precise_labels = [labelPreciseDepthData(p) for p in visualization_data]
time_labels = [labelTimeData(p) for p in visualization_data]
time_precise_labels = [labelPreciseTimeData(p) for p in visualization_data]
depthTime_labels = [labelDepthTimeData(p) for p in visualization_data]
# depthTimePlus_labels = [l for l in depthTime_labels]
# depthTimePlus_labels[-4] = "Temporal 3D Case Study"
# depthTimePlus_labels[-3] = "Temporal 2D Case Study"
# depthTimePlus_labels[-2] = "Static 3D Case Study"
# depthTimePlus_labels[-1] = "Static 2D Case Study"
all_labels = [labelAllData(p) for p in visualization_data]

model = UniViT(
    config.max_height,
    config.max_width,
    config.max_depth,
    config.max_time,
    config.num_channels,
    config.patch_size,
    config.depth_patch_size,
    config.time_patch_size,
    config.representation_size,
    config.num_layers,
    config.num_heads,
    config.projection_size,
    config.mlp_dim,
    config.dropout,
    config.attention_dropout,
    config.mask_prob,
    config.patch_prob,
    extra_cls=True,
).to(device)
print("Loading previous model")
model.load_state_dict(
    torch.load(f"{save_dir}/{model_key}.pt", map_location="cpu")["model"]
)
model.eval()
model.requires_grad_(False)

embeddings = []
visualization_data = ImageDataset(visualization_data, config, "cpu", augment=False)
visualization_loader = DataLoader(
    visualization_data,
    batch_size=config.downstream_batch_size,
    shuffle=False,
    num_workers=config.num_workers,
)
for batch_images, batch_dimensions in tqdm(
    visualization_loader, desc=f"Generating Embeddings", leave=False
):
    batch_images = batch_images.to(device)
    with torch.no_grad():
        representations = model.embed(batch_images, batch_dimensions)

    for i in range(len(representations)):
        embeddings.append(representations[i].cpu().clone().numpy())

embeddings = np.array(embeddings)
tsne = TSNE(n_components=2)
reduced_embeddings = tsne.fit_transform(embeddings)

plot_tsne(reduced_embeddings, dataset_labels, "Dataset", f"results/{model_key}_tsne_dataset.png")
plot_tsne(reduced_embeddings, modality_labels, "Modality", f"results/{model_key}_tsne_modality.png")
plot_tsne(reduced_embeddings, depth_labels, "Depth", f"results/{model_key}_tsne_depth.png")
plot_tsne(reduced_embeddings, time_labels, "Time", f"results/{model_key}_tsne_time.png")
plot_tsne(reduced_embeddings, depth_precise_labels, "Precise Depth", f"results/{model_key}_tsne_depth_precise.png")
plot_tsne(reduced_embeddings, time_precise_labels, "Precise Time", f"results/{model_key}_tsne_time_precise.png")
plot_tsne(reduced_embeddings, depthTime_labels, "Depth-Time", f"results/{model_key}_tsne_depthTime.png")
plot_tsne(reduced_embeddings, all_labels, "All", f"results/{model_key}_tsne_all.png")
# plot_tsne(reduced_embeddings, depthTimePlus_labels, "Depth-Time Plus", f"results/{model_key}_tsne_depthTimePlus.png")