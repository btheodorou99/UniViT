import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from src.config import Config
from collections import Counter
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from src.baselines.external.medcoss.model import MedCoSS
from src.baselines.external.medcoss.buffer_dataset import (
    ImageDataset,
    BufferDataset,
)

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

config = Config()
cuda_num = 3
device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

NUM_CENTERS = 0.01
DATA_PER_CENTER = 5
DECODER_DIM = 512
NUM_DECODER_HEADS = 16
NUM_DECODER_LAYERS = 8

data_dir = "/shared/eng/bpt3/data/UniViT/data"
save_dir = "/shared/eng/bpt3/data/UniViT/save"
train_data_all = pickle.load(open(f"{data_dir}/trainingDataset.pkl", "rb"))
modalities = list(reversed([mod for mod, _ in Counter([p[0][3] for p in train_data_all]).most_common()]))

model = MedCoSS(
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
    DECODER_DIM,
    NUM_DECODER_HEADS,
    NUM_DECODER_LAYERS,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
model = model.to(device)

if os.path.exists(f"{save_dir}/medcoss.pt"):
    print("Loading previous model")
    checkpoint = torch.load(f"{save_dir}/medcoss.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    modality = checkpoint["modality"]
    modality = modalities[modalities.index(modality) :][1]
else:
    modality = modalities[0]

def findModalityBuffer(model, loader, num_centers, data_per_center):
    model.eval()
    embds = []
    for batch_images, batch_dimensions in loader:
        batch_cls = model.embed(batch_images.to(device), batch_dimensions.to(device))
        embds.extend(batch_cls.cpu().detach().numpy())
    embds = np.array(embds)
    kmeans = KMeans(n_clusters=num_centers).fit(embds)
    centers = kmeans.cluster_centers_
    idx = []
    for center in centers:
        dist = np.linalg.norm(embds - center, axis=1)
        idx.extend(np.argsort(dist)[:data_per_center])
    return idx


# Train Model
modalities = modalities[modalities.index(modality) :]
loss_plot = []
buffer = []
for m_num, modality in enumerate(modalities):
    num_steps = 0
    train_data_modality = [p for p in train_data_all if p[0][3] == modality]
    train_data = BufferDataset(
        [(p, 0) for p in train_data_modality] + [(p, 1) for p in buffer], 
        config, "cpu", augment=True
    )
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    tot_steps = int(config.tot_epochs * len(train_data_modality) / config.batch_size)
    pbar = tqdm(
        total=tot_steps,
        leave=False,
        desc=f"{modality} Current Loss: N/A",
    )
    pbar.update(num_steps)

    teacher_model = deepcopy(model)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model = teacher_model.to(device)

    model.train()
    while num_steps < tot_steps:
        running_loss = []
        for batch_images, batch_dimensions, batch_buffer_mask in train_loader:
            batch_images_curr = batch_images[batch_buffer_mask == 0]
            batch_dimensions_curr = batch_dimensions[batch_buffer_mask == 0]
            batch_images_buffer = batch_images[batch_buffer_mask == 1]
            batch_dimensions_buffer = batch_dimensions[batch_buffer_mask == 1]

            if len(batch_images_curr) > 0:
                batch_images_curr = batch_images_curr.to(device)
                batch_dimensions_curr = batch_dimensions_curr.to(device)
                loss_curr = model(
                    batch_images_curr, batch_dimensions_curr, seqMask=True, train=True
                )
                curr_scale = len(batch_images_curr) / len(batch_images)
            else:
                loss_curr = 0
                curr_scale = 1.0

            if len(batch_images_buffer) > 0:
                batch_images_buffer = batch_images_buffer.to(device)
                batch_dimensions_buffer = batch_dimensions_buffer.to(device)
                x_student, batch_seq_mask = model(
                    batch_images_buffer,
                    batch_dimensions_buffer,
                    seqMask=True,
                    train=True,
                    buffer=True,
                )
                with torch.no_grad():
                    x_teacher, _ = teacher_model(
                        batch_images_buffer,
                        batch_dimensions_buffer,
                        seqMask=True,
                        currMask=batch_seq_mask,
                        train=True,
                        buffer=True,
                    )
                loss_buffer = F.mse_loss(x_student, x_teacher)
                buffer_scale = len(batch_images_buffer) / len(batch_images)
            else:
                loss_buffer = 0
                buffer_scale = 1.0

            loss = ((curr_scale) * loss_curr) + ((buffer_scale) * loss_buffer)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss = running_loss[-999:] + [loss.detach().cpu().item()]
            pbar.set_description(
                f"Current Loss: {np.mean(running_loss):.4f}"
            )
            pbar.update(1)
            num_steps += 1
            if num_steps >= config.tot_steps:
                break

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "modality": modality,
        },
        f"{save_dir}/medcoss.pt",
    )
    
    pbar.close()
    
    modality_data = ImageDataset(train_data_modality, config, "cpu")
    modality_loader = DataLoader(
        modality_data,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    modality_idx = findModalityBuffer(
        model,
        modality_loader,
        int(NUM_CENTERS * len(train_data_modality)),
        DATA_PER_CENTER,
    )
    modality_buffer = [train_data_modality[i] for i in modality_idx]
    buffer = buffer + modality_buffer

pbar.close()
pickle.dump(loss_plot, open(f"{save_dir}/medcoss_loss_plot.pkl", "wb"))
