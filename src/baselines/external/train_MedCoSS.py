import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from src.config import Config
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from src.baselines.external.models.medcoss import MedCoSS
from src.baselines.external.data.buffer_dataset import ImageDataset, BufferDataset, KNNDataset

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

config = Config()
cuda_num = 2
device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)
  
NUM_CENTERS = 0.01
DATA_PER_CENTER = 5
DECODER_DIM = 512
NUM_DECODER_LAYERS = 8

config.batch_size = config.effective_batch_size
data_dir = '/shared/eng/bpt3/data/UniViT/data'
save_dir = '/shared/eng/bpt3/data/UniViT/save'
save_dir = '/srv/local/data/bpt3/UniViT/save'
train_data_all = pickle.load(open(f'{data_dir}/trainingDataset.pkl', 'rb'))
knn_data = {mod: random.choices(data, k=500) for (mod, data) in pickle.load(open(f'{data_dir}/tuningDataset.pkl', 'rb')).items()}
knn_train_data = [([p], mod) for mod in knn_data for p in knn_data[mod][:450]]
knn_test_data = [([p], mod) for mod in knn_data for p in knn_data[mod][450:500]]
mod_list = list(knn_data.keys())
knn_train_data = KNNDataset(knn_train_data, config, 'cpu', mod_list)
knn_test_data = KNNDataset(knn_test_data, config, 'cpu', mod_list)
knn_train_loader = DataLoader(knn_train_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
knn_test_loader = DataLoader(knn_test_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
modalities = ['Chest X-Ray (MIMIC)', 'Chest X-Ray (CheXpert)', 'Skin Lesion', 'MRI', 'Amyloid PET', 'FDG PET', 'CT', 'Chest X-Ray (COVID-QU-Ex)', 'Histopathology']
config.tot_steps = config.tot_steps // len(modalities)

model = MedCoSS(config.max_height, 
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
               config.mask_prob,
               DECODER_DIM,
               NUM_DECODER_LAYERS)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
model = model.to(device)

if os.path.exists(f"{save_dir}/medcoss.pt"):
    print("Loading previous model")
    checkpoint = torch.load(f'{save_dir}/medcoss.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    modality = checkpoint['modality']
    num_steps = checkpoint['steps']
else:
    modality = 'Chest X-Ray (MIMIC)'
    num_steps = 0

def validate(model, train, test):
    model.eval()
    images = []
    labels = []
    for batch_images, batch_dimensions, batch_labels in train:
        batch_cls = model.embed(batch_images.to(device), batch_dimensions.to(device))
        images.extend(batch_cls.cpu().detach().numpy())
        labels.extend(batch_labels.numpy())
    images = np.array(images)
    labels = np.array(labels)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(images, labels)
    images = []
    labels = []
    for batch_images, batch_dimensions, batch_labels in test:
        batch_cls = model.embed(batch_images.to(device), batch_dimensions.to(device))
        images.extend(batch_cls.cpu().detach().numpy())
        labels.extend(batch_labels.numpy())
    images = np.array(images)    
    labels = np.array(labels)
    preds = knn.predict(images)
    acc = (preds == labels).mean()
    model.train()
    return acc

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
modalities = modalities[modalities.index(modality):]
loss_plot = []
knn_plot = []
knn_acc = 0
buffer = []
for modality in modalities:
    train_data_modality = [p for p in train_data_all if p[0][3] == modality]
    train_data = BufferDataset([(p, 0) for p in train_data_modality] + [(p, 1) for p in buffer], config, 'cpu')
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    pbar = tqdm(total=config.tot_steps, leave=True, desc='Current Loss: N/A; Current KNN Acc: N/A')
    pbar.update(num_steps)
    
    teacher_model = deepcopy(model)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model = teacher_model.to(device)
    
    model.train()
    while num_steps < config.tot_steps:
        running_loss = []
        for batch_images, batch_dimensions, batch_buffer_mask in train_loader:
            batch_images, batch_dimensions = train_data.augment_batch(batch_images, batch_dimensions)
            batch_images_curr = batch_images[batch_buffer_mask == 0]
            batch_dimensions_curr = batch_dimensions[batch_buffer_mask == 0]
            batch_images_buffer = batch_images[batch_buffer_mask == 1]
            batch_dimensions_buffer = batch_dimensions[batch_buffer_mask == 1]
            
            if len(batch_images_curr) > 0:
                batch_images_curr = batch_images_curr.to(device)
                batch_dimensions_curr = batch_dimensions_curr.to(device)
                loss_curr = model(batch_images_curr, batch_dimensions_curr, seqMask=True, train=True)
            else:
                loss_curr = 0
                
            if len(batch_images_buffer) > 0:
                batch_images_buffer = batch_images_buffer.to(device)
                batch_dimensions_buffer = batch_dimensions_buffer.to(device)
                x_student, batch_seq_mask = model(batch_images_buffer, batch_dimensions_buffer, seqMask=True, train=True, buffer=True)
                with torch.no_grad():
                    x_teacher, _ = teacher_model(batch_images_buffer, batch_dimensions_buffer, seqMask=True, currMask=batch_seq_mask, train=True, buffer=True)
                loss_buffer = F.mse_loss(x_student, x_teacher)
            else:
                loss_buffer = 0
                
            loss = loss_curr + loss_buffer
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss = running_loss[-999:] + [loss.detach().cpu().item()]
            pbar.set_description(f'Current Loss: {np.mean(running_loss):.4f}; Current KNN Acc: {knn_acc:.4f}')
            pbar.update(1)
            num_steps += 1
            if num_steps % 1000 == 0:
                loss_plot.append(np.mean(running_loss))
                knn_acc = validate(model, knn_train_loader, knn_test_loader)
                knn_plot.append(knn_acc)
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'steps': num_steps}, f'{save_dir}/medcoss.pt')
            if num_steps >= config.tot_steps:
                break
         
    modality_data = ImageDataset(train_data_modality, config, 'cpu')
    modality_loader = DataLoader(modality_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    modality_idx = findModalityBuffer(model, modality_loader, int(NUM_CENTERS * len(train_data_modality)), DATA_PER_CENTER)
    modality_buffer = [train_data_modality[i] for i in modality_idx]
    buffer = buffer + modality_buffer
    num_steps = 0
  
pbar.close()
pickle.dump(loss_plot, open(f'{save_dir}/medcoss_loss_plot.pkl', 'wb'))
