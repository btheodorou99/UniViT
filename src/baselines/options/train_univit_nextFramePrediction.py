import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from src.config import Config
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from src.models.univit_cls import UniViT
from src.data.image_dataset import ImageDataset

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

config = Config()
cuda_list = [0, 1, 2, 3]
device = torch.device(f"cuda:{cuda_list[0]}" if torch.cuda.is_available() and cuda_list else "cpu")
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)

batches_per_step = config.effective_batch_size // config.batch_size
data_dir = '/shared/bpt3/data/UniViT/data'
save_dir = '/shared/bpt3/data/UniViT/save'
train_data = pickle.load(open(f'{data_dir}/trainingDataset.pkl', 'rb'))
train_data = ImageDataset(train_data, config, device)
train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

model = UniViT(config.max_height, 
               config.max_width, 
               config.max_time, 
               config.max_slice, 
               config.num_channels, 
               config.patch_size, 
               config.representation_size, 
               config.num_layers, 
               config.num_heads, 
               config.hidden_dim, 
               config.mlp_dim, 
               config.dropout, 
               config.attention_dropout,
               config.mask_prob,
               clsExtraHead=True)
model = DataParallel(model, device_ids=cuda_list)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
if os.path.exists(f"{save_dir}/univit.pt"):
    print("Loading previous model")
    checkpoint = torch.load(f'{save_dir}/univit.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    num_steps = checkpoint['steps']
else:
    num_steps = 0
    
# Setup teacher model
teacher_model = deepcopy(model)
teacher_model.eval()
        
def update_teacher_model(teacher_model, student_model, alpha=0.999):
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
            teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)

def cls_loss_fn(student_emb, teacher_emb):
    student_norm = F.log_softmax(student_emb, dim=-1)
    teacher_norm = F.softmax(teacher_emb, dim=-1)
    loss = F.kl_div(student_norm, teacher_norm, reduction='batchmean')
    return loss

def mim_loss_fn(student_emb, teacher_emb, mask):
    student_norm = F.log_softmax(student_emb, dim=-1) * mask.unsqueeze(-1)
    teacher_norm = F.softmax(teacher_emb, dim=-1) * mask.unsqueeze(-1)
    loss = F.kl_div(student_norm.view(-1, config.representation_size), teacher_norm.view(-1, config.representation_size), reduction='batchmean')
    loss /= (mask.sum() / mask.numel()) # scale loss by mask
    return loss

def framePred_loss_fn(time_pred, slice_pred, time_seq, slice_seq, time_mask, slice_mask):
    time_pred = F.normalize(time_pred, dim=-1)
    slice_pred = F.normalize(slice_pred, dim=-1)
    time_seq = F.normalize(time_seq, dim=-1)
    slice_seq = F.normalize(slice_seq, dim=-1)
    time_dists = 1 - F.cosine_similarity(time_pred[:, :-1], time_seq[:, 1:], dim=-1)
    time_dists = time_dists * (~time_mask[:, 1:])
    time_loss = time_dists.sum() / (~time_mask[:, 1:]).sum()
    slice_dists = 1 - F.cosine_similarity(slice_pred[:, :-1], slice_seq[:, 1:], dim=-1)
    slice_dists = slice_dists * (~slice_mask[:, 1:])
    slice_loss = slice_dists.sum() / (~slice_mask[:, 1:]).sum()
    loss = time_loss + slice_loss
    return loss
    

# Train Model
model.train()
pbar = tqdm(total=config.tot_steps, leave=False, desc='Current Loss: N/A')
pbar.update(num_steps)

loss_plot = []
step_loss = 0
batches_since_step = 0
while num_steps < config.tot_steps:
    running_loss = []
    for batch_images, batch_dimensions in train_loader:
        batch_images_aug1, batch_dimensions1 = train_data.augment_batch(batch_images, batch_dimensions)
        batch_images_aug2, batch_dimensions2 = train_data.augment_batch(batch_images, batch_dimensions)
        
        cls_model = model.embed(batch_images_aug1, batch_dimensions1, train=True)
        embd_seq_model, mask, orig_mask, time_mask, slice_mask = model(batch_images, batch_dimensions, seqMask=True, train=True)
        time_seq_model = embd_seq_model[:, :config.max_time]
        slice_seq_model = embd_seq_model[:, config.max_time:config.max_time + config.max_slice]
        embd_seq_model = embd_seq_model[:, config.max_time + config.max_slice:]
        with torch.no_grad():
            cls_teacher = teacher_model.embed(batch_images_aug2, batch_dimensions2, train=True)
            embd_seq_teacher, _, _ = teacher_model(batch_images, batch_dimensions, train=True)
            embd_seq_teacher, _, _ = embd_seq_teacher[:, config.max_time + config.max_slice:]
            
        loss_cls = cls_loss_fn(cls_model, cls_teacher)
        loss_mim = mim_loss_fn(embd_seq_model, embd_seq_teacher, mask & ~orig_mask)
        loss_frameSimilarity = frameSim_loss_fn(time_seq_model, slice_seq_model, time_mask, slice_mask)
        loss = loss_cls + loss_mim + loss_frameSimilarity
        loss = loss / batches_per_step
        loss.backward()
        step_loss += loss.detach().cpu().item()
        batches_since_step += 1

        if batches_since_step == batches_per_step:
            batches_since_step = 0
            optimizer.step()
            optimizer.zero_grad()
            update_teacher_model(teacher_model, model)
            running_loss = running_loss[-999:] + [step_loss]
            step_loss = 0
            pbar.set_description(f'Current Loss: {np.mean(running_loss):.4f}')
            pbar.update(1)
            num_steps += 1
        if num_steps % 1000 == 0 and batches_since_step % batches_per_step == 0:
            loss_plot.append(np.mean(running_loss))
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'steps': num_steps}, f'{save_dir}/univit.pt')
        if num_steps >= config.tot_steps:
            break
  
pbar.close()
pickle.dump(loss_plot, open(f'{save_dir}/univit_loss_plot.pkl', 'wb'))