import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from src.config import Config
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data.image_dataset import ImageDataset
from src.models.univit_hierarchical import UniViT

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

config = Config()
config.batch_size = 8
cuda_num = 5
device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)

batches_per_step = config.effective_batch_size // config.batch_size
data_dir = '/shared/bpt3/data/UniViT/data'
save_dir = '/shared/bpt3/data/UniViT/save'
train_data_all = pickle.load(open(f'{data_dir}/trainingDataset.pkl', 'rb'))

for modality in ['Chest X-Ray (MIMIC)', 'Chest X-Ray (CheXpert)', 'Skin Lesion', 'MRI', 'Amyloid PET', 'FDG PET', ]:
    save_name = modality.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '').lower()
    print(f'Training {modality}')
    train_data_modality = [p for p in train_data_all if p[0][3] == modality]
    train_data = ImageDataset(train_data_modality, config, device)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    model = UniViT(config.max_height, 
                config.max_width, 
                config.max_time, 
                config.max_slice, 
                config.num_channels, 
                config.patch_size, 
                config.representation_size, 
                config.num_layers, 
                config.num_secondary_layers,
                config.num_heads, 
                config.projection_size, 
                config.mlp_dim, 
                config.dropout, 
                config.attention_dropout,
                config.mask_prob)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    model = model.to(device)

    if os.path.exists(f"{save_dir}/univit_individual_{save_name}.pt"):
        print("Loading previous model")
        checkpoint = torch.load(f'{save_dir}/univit_individual_{save_name}.pt', map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        num_steps = checkpoint['steps']
    else:
        num_steps = 0
        
    # Setup teacher model
    teacher_model = deepcopy(model)
    teacher_model.eval()
    teacher_model = teacher_model.to(device)
            
    def update_teacher_model(teacher_model, student_model, alpha=0.999):
        with torch.no_grad():
            for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
                teacher_param.data.mul_(alpha).add_(student_param.data.to(device), alpha=1 - alpha)

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
    
    def update_centers(center_cls, center_patch, cls1, cls2, embd_seq1, embd_seq2):
        cls = torch.cat([cls1, cls2], dim=0).mean(dim=0)
        center_cls = config.center_momentum * center_cls + (1 - config.center_momentum) * cls
        patch = torch.cat([embd_seq1, embd_seq2], dim=0).mean(dim=(0,1))
        center_patch = config.center_momentum * center_patch + (1 - config.center_momentum) * patch
        return center_cls, center_patch

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
            batch_images1, batch_dimensions1 = train_data.augment_batch(batch_images, batch_dimensions)
            batch_images2, batch_dimensions2 = train_data.augment_batch(batch_images, batch_dimensions)
            
            cls_model1, embd_seq_model1, mask1, orig_mask1 = model(batch_images1.to(device), batch_dimensions1.to(device), seqMask=True, train=True)
            cls_model2, embd_seq_model2, mask2, orig_mask2 = model(batch_images2.to(device), batch_dimensions2.to(device), seqMask=True, train=True)
            with torch.no_grad():
                cls_teacher1, embd_seq_teacher1, _, _ = teacher_model(batch_images1.to(device), batch_dimensions1.to(device), seqMask=False, train=True)
                cls_teacher1 = cls_teacher1.to(device)
                embd_seq_teacher1 = embd_seq_teacher1.to(device)
                cls_teacher2, embd_seq_teacher2, _, _ = teacher_model(batch_images2.to(device), batch_dimensions2.to(device), seqMask=False, train=True)
                cls_teacher2 = cls_teacher2.to(device)
                embd_seq_teacher2 = embd_seq_teacher2.to(device)
                
            loss_cls = (cls_loss_fn(cls_model1, cls_teacher1, center_cls) + cls_loss_fn(cls_model2, cls_teacher2, center_cls)) / 2
            loss_mim = (mim_loss_fn(embd_seq_model1, embd_seq_teacher1, center_patch, mask1 & ~orig_mask1) + mim_loss_fn(embd_seq_model2, embd_seq_teacher2, center_patch, mask2 & ~orig_mask2)) / 2
            loss = loss_cls + loss_mim
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            center_cls, center_patch = update_centers(center_cls, center_patch, cls_teacher1, cls_teacher2, embd_seq_teacher1, embd_seq_teacher2)
            momentum_val = 1.0 + 0.5 * (config.momentum - 1.0) * (1 + np.cos(np.pi * num_steps / config.tot_steps))
            update_teacher_model(teacher_model, model, momentum_val)
            running_loss = running_loss[-999:] + [loss.detach().cpu().item()]
            pbar.set_description(f'Current Loss: {np.mean(running_loss):.4f}')
            pbar.update(1)
            num_steps += 1
            if num_steps % 1000 == 0:
                loss_plot.append(np.mean(running_loss))
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'steps': num_steps}, f'{save_dir}/univit{save_name}.pt')
            if num_steps >= config.tot_steps:
                break
    
    pbar.close()
    pickle.dump(loss_plot, open(f'{save_dir}/univit{save_name}_loss_plot.pkl', 'wb'))
