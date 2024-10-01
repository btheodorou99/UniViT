import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from src.config import Config
import torch.nn.functional as F
from src.models.univit import UniViT
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from src.data.image_dataset import ImageDataset, KNNDataset

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


# SEED = 4
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)

config = Config()
cuda_num = 3
device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(SEED)

data_dir = "/shared/eng/bpt3/data/UniViT/data"
save_dir = "/shared/eng/bpt3/data/UniViT/save"
train_data = pickle.load(open(f"{data_dir}/trainingDataset.pkl", "rb"))
train_modalities = set([p[0][3] for p in train_data])
train_data = ImageDataset(train_data, config, "cpu", augment=True)
config.dataset_to_steps(len(train_data))
train_loader = DataLoader(
    train_data,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
)
knn_data = {
    mod: random.choices(data, k=250)
    for (mod, data) in pickle.load(open(f"{data_dir}/tuningDataset.pkl", "rb")).items()
    if mod in train_modalities
}
knn_train_data = [([p], mod) for mod in knn_data for p in knn_data[mod][:200]]
knn_test_data = [([p], mod) for mod in knn_data for p in knn_data[mod][200:250]]
mod_list = list(knn_data.keys())
knn_train_data = KNNDataset(knn_train_data, config, "cpu", mod_list)
knn_test_data = KNNDataset(knn_test_data, config, "cpu", mod_list)
knn_train_loader = DataLoader(
    knn_train_data,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=int(0.5*config.num_workers),
)
knn_test_loader = DataLoader(
    knn_test_data,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=int(0.5*config.num_workers),
)

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
)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
model = model.to(device)

if os.path.exists(f"{save_dir}/univit.pt"):
    print("Loading previous model")
    checkpoint = torch.load(f"{save_dir}/univit.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    num_steps = checkpoint["steps"]
else:
    num_steps = 0

# Setup teacher model
teacher_model = deepcopy(model)
teacher_model.eval()
teacher_model = teacher_model.to(device)


def update_teacher_model(teacher_model, student_model, alpha=0.999):
    with torch.no_grad():
        for teacher_param, student_param in zip(
            teacher_model.parameters(), student_model.parameters()
        ):
            teacher_param.data.mul_(alpha).add_(
                student_param.data.to(device), alpha=1 - alpha
            )


def cls_loss_fn(student_cls, teacher_cls, center):
    student_cls = F.log_softmax(student_cls / config.student_temp, dim=-1)
    teacher_cls = F.softmax((teacher_cls - center) / config.teacher_cls_temp, dim=-1)
    loss = torch.sum(-teacher_cls * student_cls, dim=-1).mean()
    return loss


def mim_loss_fn(student_emb, teacher_emb, center, mask):
    student_emb = F.log_softmax(student_emb / config.student_temp, dim=-1)
    teacher_emb = F.softmax((teacher_emb - center) / config.teacher_patch_temp, dim=-1)
    loss = torch.sum(-teacher_emb * student_emb, dim=-1)
    loss = torch.sum(loss * mask, dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
    loss = loss.mean()
    return loss


def frameSim_loss_fn(
    student_time_seq, student_depth_seq, teacher_time_seq, teacher_depth_seq, time_center, depth_center, time_mask, depth_mask,
):
    student_time_seq = F.log_softmax(student_time_seq / config.student_temp, dim=-1)
    teacher_time_seq = F.softmax((teacher_time_seq - time_center) / config.teacher_cls_temp, dim=-1)
    student_depth_seq = F.log_softmax(student_depth_seq / config.student_temp, dim=-1)
    teacher_depth_seq = F.softmax((teacher_depth_seq - depth_center) / config.teacher_cls_temp, dim=-1)
    time_loss1 = torch.sum(-teacher_time_seq[:, :-1] * student_time_seq[:, 1:], dim=-1)
    time_loss2 = torch.sum(-teacher_time_seq[:, 1:] * student_time_seq[:, :-1], dim=-1)
    time_loss = torch.sum((time_loss1 + time_loss2) * time_mask[:, 1:], dim=-1) / (2 * time_mask[:, 1:].sum(dim=-1).clamp(min=1.0))
    time_loss = time_loss.mean()
    if time_loss.isnan():
        time_loss = torch.tensor(0.0).to(device)
    depth_loss1 = torch.sum(-teacher_depth_seq[:, :-1] * student_depth_seq[:, 1:], dim=-1)
    depth_loss2 = torch.sum(-teacher_depth_seq[:, 1:] * student_depth_seq[:, :-1], dim=-1)
    depth_loss = torch.sum((depth_loss1 + depth_loss2) * depth_mask[:, 1:], dim=-1) / (2 * depth_mask[:, 1:].sum(dim=-1).clamp(min=1.0))
    depth_loss = depth_loss.mean()
    if depth_loss.isnan():
        depth_loss = torch.tensor(0.0).to(device)
    loss = (time_loss + depth_loss) / 2
    return loss


def update_centers(center_cls, center_patch, center_time, center_depth, cls1, cls2, embd_seq1, embd_seq2, time_seq1, time_seq2, depth_seq1, depth_seq2):
    cls = torch.cat([cls1, cls2], dim=0).mean(dim=0)
    center_cls = (
        config.center_momentum * center_cls + (1 - config.center_momentum) * cls
    )
    patch = torch.cat([embd_seq1, embd_seq2], dim=0).mean(dim=(0, 1))
    center_patch = (
        config.center_momentum * center_patch + (1 - config.center_momentum) * patch
    )
    time = torch.cat([time_seq1, time_seq2], dim=0).mean(dim=0)
    center_time = (
        config.center_momentum * center_time + (1 - config.center_momentum) * time
    )
    depth = torch.cat([depth_seq1, depth_seq2], dim=0).mean(dim=0)
    center_depth = (
        config.center_momentum * center_depth + (1 - config.center_momentum) * depth
    )
    return center_cls, center_patch, center_time, center_depth


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


# Train Model
model.train()
pbar = tqdm(
    total=config.tot_steps, leave=True, desc="Current Loss: N/A; Current KNN Acc: N/A"
)
pbar.update(num_steps)

loss_plot = []
knn_plot = []
knn_acc = 0
center_cls = torch.zeros(1, config.projection_size).to(device)
center_patch = torch.zeros(1, 1, config.projection_size).to(device)
center_time = torch.zeros(1, 1, config.projection_size).to(device)
center_depth = torch.zeros(1, 1, config.projection_size).to(device)
while num_steps < config.tot_steps:
    running_loss = []
    for (batch_images1, batch_dimensions1), (batch_images2, batch_dimensions2) in train_loader:
        batch_images1 = batch_images1.to(device)
        batch_dimensions1 = batch_dimensions1.to(device)
        batch_images2 = batch_images2.to(device)
        batch_dimensions2 = batch_dimensions2.to(device)

        cls_model1, embd_seq_model1, time_seq_model1, depth_seq_model1, mask1, orig_mask1, time_mask1, depth_mask1 = model(
            batch_images1,
            batch_dimensions1,
            seqMask=True,
            train=True,
        )
        cls_model2, embd_seq_model2, time_seq_model2, depth_seq_model2, mask2, orig_mask2, time_mask2, depth_mask2 = model(
            batch_images2,
            batch_dimensions2,
            seqMask=True,
            train=True,
        )
        with torch.no_grad():
            cls_teacher1, embd_seq_teacher1, time_seq_teacher1, depth_seq_teacher1, _, _, _, _ = teacher_model(
                batch_images1,
                batch_dimensions1,
                seqMask=False,
                train=True,
            )
            cls_teacher2, embd_seq_teacher2, time_seq_teacher2, depth_seq_teacher2, _, _, _, _ = teacher_model(
                batch_images2,
                batch_dimensions2,
                seqMask=False,
                train=True,
            )

        loss_cls = (
            cls_loss_fn(cls_model1, cls_teacher2, center_cls)
            + cls_loss_fn(cls_model2, cls_teacher1, center_cls)
        ) / 2
        loss_mim = (
            mim_loss_fn(
                embd_seq_model1, embd_seq_teacher1, center_patch, mask1 & ~orig_mask1
            )
            + mim_loss_fn(
                embd_seq_model2, embd_seq_teacher2, center_patch, mask2 & ~orig_mask2
            )
        ) / 2
        loss_frameSimilarity = (
            frameSim_loss_fn(
                time_seq_model1,
                depth_seq_model1,
                time_seq_teacher1,
                depth_seq_teacher1,
                center_time,
                center_depth,
                time_mask1,
                depth_mask1,
            )
            + frameSim_loss_fn(
                time_seq_model2,
                depth_seq_model2,
                time_seq_teacher2,
                depth_seq_teacher2,
                center_time,
                center_depth,
                time_mask2,
                depth_mask2,
            )
        ) / 2
        loss = loss_cls + loss_mim + loss_frameSimilarity
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()
        optimizer.zero_grad()
        center_cls, center_patch, center_time, center_depth = update_centers(
            center_cls,
            center_patch,
            center_time,
            center_depth,
            cls_teacher1,
            cls_teacher2,
            embd_seq_teacher1,
            embd_seq_teacher2,
            time_seq_teacher1,
            time_seq_teacher2,
            depth_seq_teacher1,
            depth_seq_teacher2,
        )
        momentum_val = 1.0 + 0.5 * (config.momentum - 1.0) * (
            1 + np.cos(np.pi * num_steps / 2*config.tot_steps)
        )
        update_teacher_model(teacher_model, model, momentum_val)
        running_loss = running_loss[-999:] + [loss.detach().cpu().item()]
        pbar.set_description(
            f"Current Loss: {np.mean(running_loss):.4f}; Current KNN Acc: {knn_acc:.4f}"
        )
        pbar.update(1)
        num_steps += 1
        if num_steps % 5000 == 0:
            loss_plot.append(np.mean(running_loss))
            knn_acc = validate(model, knn_train_loader, knn_test_loader)
            knn_plot.append(knn_acc)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "steps": num_steps,
                },
                f"{save_dir}/univit.pt",
            )
        if num_steps >= config.tot_steps:
            break

torch.save(
    {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "steps": num_steps,
    },
    f"{save_dir}/univit.pt",
)

pbar.close()
pickle.dump(loss_plot, open(f"{save_dir}/univit_loss_plot.pkl", "wb"))
pickle.dump(knn_plot, open(f"{save_dir}/univit_knn_plot.pkl", "wb"))
