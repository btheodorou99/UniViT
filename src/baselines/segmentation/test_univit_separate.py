import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from src.config import Config
import medpy.metric.binary as metrics
from torch.utils.data import DataLoader
from src.models.univit_separate import UniViT
from src.models.downstream import SegmentationModel
from src.baselines.segmentation.data.image_dataset import ImageDataset

model_key = "univit_separate"

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

config = Config()
cuda_num = 0
device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

data_dir = "/shared/eng/bpt3/data/UniViT/data"
save_dir = "/shared/eng/bpt3/data/UniViT/save"
tune_data = pickle.load(open(f"{data_dir}/tuningDataset.pkl", "rb"))
tune_data = {
    task: [[p] for p in tune_data[task] if p[4] is not None and isinstance(p[4], str) and os.path.exists(p[4])] for task in tune_data
}
test_data = pickle.load(open(f"{data_dir}/testingDataset.pkl", "rb"))
test_data = {
    task: [[p] for p in test_data[task] if p[4] is not None and isinstance(p[4], str) and os.path.exists(p[4])] for task in test_data
}
task_map = pickle.load(open(f"{data_dir}/taskMap.pkl", "rb"))
valid_tasks = [t for t in tune_data if tune_data[t] and test_data[t] if t in task_map and task_map[t] == "Segmentation" and 'T1C' in t]
tune_data = {task: tune_data[task] for task in valid_tasks}
test_data = {task: test_data[task] for task in valid_tasks}

model = UniViT(
    config.max_height,
    config.max_width,
    config.max_time,
    config.max_depth,
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
    config.patch_prob,
    extra_cls=True,
).to(device)
print("Loading previous model")
model.load_state_dict(
    torch.load(f"{save_dir}/{model_key}.pt", map_location="cpu")["model"], strict=False
)
model.eval()
model.requires_grad_(False)

allResults = {}
for task in valid_tasks:
    print(f"\n\nDownstream Evaluation on {task}")
    task_tune = tune_data[task]
    task_tune_data = ImageDataset(task_tune, config, "cpu")
    task_tune_loader = DataLoader(
        task_tune_data,
        batch_size=config.downstream_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    task_test = test_data[task]
    task_test_data = ImageDataset(task_test, config, "cpu")
    task_test_loader = DataLoader(
        task_test_data,
        batch_size=config.downstream_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    
    bce_loss = torch.nn.BCELoss()
    def dice_loss(pred, target):
        pred2 = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
        target2 = target.contiguous().view(target.shape[0], target.shape[1], -1)
        num = torch.sum(torch.mul(pred2, target2), dim=-1)
        den = torch.sum(pred2, dim=-1) + torch.sum(target2, dim=-1)
        dice_score = (2 * num) / (den + 1)
        dice_loss = 1 - dice_score.mean()
        return dice_loss

    downstream = SegmentationModel(config.representation_size, config.patch_size, config.max_depth).to(device)
    optimizer = torch.optim.Adam(downstream.parameters(), lr=config.downstream_lr)
    for epoch in tqdm(
        range(10*config.downstream_epochs), leave=False, desc=f"{task} Tuning"
    ):
        for batch_images, batch_dimensions, batch_labels in tqdm(
            task_tune_loader, desc=f"{task} Tuning Epoch {epoch+1}", leave=False
        ):
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            with torch.no_grad():
                representations = model.embed_patches(batch_images, batch_dimensions)
            predictions = downstream(representations)
            loss = bce_loss(predictions, batch_labels) + dice_loss(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    dice_values = []
    hausdorff_values = []
    for batch_images, batch_dimensions, batch_labels in tqdm(
        task_test_loader, desc=f"{task} Testing", leave=False
    ):
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.numpy()
        with torch.no_grad():
            representations = model.embed_patches(batch_images, batch_dimensions)
            predictions = downstream(representations)
            predictions = (predictions > 0.5).cpu().numpy()
        for i in range(len(predictions)):
            if batch_labels[i].max() == 0:
                continue
            
            dice_values.append(metrics.dc(predictions[i], batch_labels[i]))
            hausdorff_values.append(metrics.hd(predictions[i], batch_labels[i]))

    dice_score = np.mean(dice_values)
    hausdorff_score = np.mean(hausdorff_values)
    taskResults = {"Dice Coefficient": dice_score, "95th Percentile Hausdorff Distance": hausdorff_score}
    print(taskResults)
    allResults[task] = taskResults
pickle.dump(allResults, open(f"{save_dir}/{model_key}_segmentationResults.pkl", "wb"))
