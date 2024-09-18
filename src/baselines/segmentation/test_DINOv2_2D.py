import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from src.config import Config
from torchvision import transforms
import medpy.metric.binary as metrics
from torch.utils.data import DataLoader
from src.models.downstream import SegmentationModel
from src.baselines.segmentation.data.image_dataset_pretrained import ImageDataset
from src.baselines.external.dinov2.dinov2.models.vision_transformer import vit_base

model_key = "dinov2"

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
    task: [p for p in tune_data[task] if p[4] is not None and isinstance(p[4], str) and os.path.exists(p[4])] for task in tune_data
}
test_data = pickle.load(open(f"{data_dir}/testingDataset.pkl", "rb"))
test_data = {
    task: [p for p in test_data[task] if p[4] is not None and isinstance(p[4], str) and os.path.exists(p[4])] for task in test_data
}
task_map = pickle.load(open(f"{data_dir}/taskMap.pkl", "rb"))
valid_tasks = [t for t in tune_data if tune_data[t] and test_data[t] if t in task_map and task_map[t] == "Segmentation"]
tune_data = {task: tune_data[task] for task in valid_tasks}
test_data = {task: test_data[task] for task in valid_tasks}

model = vit_base(
    img_size=config.max_height,
    patch_size=config.patch_size, 
    init_values=1e-5,
    ffn_layer="swiglu",
    block_chunks=0,
    qkv_bias=True,
    proj_bias=True,
    ffn_bias=True,
    num_register_tokens=0,
    interpolate_offset=0.1,
    interpolate_antialias=False,
    drop_path_rate=0.3,
    drop_path_uniform=True,
)
state_dict = torch.load(f"{save_dir}/{model_key}.pt", map_location='cpu')['model']
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('student.backbone.'):
        new_key = key[len('student.backbone.'):]  # Remove the 'module.backbone.' prefix
        # blocks.x.y.norm1.weight to blocks.y.norm1.weight
        if new_key.startswith('blocks.') and new_key[7].isdigit():
            new_key = new_key[:7] + new_key[new_key.index('.', 7)+1:]
        new_state_dict[new_key] = value
    elif key.startswith('teacher.backbone.'):
        continue # Teacher is not needed for downstream evaluation
    elif '_head.' in key or '_loss.' in key:
        continue # Head and loss not needed for downstream evaluation
    else:
        new_state_dict[key] = value  # Otherwise, keep the key as it is
model.load_state_dict(new_state_dict)
model.eval()
model.requires_grad_(False)
model.to(device)

transform = transforms.Compose([
    transforms.Resize((config.max_height, config.max_width), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

allResults = {}
for task in valid_tasks:
    print(f"Downstream Evaluation on {task}")
    task_tune = tune_data[task]
    task_tune_data = ImageDataset(
        task_tune,
        config,
        "cpu",
        transform=transform,
        image_depth=config.max_depth,
    )
    task_tune_loader = DataLoader(
        task_tune_data,
        batch_size=config.downstream_batch_size,
        shuffle=True,
        num_workers=int(0.5*config.num_workers),
    )
    task_test = test_data[task]
    task_test_data = ImageDataset(
        task_test,
        config,
        "cpu",
        transform=transform,
        image_depth=config.max_depth,
    )
    task_test_loader = DataLoader(
        task_test_data,
        batch_size=config.downstream_batch_size,
        shuffle=False,
        num_workers=int(0.5*config.num_workers),
    )
    
    bce_loss = torch.nn.BCELoss()
    def dice_loss(pred, target):
        pred2 = pred.contiguous().view(pred.shape[0], -1)
        target2 = target.contiguous().view(target.shape[0], -1)
        num = torch.sum(torch.mul(pred2, target2), dim=-1)
        den = torch.sum(pred2, dim=-1) + torch.sum(target2, dim=-1)
        dice_score = (2 * num) / (den + 1)
        dice_loss = 1 - dice_score.mean()
        return dice_loss

    downstream = SegmentationModel(config.representation_size, 14).to(device)
    optimizer = torch.optim.Adam(downstream.parameters(), lr=config.downstream_lr)
    for epoch in tqdm(
        range(config.downstream_epochs), leave=False, desc=f"{task} Tuning"
    ):
        for batch_images, batch_labels in tqdm(
            task_tune_loader, desc=f"{task} Tuning Epoch {epoch+1}", leave=False
        ):
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            
            # Flatten everything for 2D segmentation
            batch_images = batch_images.permute(0,2,1,3,4).repeat(1,1,3,1,1)
            batch_images = batch_images.view(-1, 3, config.max_height, config.max_height)
            batch_labels = batch_labels.view(-1, config.max_height, config.max_height)
            with torch.no_grad():
                representations = model.forward_features(batch_images)['x_norm_patchtokens']
            predictions = downstream(representations)
            loss = bce_loss(predictions, batch_labels) + dice_loss(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    dice_values = []
    hausdorff_values = []
    for batch_images, batch_labels in tqdm(
        task_test_loader, desc=f"{task} Testing", leave=False
    ):
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.numpy()
        
        # Flatten everything for 2D segmentation
        bs, channels, depth, height, width = batch_images.shape
        batch_images = batch_images.permute(0,2,1,3,4).repeat(1,1,3,1,1)
        batch_images = batch_images.view(-1, 3, config.max_height, config.max_height)
        with torch.no_grad():
            representations = model.forward_features(batch_images)['x_norm_patchtokens']
            predictions = downstream(representations)
            predictions = (predictions > 0.5).cpu().numpy()
            
        # Reshape back to 3D for evaluation
        predictions = predictions.reshape(bs, depth, height, width)
        for i in range(len(predictions)):
            if batch_labels[i].max() == 0:
                continue
            
            dice_values.append(metrics.dc(predictions[i], batch_labels[i]))
            hausdorff_values.append(metrics.hd(predictions[i], batch_labels[i]))

    dice_score = np.mean(dice_values)
    hausdorff_score = np.mean(hausdorff_values)
    taskResults = {"Dice Coefficient": dice_score, "95th Percentile Hausdorff Distance": hausdorff_score}
    print('\t', taskResults)
    allResults[task] = taskResults
pickle.dump(allResults, open(f"{save_dir}/{model_key}_2D_segmentationResults.pkl", "wb"))
