import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from src.config import Config
from torchvision import transforms
from torch.utils.data import DataLoader
from src.models.downstream import SegmentationModel
from src.baselines.segmentation.metrics import get_LesionWiseResults
from src.baselines.segmentation.data.image_dataset_pretrained import ImageDataset
from src.baselines.external.dinov2.dinov2.models.vision_transformer import vit_base

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
MAX_PENALTY = 1000
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
    task: [
        p
        for p in tune_data[task]
        if p[4] is not None and isinstance(p[4], str) and os.path.exists(p[4])
    ]
    for task in tune_data
}
test_data = pickle.load(open(f"{data_dir}/testingDataset.pkl", "rb"))
test_data = {
    task: [
        p
        for p in test_data[task]
        if p[4] is not None and isinstance(p[4], str) and os.path.exists(p[4])
    ]
    for task in test_data
}
task_map = pickle.load(open(f"{data_dir}/taskMap.pkl", "rb"))
valid_tasks = [
    t
    for t in tune_data
    if tune_data[t] and test_data[t]
    if t in task_map and task_map[t] == "Segmentation"
]
tune_data = {task: tune_data[task] for task in valid_tasks}
test_data = {task: test_data[task] for task in valid_tasks}


def print_both(text, filename="seg_dino.log"):
    print(text)
    with open(filename, "a") as f:
        print(text, file=f)


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
state_dict = torch.load(f"{save_dir}/{model_key}.pt", map_location="cpu")["model"]
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith("student.backbone."):
        new_key = key[
            len("student.backbone.") :
        ]  # Remove the 'module.backbone.' prefix
        # blocks.x.y.norm1.weight to blocks.y.norm1.weight
        if new_key.startswith("blocks.") and new_key[7].isdigit():
            new_key = new_key[:7] + new_key[new_key.index(".", 7) + 1 :]
        new_state_dict[new_key] = value
    elif key.startswith("teacher.backbone."):
        continue  # Teacher is not needed for downstream evaluation
    elif "_head." in key or "_loss." in key:
        continue  # Head and loss not needed for downstream evaluation
    else:
        new_state_dict[key] = value  # Otherwise, keep the key as it is
model.load_state_dict(new_state_dict)
model.eval()
model.requires_grad_(False)
model.to(device)

transform = transforms.Compose(
    [
        transforms.Resize((config.max_height, config.max_width), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

bce_loss = torch.nn.BCELoss()


def dice_loss(pred, target, smooth=1e-5):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice


def bootstrap_mean_error(data, num_samples=1000, metric=np.mean):
    # Generate bootstrap samples
    bootstrap_samples = np.random.choice(data, (num_samples, len(data)), replace=True)

    # Calculate the metric for each bootstrap sample
    bootstrap_metrics = np.array([metric(sample) for sample in bootstrap_samples])

    # Calculate the mean and standard deviation of the bootstrap metrics
    mean_estimate = np.mean(bootstrap_metrics)
    error = np.std(bootstrap_metrics)  # This represents the "plus-minus" error

    return mean_estimate, error


allResults = {}
for task in valid_tasks:
    print_both(f"Downstream Evaluation on {task}")
    task_tune = tune_data[task]
    task_test = test_data[task]
    task_data = task_tune + task_test

    global_data = ImageDataset(
        task_data,
        config,
        "cpu",
        transform=transform,
        image_depth=config.segmentation_depth,
    )
    global_loader = DataLoader(
        global_data,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    valid_img = [i for i, (_, img) in enumerate(global_loader) if img.max() > 0]
    task_data = [task_data[i] for i in valid_img]
    del global_data, global_loader, valid_img

    totData = len(task_data)
    testSize = totData // 3
    task_tune = task_data[:-testSize]
    task_test = task_data[-testSize:]

    task_tune_data = ImageDataset(
        task_tune,
        config,
        "cpu",
        transform=transform,
        image_depth=config.segmentation_depth,
    )
    task_tune_loader = DataLoader(
        task_tune_data,
        batch_size=config.segmentation_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    task_test_data = ImageDataset(
        task_test,
        config,
        "cpu",
        transform=transform,
        image_depth=config.segmentation_depth,
    )
    task_test_loader = DataLoader(
        task_test_data,
        batch_size=config.segmentation_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    downstream = SegmentationModel(
        config.representation_size,
        config.patch_size,
        config.segmentation_depth,
    ).to(device)
    optimizer = torch.optim.Adam(downstream.parameters(), lr=config.downstream_lr)
    for epoch in tqdm(
        range(config.downstream_epochs), leave=False, desc=f"{task} Tuning"
    ):
        for batch_images, batch_labels in tqdm(
            task_tune_loader, desc=f"{task} Tuning Epoch {epoch+1}", leave=False
        ):
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            # Flatten everything for 2D embedding before 3D segmentation
            bs, depth, channels, height, width = batch_images.shape
            batch_images = batch_images.view(-1, channels, height, width)
            with torch.no_grad():
                representations = model.forward_features(batch_images)[
                    "x_norm_patchtokens"
                ]
                representations = representations.reshape(
                    bs, depth, -1, config.representation_size
                )
            predictions = downstream(representations)
            loss = bce_loss(predictions, batch_labels) + dice_loss(
                predictions, batch_labels
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    results = {}
    for batch_images, batch_labels in tqdm(
        task_test_loader, desc=f"{task} Testing", leave=False
    ):
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.numpy()

        # Flatten everything for 2D embedding before 3D segmentation
        bs, depth, channels, height, width = batch_images.shape
        batch_images = batch_images.view(-1, channels, height, width)
        with torch.no_grad():
            representations = model.forward_features(batch_images)["x_norm_patchtokens"]
            representations = representations.reshape(
                bs, depth, -1, config.representation_size
            )
            predictions = downstream(representations)
            predictions = (predictions > 0.5).cpu().numpy()

        for i in range(len(predictions)):
            indiv_results = get_LesionWiseResults(predictions[i], batch_labels[i])
            if indiv_results is not None:
                for key in indiv_results:
                    if key not in results:
                        results[key] = []
                    results[key].append(indiv_results[key])

    taskResults = {}
    for key in results:
        results[key] = np.array(results[key])
        keyVals = bootstrap_mean_error(results[key])
        taskResults[key] = keyVals[0]
        taskResults[f"{key} PM"] = keyVals[1]
    print_both(taskResults)
    allResults[task] = taskResults
pickle.dump(allResults, open(f"{save_dir}/{model_key}_segmentationResults.pkl", "wb"))
