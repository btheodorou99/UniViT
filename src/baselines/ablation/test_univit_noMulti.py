import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from src.config import Config
from torch.utils.data import DataLoader
from src.models.univit_noMulti import UniViT
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from src.data.image_dataset_noMulti import ImageDataset

model_key = "univit_noMulti"

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
save_dir = "/srv/local/data/bpt3/UniViT/save"
tune_data = pickle.load(open(f"{data_dir}/tuningDataset.pkl", "rb"))
tune_data = {
    task: [[p] for p in tune_data[task] if p[4] is not None] for task in tune_data
}
test_data = pickle.load(open(f"{data_dir}/testingDataset.pkl", "rb"))
test_data = {
    task: [[p] for p in test_data[task] if p[4] is not None] for task in test_data
}
task_map = pickle.load(open(f"{data_dir}/taskMap.pkl", "rb"))

model = UniViT(
    config.max_height,
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
for task in tune_data:
    print(f"\n\nDownstream Evaluation on {task}")
    task_tune = tune_data[task]
    label = task_tune[0][0][4]
    if isinstance(label, list):
        label_size = len(label)
        multiclass = False
    elif isinstance(label, int):
        label_size = len(set([p[0][4] for p in task_tune]))
        multiclass = True
    else:
        continue
    task_tune_data = ImageDataset(
        task_tune, config, "cpu", augment=False, downstream=True, multiclass=multiclass
    )
    task_tune_loader = DataLoader(
        task_tune_data,
        batch_size=config.downstream_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    task_test = test_data[task]
    task_test_data = ImageDataset(
        task_test, config, "cpu", augment=False, downstream=True, multiclass=multiclass
    )
    task_test_loader = DataLoader(
        task_test_data,
        batch_size=config.downstream_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    taskType = task_map[task]
    downstream = LogisticRegression(max_iter=10000)
    if taskType == "Multi-Label Classification":
        downstream = MultiOutputClassifier(downstream)
    X = []
    y = []

    downstream = LinearClassifier(config.representation_size, label_size).to(device)
    optimizer = torch.optim.SGD(
        downstream.parameters(), lr=config.downstream_lr, momentum=0.9, weight_decay=0
    )
    for epoch in tqdm(
        range(config.downstream_epochs), leave=False, desc=f"{task} Tuning"
    ):
        batches_since_step = 0
        for batch_images, batch_dimensions, batch_labels in tqdm(
            task_tune_loader, desc=f"{task} Tuning Epoch {epoch+1}", leave=False
        ):
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            with torch.no_grad():
                representations = model.embed(batch_images, batch_dimensions)
            predictions = downstream(representations)
            predictions = train_activation(predictions)
            loss = loss_fn(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    task_preds = []
    task_labels = []
    for batch_images, batch_dimensions, batch_labels in tqdm(
        task_test_loader, desc=f"{task} Testing", leave=False
    ):
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        with torch.no_grad():
            representations = model.embed(batch_images, batch_dimensions)
            predictions = downstream(representations)
            predictions = test_activation(predictions)
            task_preds.extend(predictions.cpu().tolist())
            task_labels.extend(batch_labels.cpu().tolist())

    if taskType == "Multi-Label Classification":
        task_preds = np.array(task_preds)
        task_labels = np.array(task_labels)
        task_rounded_preds = np.round(task_preds)
        accPerLabel = [
            metrics.accuracy_score(
                [label[i] for label in task_labels],
                [pred[i] for pred in task_rounded_preds],
            )
            for i in range(label_size)
        ]
        f1PerLabel = [
            metrics.f1_score(
                [label[i] for label in task_labels],
                [pred[i] for pred in task_rounded_preds],
            )
            for i in range(label_size)
        ]
        aurocPerLabel = [
            metrics.roc_auc_score(
                [label[i] for label in task_labels], [pred[i] for pred in task_preds]
            )
            for i in range(label_size)
        ]
        overallAcc = metrics.accuracy_score(
            task_labels.flatten(), task_rounded_preds.flatten()
        )
        overallF1 = metrics.f1_score(
            task_labels.flatten(), task_rounded_preds.flatten()
        )
        overallAUROC = metrics.roc_auc_score(
            task_labels.flatten(), task_preds.flatten()
        )
        taskResults = {
            "Accuracy": overallAcc,
            "F1": overallF1,
            "AUROC": overallAUROC,
            "Accuracy Per Label": accPerLabel,
            "F1 Per Label": f1PerLabel,
            "AUROC Per Label": aurocPerLabel,
        }
        print(taskResults)
    elif taskType == "Multi-Class Classification":
        task_probs = np.array(task_preds)
        task_labels = np.array(task_labels)
        task_preds = np.argmax(task_probs, axis=1)
        acc = metrics.accuracy_score(task_labels, task_preds)
        f1 = metrics.f1_score(task_labels, task_preds, average="macro")
        auroc = metrics.roc_auc_score(task_labels, task_probs, average="macro", multi_class="ovr")
        taskResults = {"Accuracy": acc, "F1": f1, "AUROC": auroc}
        print(taskResults)

    allResults[task] = taskResults
pickle.dump(allResults, open(f"{save_dir}/{model_key}_downstreamResults.pkl", "wb"))
