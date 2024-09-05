import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from src.config import Config
from torch.utils.data import DataLoader
from src.models.direct import Embedding2D
from src.baselines.temporal.data.image_dataset_direct2D import ImageDataset

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

config = Config()
cuda_num = 1
device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

modalities = [
    "Chest X-Ray (MIMIC)",
]

data_dir = "/shared/eng/bpt3/data/UniViT/data"
tune_data = pickle.load(open(f"{data_dir}/tuningTemporalDataset.pkl", "rb"))
tune_data = {
    task: [p for p in tune_data[task] if p[-1][4] is not None]  # + train_data
    for task in tune_data
    if task in modalities
}
test_data = pickle.load(open(f"{data_dir}/testingTemporalDataset.pkl", "rb"))
test_data = {
    task: [p for p in test_data[task] if p[-1][4] is not None] for task in test_data
}
task_map = pickle.load(open(f"{data_dir}/taskMap.pkl", "rb"))
valid_tasks = [t for t in tune_data if tune_data[t] and test_data[t] if t in modalities]
tune_data = {task: tune_data[task] for task in valid_tasks}
test_data = {task: test_data[task] for task in valid_tasks}

allResults = {}
config.downstream_batch_size = 1024
config.downstream_epochs = 250
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

    model1 = Embedding2D(label_size).to(device)
    model2 = Embedding2D(label_size, temporal=True).to(device)
    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=config.downstream_lr)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=config.downstream_lr)
    for epoch in tqdm(
        range(config.downstream_epochs), leave=False, desc=f"{task} Tuning"
    ):
        batches_since_step = 0
        for batch_image1, batch_image2, batch_labels in tqdm(
            task_tune_loader, desc=f"{task} Tuning Epoch {epoch+1}", leave=False
        ):
            batch_image1 = batch_image1.to(device)
            batch_image2 = batch_image2.to(device)
            batch_labels = batch_labels.to(device)
            predictions1 = model1(batch_image2)
            predictions2 = model2(torch.cat([batch_image1, batch_image2], dim=1))
            predictions1 = train_activation(predictions1)
            predictions2 = train_activation(predictions2)
            loss1 = loss_fn(predictions1, batch_labels)
            loss2 = loss_fn(predictions2, batch_labels)
            loss1.backward()
            loss2.backward()
            optimizer1.step()
            optimizer2.step()
            optimizer1.zero_grad()
            optimizer2.zero_grad()

    task_preds1 = []
    task_preds2 = []
    task_labels = []
    for batch_image1, batch_image2, batch_labels in tqdm(
        task_test_loader, desc=f"{task} Testing", leave=False
    ):
        batch_image1 = batch_image1.to(device)
        batch_image2 = batch_image2.to(device)
        batch_labels = batch_labels.to(device)
        with torch.no_grad():
            predictions1 = model1(batch_image2)
            predictions2 = model2(torch.cat([batch_image1, batch_image2], dim=1))
            predictions1 = test_activation(predictions1)
            predictions2 = test_activation(predictions2)
            task_preds1.extend(predictions1.cpu().tolist())
            task_preds2.extend(predictions2.cpu().tolist())
            task_labels.extend(batch_labels.cpu().tolist())

    if taskType == "Multi-Label Classification":
        task_preds1 = np.array(task_preds1)
        task_preds2 = np.array(task_preds2)
        task_labels = np.array(task_labels)
        task_rounded_preds1 = np.round(task_preds1)
        task_rounded_preds2 = np.round(task_preds2)
        print("STATIC")
        accPerLabel = [
            metrics.accuracy_score(
                [label[i] for label in task_labels],
                [pred[i] for pred in task_rounded_preds1],
            )
            for i in range(label_size)
        ]
        f1PerLabel = [
            metrics.f1_score(
                [label[i] for label in task_labels],
                [pred[i] for pred in task_rounded_preds1],
            )
            for i in range(label_size)
        ]
        aurocPerLabel = [
            metrics.roc_auc_score(
                [label[i] for label in task_labels], [pred[i] for pred in task_preds1]
            )
            for i in range(label_size)
        ]
        overallAcc = metrics.accuracy_score(
            task_labels.flatten(), task_rounded_preds1.flatten()
        )
        overallF1 = metrics.f1_score(
            task_labels.flatten(), task_rounded_preds1.flatten()
        )
        overallAUROC = metrics.roc_auc_score(
            task_labels.flatten(), task_preds1.flatten()
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
        print("TEMPORAL")
        accPerLabel = [
            metrics.accuracy_score(
                [label[i] for label in task_labels],
                [pred[i] for pred in task_rounded_preds2],
            )
            for i in range(label_size)
        ]
        f1PerLabel = [
            metrics.f1_score(
                [label[i] for label in task_labels],
                [pred[i] for pred in task_rounded_preds2],
            )
            for i in range(label_size)
        ]
        aurocPerLabel = [
            metrics.roc_auc_score(
                [label[i] for label in task_labels], [pred[i] for pred in task_preds2]
            )
            for i in range(label_size)
        ]
        overallAcc = metrics.accuracy_score(
            task_labels.flatten(), task_rounded_preds2.flatten()
        )
        overallF1 = metrics.f1_score(
            task_labels.flatten(), task_rounded_preds2.flatten()
        )
        overallAUROC = metrics.roc_auc_score(
            task_labels.flatten(), task_preds2.flatten()
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
        task_preds1 = np.array(task_preds1)
        task_preds2 = np.array(task_preds2)
        task_labels = np.array(task_labels)
        task_preds1 = np.argmax(task_preds1, axis=1)
        task_preds2 = np.argmax(task_preds2, axis=1)
        print("STATIC")
        acc = metrics.accuracy_score(task_labels, task_preds1)
        f1 = metrics.f1_score(task_labels, task_preds1, average="macro")
        taskResults = {"Accuracy": acc, "F1": f1}
        print(taskResults)
        print("TEMPORAL")
        acc = metrics.accuracy_score(task_labels, task_preds2)
        f1 = metrics.f1_score(task_labels, task_preds2, average="macro")
        taskResults = {"Accuracy": acc, "F1": f1}
        print(taskResults)

    allResults[task] = taskResults
