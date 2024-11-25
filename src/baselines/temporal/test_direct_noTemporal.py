import torch
import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from src.config import Config
from torch.utils.data import DataLoader
from src.models.direct import Embedding2D, Embedding3D
from src.baselines.temporal.data.image_dataset_direct import ImageDataset

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

config = Config()
cuda_num = 0
device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

modalities = [
    "ACDC",
    "ADNI PET",
    "ADNI MRI",
    "DeepLesion",
]

data_dir = "/shared/eng/bpt3/data/UniViT/data"
tune_data = pickle.load(open(f"{data_dir}/tuningDataset.pkl", "rb"))
test_data = pickle.load(open(f"{data_dir}/testingDataset.pkl", "rb"))
task_map = pickle.load(open(f"{data_dir}/taskMap.pkl", "rb"))
tune_data = {
    task: [[p] for p in tune_data[task] if p[4] is not None]
    for task in tune_data if task in modalities
}
test_data = {
    task: [[p] for p in test_data[task] if p[4] is not None] 
    for task in test_data if task in modalities
}

valid_tasks = [t for t in tune_data if tune_data[t] and test_data[t] if t in modalities]
tune_data = {task: tune_data[task] for task in valid_tasks}
test_data = {task: test_data[task] for task in valid_tasks}

allResults = {}
config.downstream_batch_size = 32
config.downstream_epochs = 50
for task in modalities:
    print(f"Downstream Evaluation on {task}")
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
        task_tune, config, "cpu", temporal=False, multiclass=multiclass
    )
    task_tune_loader = DataLoader(
        task_tune_data,
        batch_size=config.downstream_batch_size,
        shuffle=True,
        num_workers=int(2*config.num_workers),
        pin_memory=True,
        prefetch_factor=16,
        persistent_workers=True,
    )
    task_test = test_data[task]
    task_test_data = ImageDataset(
        task_test, config, "cpu", temporal=False, multiclass=multiclass
    )
    task_test_loader = DataLoader(
        task_test_data,
        batch_size=config.downstream_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    taskType = task_map[task]
    if taskType == "Multi-Label Classification":
        loss_fn = torch.nn.BCELoss()
        train_activation = torch.nn.Sigmoid()
        test_activation = torch.nn.Sigmoid()
    elif taskType == "Multi-Class Classification":
        loss_fn = torch.nn.CrossEntropyLoss()
        train_activation = torch.nn.Identity()
        test_activation = torch.nn.Softmax(dim=1)
    else:
        continue

    model1 = Embedding3D(label_size).to(device)
    model3 = Embedding2D(label_size).to(device)
    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=config.downstream_lr)
    optimizer3 = torch.optim.AdamW(model3.parameters(), lr=config.downstream_lr)
    for epoch in tqdm(
        range(config.downstream_epochs), leave=False, desc=f"{task} Tuning"
    ):
        for batch_image2, batch_labels in tqdm(
            task_tune_loader, desc=f"{task} Tuning Epoch {epoch+1}", leave=False
        ):
            batch_image2 = batch_image2.to(device)
            batch_labels = batch_labels.to(device)
            predictions1 = model1(batch_image2)
            predictions3 = model3(batch_image2[:, :, 122, :, :])
            predictions1 = train_activation(predictions1)
            predictions3 = train_activation(predictions3)
            loss1 = loss_fn(predictions1, batch_labels)
            loss3 = loss_fn(predictions3, batch_labels)
            loss1.backward()
            loss3.backward()
            optimizer1.step()
            optimizer3.step()
            optimizer1.zero_grad()
            optimizer3.zero_grad()

    task_preds1 = []
    task_preds3 = []
    task_labels = []
    for batch_image2, batch_labels in tqdm(
        task_test_loader, desc=f"{task} Testing", leave=False
    ):
        batch_image2 = batch_image2.to(device)
        batch_labels = batch_labels.to(device)
        with torch.no_grad():
            predictions1 = model1(batch_image2)
            predictions3 = model3(batch_image2[:, :, 122, :, :])
            predictions1 = test_activation(predictions1)
            predictions3 = test_activation(predictions3)
            task_preds1.extend(predictions1.cpu().tolist())
            task_preds3.extend(predictions3.cpu().tolist())
            task_labels.extend(batch_labels.cpu().tolist())

    if taskType == "Multi-Label Classification":
        task_preds1 = np.array(task_preds1)
        task_preds3 = np.array(task_preds3)
        task_labels = np.array(task_labels)
        task_rounded_preds1 = np.round(task_preds1)
        task_rounded_preds3 = np.round(task_preds3)
        print("STATIC 3D")
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
        print('\t', taskResults)
        print("STATIC 2D")
        accPerLabel = [
            metrics.accuracy_score(
                [label[i] for label in task_labels],
                [pred[i] for pred in task_rounded_preds3],
            )
            for i in range(label_size)
        ]
        f1PerLabel = [
            metrics.f1_score(
                [label[i] for label in task_labels],
                [pred[i] for pred in task_rounded_preds3],
            )
            for i in range(label_size)
        ]
        aurocPerLabel = [
            metrics.roc_auc_score(
                [label[i] for label in task_labels], [pred[i] for pred in task_preds3]
            )
            for i in range(label_size)
        ]
        overallAcc = metrics.accuracy_score(
            task_labels.flatten(), task_rounded_preds3.flatten()
        )
        overallF1 = metrics.f1_score(
            task_labels.flatten(), task_rounded_preds3.flatten()
        )
        overallAUROC = metrics.roc_auc_score(
            task_labels.flatten(), task_preds3.flatten()
        )
        taskResults = {
            "Accuracy": overallAcc,
            "F1": overallF1,
            "AUROC": overallAUROC,
            "Accuracy Per Label": accPerLabel,
            "F1 Per Label": f1PerLabel,
            "AUROC Per Label": aurocPerLabel,
        }
        print('\t', taskResults)
    elif taskType == "Multi-Class Classification":
        task_labels = np.array(task_labels)
        task_probs1 = np.array(task_preds1)
        task_probs3 = np.array(task_preds3)
        if label_size == 2:
            task_probs1 = task_probs1[:, 1]
            task_probs3 = task_probs3[:, 1]
            task_preds1 = np.round(task_probs1)
            task_preds3 = np.round(task_probs3)
        else:
            task_preds1 = np.argmax(task_probs1, axis=1)
            task_preds3 = np.argmax(task_probs3, axis=1)
        print("STATIC 3D")
        acc = metrics.accuracy_score(task_labels, task_preds1)
        f1 = metrics.f1_score(task_labels, task_preds1, average="macro")
        auroc = metrics.roc_auc_score(task_labels, task_probs1, average="macro", multi_class="ovr")
        taskResults = {"Accuracy": acc, "F1": f1, "AUROC": auroc}
        print('\t', taskResults)
        print("STATIC 2D")
        acc = metrics.accuracy_score(task_labels, task_preds3)
        f1 = metrics.f1_score(task_labels, task_preds3, average="macro")
        auroc = metrics.roc_auc_score(task_labels, task_probs3, average="macro", multi_class="ovr")
        taskResults = {"Accuracy": acc, "F1": f1, "AUROC": auroc}
        print('\t', taskResults)

    allResults[task] = taskResults
