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
    # "ADNI PET CCI",
    # "ADNI MRI CCI",
    # "ACDC NORMAL",
    # "DeepLesion Set",
]

data_dir = "/shared/eng/bpt3/data/UniViT/data"
tune_data = pickle.load(open(f"{data_dir}/tuningTemporalDataset.pkl", "rb"))
test_data = pickle.load(open(f"{data_dir}/testingTemporalDataset.pkl", "rb"))
task_map = pickle.load(open(f"{data_dir}/taskMap.pkl", "rb"))
tune_data = {
    task: [p for p in tune_data[task] if p[-1][4] is not None]
    for task in tune_data if task in modalities
}
test_data = {
    task: [p for p in test_data[task] if p[-1][4] is not None] 
    for task in test_data if task in modalities
}

# cci_map = pd.read_csv(f"{data_dir}/CCI.csv")
# cci_map = {row['PTID']: 1 if sum(row[f'CCI{i}'] for i in range(1,21)) > 40 else 0 for _, row in cci_map.iterrows()}

# tune_data['DeepLesion Set'] = [[(v[0], v[1], v[2], v[3], [1 if any([p[i][4][j] for i in range(len(p))]) else 0 for j in range(len(p[0][4]))]) for v in p] for p in tune_data['DeepLesion']]
# test_data['DeepLesion Set'] = [[(v[0], v[1], v[2], v[3], [1 if any([p[i][4][j] for i in range(len(p))]) else 0 for j in range(len(p[0][4]))]) for v in p] for p in test_data['DeepLesion']]
# task_map['DeepLesion Set'] = 'Multi-Label Classification'
# tune_data['ADNI PET CCI'] = [[(v[0], v[1], v[2], v[3], cci_map[v[0].split('/')[-1].split('--')[0]]) for v in p] for p in tune_data['ADNI PET'] if p[0][0].split('/')[-1].split('--')[0] in cci_map]
# test_data['ADNI PET CCI'] = [[(v[0], v[1], v[2], v[3], cci_map[v[0].split('/')[-1].split('--')[0]]) for v in p] for p in test_data['ADNI PET'] if p[0][0].split('/')[-1].split('--')[0] in cci_map]
# task_map['ADNI PET CCI'] = 'Multi-Class Classification'
# tune_data['ADNI MRI CCI'] = [[(v[0], v[1], v[2], v[3], cci_map[v[0].split('/')[-1].split('--')[0]]) for v in p] for p in tune_data['ADNI MRI'] if p[0][0].split('/')[-1].split('--')[0] in cci_map]
# test_data['ADNI MRI CCI'] = [[(v[0], v[1], v[2], v[3], cci_map[v[0].split('/')[-1].split('--')[0]]) for v in p] for p in test_data['ADNI MRI'] if p[0][0].split('/')[-1].split('--')[0] in cci_map]
# task_map['ADNI MRI CCI'] = 'Multi-Class Classification'
# tune_data['ACDC NORMAL'] = [[(v[0], v[1], v[2], v[3], 1 if v[4] > 0 else 0) for v in p] for p in tune_data['ACDC']]
# test_data['ACDC NORMAL'] = [[(v[0], v[1], v[2], v[3], 1 if v[4] > 0 else 0) for v in p] for p in test_data['ACDC']]
# task_map['ACDC NORMAL'] = 'Multi-Class Classification'

modalities = [
    "ADNI PET",
    "ADNI MRI",
    "ACDC",
    "DeepLesion",
]

valid_tasks = [t for t in tune_data if tune_data[t] and test_data[t] if t in modalities]
tune_data = {task: tune_data[task] for task in valid_tasks}
test_data = {task: test_data[task] for task in valid_tasks}

allResults = {}
config.downstream_batch_size = 32
config.downstream_epochs = 50
for task in tune_data:
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
    model2 = Embedding3D(label_size, temporal=True).to(device)
    model3 = Embedding2D(label_size).to(device)
    model4 = Embedding2D(label_size, temporal=True).to(device)
    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=config.downstream_lr)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=config.downstream_lr)
    optimizer3 = torch.optim.AdamW(model3.parameters(), lr=config.downstream_lr)
    optimizer4 = torch.optim.AdamW(model4.parameters(), lr=config.downstream_lr)
    for epoch in tqdm(
        range(config.downstream_epochs), leave=False, desc=f"{task} Tuning"
    ):
        for batch_image1, batch_image2, batch_labels in tqdm(
            task_tune_loader, desc=f"{task} Tuning Epoch {epoch+1}", leave=False
        ):
            batch_image1 = batch_image1.to(device)
            batch_image2 = batch_image2.to(device)
            batch_labels = batch_labels.to(device)
            predictions1 = model1(batch_image2)
            predictions2 = model2(torch.cat([batch_image1, batch_image2], dim=1))
            predictions3 = model3(batch_image2[:, :, 8, :, :])
            predictions4 = model4(torch.cat([batch_image1[:, :, 8, :, :], batch_image2[:, :, 8, :, :]], dim=1))
            predictions1 = train_activation(predictions1)
            predictions2 = train_activation(predictions2)
            predictions3 = train_activation(predictions3)
            predictions4 = train_activation(predictions4)
            loss1 = loss_fn(predictions1, batch_labels)
            loss2 = loss_fn(predictions2, batch_labels)
            loss3 = loss_fn(predictions3, batch_labels)
            loss4 = loss_fn(predictions4, batch_labels)
            loss1.backward()
            loss2.backward()
            loss3.backward()
            loss4.backward()
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            optimizer4.zero_grad()

    task_preds1 = []
    task_preds2 = []
    task_preds3 = []
    task_preds4 = []
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
            predictions3 = model3(batch_image2[:, :, 8, :, :])
            predictions4 = model4(torch.cat([batch_image1[:, :, 8, :, :], batch_image2[:, :, 8, :, :]], dim=1))
            predictions1 = test_activation(predictions1)
            predictions2 = test_activation(predictions2)
            predictions3 = test_activation(predictions3)
            predictions4 = test_activation(predictions4)
            task_preds1.extend(predictions1.cpu().tolist())
            task_preds2.extend(predictions2.cpu().tolist())
            task_preds3.extend(predictions3.cpu().tolist())
            task_preds4.extend(predictions4.cpu().tolist())
            task_labels.extend(batch_labels.cpu().tolist())

    if taskType == "Multi-Label Classification":
        task_preds1 = np.array(task_preds1)
        task_preds2 = np.array(task_preds2)
        task_preds3 = np.array(task_preds3)
        task_preds4 = np.array(task_preds4)
        task_labels = np.array(task_labels)
        task_rounded_preds1 = np.round(task_preds1)
        task_rounded_preds2 = np.round(task_preds2)
        task_rounded_preds3 = np.round(task_preds3)
        task_rounded_preds4 = np.round(task_preds4)
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
        print("TEMPORAL 3D")
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
        print("TEMPORAL 2D")
        accPerLabel = [
            metrics.accuracy_score(
                [label[i] for label in task_labels],
                [pred[i] for pred in task_rounded_preds4],
            )
            for i in range(label_size)
        ]
        f1PerLabel = [
            metrics.f1_score(
                [label[i] for label in task_labels],
                [pred[i] for pred in task_rounded_preds4],
            )
            for i in range(label_size)
        ]
        aurocPerLabel = [
            metrics.roc_auc_score(
                [label[i] for label in task_labels], [pred[i] for pred in task_preds4]
            )
            for i in range(label_size)
        ]
        overallAcc = metrics.accuracy_score(
            task_labels.flatten(), task_rounded_preds4.flatten()
        )
        overallF1 = metrics.f1_score(
            task_labels.flatten(), task_rounded_preds4.flatten()
        )
        overallAUROC = metrics.roc_auc_score(
            task_labels.flatten(), task_preds4.flatten()
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
        task_probs2 = np.array(task_preds2)
        task_probs3 = np.array(task_preds3)
        task_probs4 = np.array(task_preds4)
        if label_size == 2:
            task_probs1 = task_probs1[:, 1]
            task_probs2 = task_probs2[:, 1]
            task_probs3 = task_probs3[:, 1]
            task_probs4 = task_probs4[:, 1]
            task_preds1 = np.round(task_probs1)
            task_preds2 = np.round(task_probs2)
            task_preds3 = np.round(task_probs3)
            task_preds4 = np.round(task_probs4)
        else:
            task_preds1 = np.argmax(task_probs1, axis=1)
            task_preds2 = np.argmax(task_probs2, axis=1)
            task_preds3 = np.argmax(task_probs3, axis=1)
            task_preds4 = np.argmax(task_probs4, axis=1)
        print("STATIC 3D")
        acc = metrics.accuracy_score(task_labels, task_preds1)
        f1 = metrics.f1_score(task_labels, task_preds1, average="macro")
        auroc = metrics.roc_auc_score(task_labels, task_probs1, average="macro", multi_class="ovr")
        taskResults = {"Accuracy": acc, "F1": f1, "AUROC": auroc}
        print('\t', taskResults)
        print("TEMPORAL 3D")
        acc = metrics.accuracy_score(task_labels, task_preds2)
        f1 = metrics.f1_score(task_labels, task_preds2, average="macro")
        auroc = metrics.roc_auc_score(task_labels, task_probs2, average="macro", multi_class="ovr")
        taskResults = {"Accuracy": acc, "F1": f1, "AUROC": auroc}
        print('\t', taskResults)
        print("STATIC 2D")
        acc = metrics.accuracy_score(task_labels, task_preds3)
        f1 = metrics.f1_score(task_labels, task_preds3, average="macro")
        auroc = metrics.roc_auc_score(task_labels, task_probs3, average="macro", multi_class="ovr")
        taskResults = {"Accuracy": acc, "F1": f1, "AUROC": auroc}
        print('\t', taskResults)
        print("TEMPORAL 2D")
        acc = metrics.accuracy_score(task_labels, task_preds4)
        f1 = metrics.f1_score(task_labels, task_preds4, average="macro")
        auroc = metrics.roc_auc_score(task_labels, task_probs4, average="macro", multi_class="ovr")
        taskResults = {"Accuracy": acc, "F1": f1, "AUROC": auroc}
        print('\t', taskResults)

    allResults[task] = taskResults
