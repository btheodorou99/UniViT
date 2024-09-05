import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from src.config import Config
from torch.utils.data import DataLoader
from src.models.univit import UniViT
from src.baselines.temporal.data.image_dataset import ImageDataset
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier


# Label Change (Pred and Not Pred)
# 5 Visits (Pred and Not Pred)
# 3 Visits (Pred and Not Pred)
# 2 Visits (Not Pred)
# ADNI Binary (1 or not 0) (Pred and Not Pred)
# ADNI Binary (2 to 1, 0 other) (Pred and Not Pred)
# ACDC Binary (1 if not 0) (Pred and Not Pred)

def hasChanged(p):
    if isinstance(p[-1][4], list):
        finalList = p[-1][4]
        for i in range(len(p) - 1):
            for j in range(len(finalList)):
                if p[i][4][j] != finalList[j]:
                    return True
    elif isinstance(p[-1][4], int):
        finalVal = p[-1][4]
        for i in range(len(p) - 1):
            if p[i][4] != finalVal:
                return True
    return False

model_key = "univit_more"

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
# save_dir = "/srv/local/data/bpt3/UniViT/save"
task_map = pickle.load(open(f"{data_dir}/taskMap.pkl", "rb"))
tune_data = pickle.load(open(f"{data_dir}/tuningTemporalDataset.pkl", "rb"))
tune_data = {
    task: [p for p in tune_data[task] if p[-1][4] is not None] for task in tune_data  if task_map[task] in ["Multi-Class Classification", "Multi-Label Classification"]
}
name_map = {m: m for m in tune_data}
toAdd = {}
for mod in tune_data:
    name_map[f'{mod} 5'] = mod
    toAdd[f'{mod} 5'] = [p for p in tune_data[mod] if len(p) >= 5]
    name_map[f'{mod} 3'] = mod
    toAdd[f'{mod} 3'] = [p for p in tune_data[mod] if len(p) <= 3]
    name_map[f'{mod} Change'] = mod
    toAdd[f'{mod} Change'] = [p for p in tune_data[mod] if hasChanged(p)]
for mod in toAdd:
    tune_data[mod] = toAdd[mod]
name_map['All PET'] = 'Amyloid PET'
tune_data['All PET'] = tune_data['Amyloid PET'] + tune_data['FDG PET']
name_map['MRI CN'] = 'MRI'
tune_data['MRI CN'] = [[(v[0], v[1], v[2], v[3], 1 if v[4] != 0 else 0) for v in p] for p in tune_data['MRI']]
name_map['PET CN'] = 'Amyloid PET'
tune_data['PET CN'] = [[(v[0], v[1], v[2], v[3], 1 if v[4] != 0 else 0) for v in p] for p in tune_data['All PET']]
name_map['MRI AD'] = 'MRI'
tune_data['MRI AD'] = [[(v[0], v[1], v[2], v[3], 1 if v[4] == 2 else 0) for v in p] for p in tune_data['MRI']]
name_map['PET AD'] = 'Amyloid PET'
tune_data['PET AD'] = [[(v[0], v[1], v[2], v[3], 1 if v[4] == 2 else 0) for v in p] for p in tune_data['All PET']]
name_map['ACDC Normal'] = 'Cardiac MRI (ACDC)'
tune_data['ACDC Normal'] = [[(v[0], v[1], v[2], v[3], 1 if v[4] != 0 else 0) for v in p] for p in tune_data['Cardiac MRI (ACDC)']]
test_data = pickle.load(open(f"{data_dir}/testingTemporalDataset.pkl", "rb"))
test_data = {
    task: [p for p in test_data[task] if p[-1][4] is not None] for task in test_data  if task_map[task] in ["Multi-Class Classification", "Multi-Label Classification"]
}
toAdd = {}
for mod in test_data:
    toAdd[f'{mod} 5'] = [p for p in test_data[mod] if len(p) >= 5]
    toAdd[f'{mod} 3'] = [p for p in test_data[mod] if len(p) <= 3]
    toAdd[f'{mod} Change'] = [p for p in test_data[mod] if hasChanged(p)]
for mod in toAdd:
    test_data[mod] = toAdd[mod]
test_data['All PET'] = test_data['Amyloid PET'] + test_data['FDG PET']
test_data['MRI CN'] = [[(v[0], v[1], v[2], v[3], 1 if v[4] != 0 else 0) for v in p] for p in test_data['MRI']]
test_data['PET CN'] = [[(v[0], v[1], v[2], v[3], 1 if v[4] != 0 else 0) for v in p] for p in test_data['All PET']]
test_data['MRI AD'] = [[(v[0], v[1], v[2], v[3], 1 if v[4] == 2 else 0) for v in p] for p in test_data['MRI']]
test_data['PET AD'] = [[(v[0], v[1], v[2], v[3], 1 if v[4] == 2 else 0) for v in p] for p in test_data['All PET']]
test_data['ACDC Normal'] = [[(v[0], v[1], v[2], v[3], 1 if v[4] != 0 else 0) for v in p] for p in test_data['Cardiac MRI (ACDC)']]
valid_tasks = [t for t in tune_data if tune_data[t] and test_data[t]]
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
).to(device)
print("Loading previous model")
model.load_state_dict(
    torch.load(f"{save_dir}/{model_key}.pt", map_location="cpu")["model"], strict=False
)
model.eval()
model.requires_grad_(False)

allResults = {}
hasSeen = False
for task in tune_data:
    # if 'MRI CN' in task:
    #     hasSeen = True
    # if not hasSeen:
    #     continue
    # if task not in task_map or task_map[task] not in ["Multi-Class Classification", "Multi-Label Classification"]:
    #     continue
    
    task_tune = tune_data[task]
    label = task_tune[0][0][4]
    if isinstance(label, list):
        label_size = len(label)
        multiclass = False
    elif isinstance(label, int):
        labels = tuple(sorted(set([p[0][4] for p in task_tune])))
        label_size = len(labels)
        multiclass = True
        if label_size == 1 or max(labels) != label_size - 1 or tuple(sorted(set([p[0][4] for p in test_data[task]]))) != labels:
            continue
    else:
        continue
        
    print(f'\n\nDownstream Evaluation on {task}')
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

    taskType = task_map[name_map[task]]
    downstream = LogisticRegression(max_iter=10000)
    if taskType == "Multi-Label Classification":
        downstream = MultiOutputClassifier(downstream)
    X = []
    y = []

    for batch_images, batch_dimensions, batch_labels in tqdm(task_tune_loader, desc=f"{task} Tuning", leave=False):
        batch_images = batch_images.to(device)
        with torch.no_grad():
            representations = model.embed(batch_images, batch_dimensions)
            X.extend(representations.cpu().tolist())
            y.extend(batch_labels.cpu().tolist())

    downstream.fit(X, y)

    task_preds = []
    task_labels = []
    for batch_images, batch_dimensions, batch_labels in tqdm(
        task_test_loader, desc=f"{task} Testing", leave=False
    ):
        batch_images = batch_images.to(device)
        with torch.no_grad():
            representations = model.embed(batch_images, batch_dimensions)
            predictions = downstream.predict_proba(representations.cpu().numpy())
            if taskType == "Multi-Label Classification":
                predictions = np.array(predictions).transpose(1, 0, 2)[:,:,1]
            task_preds.extend(predictions.tolist())
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
        if label_size == 2:
            auroc = metrics.roc_auc_score(task_labels, task_probs[:, 1])
        else:
            auroc = metrics.roc_auc_score(task_labels, task_probs, average="macro", multi_class="ovr")
        taskResults = {"Accuracy": acc, "F1": f1, "AUROC": auroc}
        print(taskResults)

    allResults[task] = taskResults
# pickle.dump(allResults, open(f"{save_dir}/{model_key}_temporal_pred_downstreamResults.pkl", "wb"))
