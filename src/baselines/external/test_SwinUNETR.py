import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from src.config import Config
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from src.data.image_dataset_pretrained import ImageDataset
from src.baselines.external.swinunetr.models.ssl_head import SSLHead

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

model_key = "swinunetr"

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

config = Config()
cuda_num = 2
device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

data_dir = "/shared/eng/bpt3/data/UniViT/data"
save_dir = "/shared/eng/bpt3/data/UniViT/save"
tune_data_static = pickle.load(open(f"{data_dir}/tuningDataset.pkl", "rb"))
tune_data_temporal = pickle.load(open(f"{data_dir}/tuningTemporalDataset.pkl", "rb"))
tune_data_temporal = {m: [[v for v in p if v[4] is not None] for p in tune_data_temporal[m]] for m in tune_data_temporal}
tune_data_temporal_paths = {m: set([v[0] for p in tune_data_temporal[m] for v in p]) for m in tune_data_temporal}
tune_data_static = {m: [[v] for v in tune_data_static[m] if v[4] is not None and (m not in tune_data_temporal_paths or v[0] not in tune_data_temporal_paths[m])] for m in tune_data_static}
tune_data = {m: tune_data_static[m] + (tune_data_temporal[m] if m in tune_data_temporal else []) for m in tune_data_static}

test_data_static = pickle.load(open(f"{data_dir}/testingDataset.pkl", "rb"))
test_data_temporal = pickle.load(open(f"{data_dir}/testingTemporalDataset.pkl", "rb"))
test_data_temporal = {m: [[v for v in p if v[4] is not None] for p in test_data_temporal[m]] for m in test_data_temporal}
test_data_temporal_paths = {m: set([v[0] for p in test_data_temporal[m] for v in p]) for m in test_data_temporal}
test_data_static = {m: [[v] for v in test_data_static[m] if v[4] is not None and (m not in test_data_temporal_paths or v[0] not in test_data_temporal_paths[m])] for m in test_data_static}
test_data = {m: test_data_static[m] + (test_data_temporal[m] if m in test_data_temporal else []) for m in tune_data_static}

task_map = pickle.load(open(f"{data_dir}/taskMap.pkl", "rb"))

config.in_channels = 1
config.feature_size = 48
config.dropout_path_rate = 0.0
config.use_checkpoint = False
config.spatial_dims = 3

model = model = SSLHead(config)
state_dict = torch.load(f"{save_dir}/{model_key}.pt", map_location='cpu')['state_dict']
model.load_state_dict(state_dict)
model.eval()
model.requires_grad_(False)
model.to(device)

allResults = {}
for task in ['ADNI PET', 'ADNI MRI', 'ACDC', 'DeepLesion']:
    if task not in task_map or task_map[task] not in ["Multi-Class Classification", "Multi-Label Classification"]:
        continue
    
    task_tune = tune_data[task]
    task_test = test_data[task]
    task_data = task_tune + task_test
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
    
    task_idx = np.array([i for i, p in enumerate(task_data) for v in p])
    task_data = [v for p in task_data for v in p]
        
    print(f'Downstream Evaluation on {task}')
    task_data = ImageDataset(
        task_data, config, "cpu", image_depth=True, multiclass=multiclass
    )
    task_loader = DataLoader(
        task_data,
        batch_size=config.downstream_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    taskType = task_map[task]
    X = []
    y = []
    for batch_images, batch_labels in tqdm(task_loader, desc=f"Generating {task} Embeddings", leave=False):
        batch_images = batch_images.to(device)
        with torch.no_grad():
            representations = model(batch_images, features_only=True)
            X.extend(representations.cpu().tolist())
            y.extend(batch_labels.cpu().tolist())
            
    X = np.array(X)
    y = np.array(y)
    
    taskResults = {"Accuracy": [], "F1": [], "AUROC": []}
    totData = max(task_idx) + 1
    foldSize = totData // config.downstream_folds
    for fold in tqdm(range(config.downstream_folds), desc=f"Evaluating {task} Folds", leave=False):
        fold_train_idx = np.where((task_idx < fold * foldSize) | (task_idx >= (fold + 1) * foldSize))[0]
        X_train = X[fold_train_idx]
        y_train = y[fold_train_idx]
        fold_test_idx = np.where((task_idx >= fold * foldSize) & (task_idx < (fold + 1) * foldSize))[0]
        X_test = X[fold_test_idx]
        y_test = y[fold_test_idx]
    
        downstream = LogisticRegression(max_iter=10000)
        if taskType == "Multi-Label Classification":
            downstream = MultiOutputClassifier(downstream)

        downstream.fit(X_train, y_train)
        task_probs = downstream.predict_proba(X_test)
        if taskType == "Multi-Label Classification":
            task_probs = np.array(task_probs).transpose(1, 0, 2)[:,:,1]
        elif label_size == 2:
            task_probs = task_probs[:,1]
            
        if taskType == "Multi-Label Classification" or label_size == 2:
            task_preds = np.round(task_probs)
            acc = metrics.accuracy_score(y_test.flatten(), task_preds.flatten())
            f1 = metrics.f1_score(y_test.flatten(), task_preds.flatten())
            auroc = metrics.roc_auc_score(y_test.flatten(), task_probs.flatten())
        elif taskType == "Multi-Class Classification":
            task_preds = np.argmax(task_probs, axis=1)
            acc = metrics.accuracy_score(y_test, task_preds)
            f1 = metrics.f1_score(y_test, task_preds, average="macro")
            present_classes = np.unique(y_test)
            if len(present_classes) != label_size:
                task_probs = task_probs[:, present_classes]
                task_probs = task_probs / task_probs.sum(axis=1, keepdims=True)
                label_mapping = {c: i for i, c in enumerate(present_classes)}
                y_test = np.array([label_mapping[c] for c in y_test])
            auroc = metrics.roc_auc_score(y_test, task_probs, average="macro", multi_class="ovr")

        taskResults["Accuracy"].append(acc)
        taskResults["F1"].append(f1)
        taskResults["AUROC"].append(auroc)

    taskResults["Accuracy PM"] = round(np.std(taskResults["Accuracy"]) / np.sqrt(config.downstream_folds), 5)
    taskResults["Accuracy"] = round(np.mean(taskResults["Accuracy"]), 5)
    taskResults["F1 PM"] = round(np.std(taskResults["F1"]) / np.sqrt(config.downstream_folds), 5)
    taskResults["F1"] = round(np.mean(taskResults["F1"]), 5)
    taskResults["AUROC PM"] = round(np.std(taskResults["AUROC"]) / np.sqrt(config.downstream_folds), 5)
    taskResults["AUROC"] = round(np.mean(taskResults["AUROC"]), 5)
    print('\t', taskResults)
    
    allResults[task] = taskResults
pickle.dump(allResults, open(f"{save_dir}/{model_key}_downstreamResults.pkl", "wb"))
