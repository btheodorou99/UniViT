import torch
import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from src.config import Config
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from src.baselines.external.medcoss.model import MedCoSS
from src.baselines.temporal.data.image_dataset_withoutTime import ImageDataset

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

model_key = "medcoss"

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

config = Config()
cuda_num = 1
device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

data_dir = "/shared/eng/bpt3/data/UniViT/data"
save_dir = "/shared/eng/bpt3/data/UniViT/save"
tune_data = pickle.load(open(f"{data_dir}/tuningTemporalDataset.pkl", "rb"))
tune_data = {
    task: [p for p in tune_data[task] if p[-1][4] is not None] for task in tune_data
}
test_data = pickle.load(open(f"{data_dir}/testingTemporalDataset.pkl", "rb"))
test_data = {
    task: [p for p in test_data[task] if p[-1][4] is not None] for task in test_data
}
task_map = pickle.load(open(f"{data_dir}/taskMap.pkl", "rb"))
valid_tasks = [t for t in tune_data if tune_data[t] and test_data[t]]
tune_data = {task: tune_data[task] for task in valid_tasks}
test_data = {task: test_data[task] for task in valid_tasks}

cci_map = pd.read_csv(f"{data_dir}/CCI.csv")
cci_map = {row['PTID']: sum(row[f'CCI{i}'] for i in range(1,21)) for _, row in cci_map.iterrows()}
cci_values = [v for v in cci_map.values() if v == v]
cci_threshold = np.percentile(cci_values, 75)
cci_map = {k: 1 if v > cci_threshold else 0 for k, v in cci_map.items()}

temporal_tasks = ['ADNI PET CCI', 'ACDC Abnormality', 'MIMIC-CXR Pneumothorax', 'MIMIC-CXR Support Devices', 'CheXpert Pneumothorax', 'CheXpert Support Devices']
tune_data['ADNI PET CCI'] = [[(v[0], v[1], v[2], v[3], cci_map[v[0].split('/')[-1].split('--')[0]]) for v in p] for p in tune_data['ADNI PET'] if p[0][0].split('/')[-1].split('--')[0] in cci_map]
test_data['ADNI PET CCI'] = [[(v[0], v[1], v[2], v[3], cci_map[v[0].split('/')[-1].split('--')[0]]) for v in p] for p in test_data['ADNI PET'] if p[0][0].split('/')[-1].split('--')[0] in cci_map]
task_map['ADNI PET CCI'] = 'Multi-Class Classification'
tune_data['ACDC Abnormality'] = [[(v[0], v[1], v[2], v[3], 1 if v[4] > 0 else 0) for v in p] for p in tune_data['ACDC']]
test_data['ACDC Abnormality'] = [[(v[0], v[1], v[2], v[3], 1 if v[4] > 0 else 0) for v in p] for p in test_data['ACDC']]
task_map['ACDC Abnormality'] = 'Multi-Class Classification'
tune_data['MIMIC-CXR Pneumothorax'] = [[(v[0], v[1], v[2], v[3], v[4][5]) for v in p] for p in tune_data['MIMIC-CXR']]
test_data['MIMIC-CXR Pneumothorax'] = [[(v[0], v[1], v[2], v[3], v[4][5]) for v in p] for p in test_data['MIMIC-CXR']]
task_map['MIMIC-CXR Pneumothorax'] = 'Multi-Class Classification'
tune_data['MIMIC-CXR Support Devices'] = [[(v[0], v[1], v[2], v[3], v[4][6]) for v in p] for p in tune_data['MIMIC-CXR']]
test_data['MIMIC-CXR Support Devices'] = [[(v[0], v[1], v[2], v[3], v[4][6]) for v in p] for p in test_data['MIMIC-CXR']]
task_map['MIMIC-CXR Support Devices'] = 'Multi-Class Classification'
tune_data['CheXpert Pneumothorax'] = [[(v[0], v[1], v[2], v[3], v[4][5]) for v in p] for p in tune_data['CheXpert']]
test_data['CheXpert Pneumothorax'] = [[(v[0], v[1], v[2], v[3], v[4][5]) for v in p] for p in test_data['CheXpert']]
task_map['CheXpert Pneumothorax'] = 'Multi-Class Classification'
tune_data['CheXpert Support Devices'] = [[(v[0], v[1], v[2], v[3], v[4][6]) for v in p] for p in tune_data['CheXpert']]
test_data['CheXpert Support Devices'] = [[(v[0], v[1], v[2], v[3], v[4][6]) for v in p] for p in test_data['CheXpert']]
task_map['CheXpert Support Devices'] = 'Multi-Class Classification'

model = MedCoSS(
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
).to(device)
print("Loading previous model")
model.load_state_dict(
    torch.load(f"{save_dir}/{model_key}.pt", map_location="cpu")["model"]
)
model.eval()
model.requires_grad_(False)

allResults = {}
for task in temporal_tasks:
    if task not in task_map or task_map[task] not in ["Multi-Class Classification", "Multi-Label Classification"]:
        continue
    
    task_tune = tune_data[task]
    task_test = test_data[task]
    task_data = task_test + task_tune
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
        
    print(f'Downstream Evaluation on {task}')
    task_data = ImageDataset(task_data, config, "cpu", multiclass=multiclass)
    task_loader = DataLoader(
        task_data,
        batch_size=config.downstream_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    taskType = task_map[task]
    X = []
    y = []
    for batch_images, batch_dimensions, batch_labels in tqdm(task_loader, desc=f"Generating {task} Embeddings", leave=False):
        batch_images = batch_images.to(device)
        with torch.no_grad():
            representations = model.embed(batch_images, batch_dimensions)
            X.extend(representations.cpu().tolist())
            y.extend(batch_labels.cpu().tolist())
            
    taskResults = {"Accuracy": [], "F1": [], "AUROC": []}
    totData = len(X)
    foldSize = totData // config.downstream_folds
    for fold in tqdm(range(config.downstream_folds), desc=f"Evaluating {task} Folds", leave=False):
        X_train = np.array(X[:fold * foldSize] + X[(fold + 1) * foldSize:])
        y_train = np.array(y[:fold * foldSize] + y[(fold + 1) * foldSize:])
        X_test = np.array(X[fold * foldSize:(fold + 1) * foldSize])
        y_test = np.array(y[fold * foldSize:(fold + 1) * foldSize])
        if len(set(y_test.flatten())) == 1:
            continue
    
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
pickle.dump(allResults, open(f'{save_dir}/{model_key}_temporal_withoutTime_downstreamResults.pkl', 'wb'))
