# https://github.com/bytedance/ibot/tree/main
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from src.config import Config
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from src.data.image_dataset_pretrained import ImageDataset
from src.baselines.external.ibot.models.vision_transformer import vit_base

model_key = "ibot"

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
    task: [p for p in tune_data[task] if p[4] is not None] for task in tune_data
}
test_data = pickle.load(open(f"{data_dir}/testingDataset.pkl", "rb"))
test_data = {
    task: [p for p in test_data[task] if p[4] is not None] for task in test_data
}
task_map = pickle.load(open(f"{data_dir}/taskMap.pkl", "rb"))

model = vit_base(patch_size=config.patch_size)
model.load_state_dict(torch.load(f"{save_dir}/{model_key}.pt", map_location='cpu')['student'])
model.eval()
model.requires_grad_(False)
model.to(device)

transform = transforms.Compose([
    transforms.Resize((config.max_height, config.max_width), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

allResults = {}
for task in tune_data:
    if task not in task_map or task_map[task] not in ["Multi-Class Classification", "Multi-Label Classification"]:
        continue
    
    task_tune = tune_data[task]
    label = task_tune[0][4]
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
    
    print(f"\n\nDownstream Evaluation on {task}")
    task_tune_data = ImageDataset(
        task_tune,
        config,
        "cpu",
        transform=transform,
        multiclass=multiclass,
    )
    task_tune_loader = DataLoader(
        task_tune_data,
        batch_size=config.downstream_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    task_test = test_data[task]
    task_test_data = ImageDataset(
        task_test,
        config,
        "cpu",
        transform=transform,
        multiclass=multiclass,
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

    for batch_images, batch_labels in tqdm(task_tune_loader, desc=f"{task} Tuning", leave=False):
        batch_images = batch_images.to(device)
        with torch.no_grad():
            representations = model(batch_images)
            X.extend(representations.cpu().tolist())
            y.extend(batch_labels.cpu().tolist())

    downstream.fit(X, y)

    task_preds = []
    task_labels = []
    for batch_images, batch_labels in tqdm(
        task_test_loader, desc=f"{task} Testing", leave=False
    ):
        batch_images = batch_images.to(device)
        with torch.no_grad():
            representations = model(batch_images)
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
        auroc = metrics.roc_auc_score(task_labels, task_probs, average="macro", multi_class="ovr")
        taskResults = {"Accuracy": acc, "F1": f1, "AUROC": auroc}
        print(taskResults)

    allResults[task] = taskResults
pickle.dump(allResults, open(f"{save_dir}/{model_key}_downstreamResults.pkl", "wb"))
