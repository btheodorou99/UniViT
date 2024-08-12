# https://github.com/bytedance/ibot/tree/main
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from src.config import Config
from torch.utils.data import DataLoader
from src.models.downstream import LinearClassifier
from src.data.image_dataset_pretrained import ImageDataset
from src.baselines.external.models.unimodel_2D import Unified_Model as Model2D
from src.baselines.external.models.unimodel_3D import Unified_Model as Model3D

model_key = "medcoss_pretrained"
EMBEDDING_DIM = 768

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
tune_data = pickle.load(open(f"{data_dir}/tuningDataset.pkl", "rb"))
tune_data = {
    task: [p for p in tune_data[task] if p[4] is not None] for task in tune_data
}
test_data = pickle.load(open(f"{data_dir}/testingDataset.pkl", "rb"))
test_data = {
    task: [p for p in test_data[task] if p[4] is not None] for task in test_data
}
task_map = pickle.load(open(f"{data_dir}/taskMap.pkl", "rb"))
flat_tasks = pickle.load(open(f"{data_dir}/flatTasks.pkl", "rb"))

chkpt = torch.load(f"{save_dir}/medcoss.pt", map_location="cpu")

allResults = {}
for task in tune_data:
    if task not in task_map:
        continue
    
    print(f"\n\nDownstream Evaluation on {task}")
    task_tune = tune_data[task]
    label = task_tune[0][4]
    if isinstance(label, list):
        label_size = len(label)
        multiclass = False
    else:
        label_size = len(set([p[4] for p in task_tune]))
        multiclass = True

    if task in flat_tasks:
        model = Model2D((config.max_height, config.max_height))
    else:
        model = Model3D(
            (config.max_height, config.max_height, max(config.max_depth, 16))
        )

    model.load_state_dict(chkpt["model"], strict=False)
    model.to(device)

    task_tune_data = ImageDataset(
        task_tune,
        config,
        "cpu",
        patch_size=16,
        image_size=config.max_height,
        image_depth=config.max_depth if task not in flat_tasks else None,
        augment=False,
        downstream=True,
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
        patch_size=16,
        image_size=config.max_height,
        image_depth=config.max_depth if task not in flat_tasks else None,
        augment=False,
        downstream=True,
        multiclass=multiclass,
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

    downstream = LinearClassifier(EMBEDDING_DIM, label_size).to(device)
    optimizer = torch.optim.SGD(
        downstream.parameters(), lr=config.downstream_lr, momentum=0.9, weight_decay=0
    )
    for epoch in tqdm(
        range(config.downstream_epochs), leave=False, desc=f"{task} Tuning"
    ):
        batches_since_step = 0
        for batch_images, batch_labels in tqdm(
            task_tune_loader, desc=f"{task} Tuning Epoch {epoch+1}", leave=False
        ):
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            with torch.no_grad():
                representations = model(
                    {
                        "data": batch_images,
                        "modality": "2D image" if task in flat_tasks else "3D image",
                    }
                )
            predictions = downstream(representations)
            predictions = train_activation(predictions)
            loss = loss_fn(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    task_preds = []
    task_labels = []
    for batch_images, batch_labels in tqdm(
        task_test_loader, desc=f"{task} Testing", leave=False
    ):
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        with torch.no_grad():
            representations = model(
                {
                    "data": batch_images,
                    "modality": "2D image" if task in flat_tasks else "3D image",
                }
            )
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
        task_preds = np.array(task_preds)
        task_labels = np.array(task_labels)
        task_preds = np.argmax(task_preds, axis=1)
        acc = metrics.accuracy_score(task_labels, task_preds)
        f1 = metrics.f1_score(task_labels, task_preds, average="macro")
        taskResults = {"Accuracy": acc, "F1": f1}
        print(taskResults)

    allResults[task] = taskResults
pickle.dump(allResults, open(f"{save_dir}/{model_key}_downstreamResults.pkl", "wb"))
