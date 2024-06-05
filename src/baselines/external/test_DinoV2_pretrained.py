
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from src.config import Config
from torch.utils.data import DataLoader
from src.models.downstream import DownstreamModel
from src.data.image_dataset_flat import ImageDataset

model_key = 'dinov2_pretrained'
EMBEDDING_DIM = 768

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

config = Config()
cuda_num = 3
device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)

config.downstream_batch_size = config.downstream_effective_batch_size
batches_per_step = config.downstream_effective_batch_size // config.downstream_batch_size
data_dir = '/shared/bpt3/data/UniViT/data'
save_dir = '/shared/bpt3/data/UniViT/save'

flat_tasks = ['Chest X-Ray (MIMIC)', 'Chest X-Ray (CheXpert)', 'Skin Lesion']
tune_data = pickle.load(open(f'{data_dir}/tuningDataset.pkl', 'rb'))
tune_data = {task: [p for p in tune_data[task] if p[4] is not None] for task in tune_data}
test_data = pickle.load(open(f'{data_dir}/testingDataset.pkl', 'rb'))
test_data = {task: [p for p in test_data[task] if p[4] is not None] for task in test_data}
task_map = pickle.load(open(f'{data_dir}/taskMap.pkl', 'rb'))

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
model.eval()
model.requires_grad_(False)

allResults = {'Chest X-Ray (MIMIC)':
{'Accuracy': 0.8424386990868985, 'F1': 0.4125466194893373, 'AUROC': 0.8252569983969216, 'Accuracy Per Label': [0.8193546732327895, 0.8316405047706987, 0.8986098286652303, 0.7948086590745871, 0.7512054991279368, 0.9606032625423208, 0.8408484661947265], 'F1 Per Label': [0.035601807476379574, 0.01824708345797188, 0.1651531151003168, 0.010880316518298714, 0.6761917478969154, 0.0, 0.5834172541121181], 'AUROC Per Label': [0.7532675656535621, 0.7409940257409655, 0.8449361066327945, 0.6966786175293662, 0.8118027249734094, 0.8024983958633718, 0.8746367334967934]},
'Chest X-Ray (CheXpert)': {'Accuracy': 0.8093162958161386, 'F1': 0.505846153846154, 'AUROC': 0.8194782282946395, 'Accuracy Per Label': [0.8574958443775551, 0.8695359180556179, 0.7776629677883103, 0.6342602992048161, 0.897749225032571, 0.9123051350015724, 0.7162046812525271], 'F1 Per Label': [0.0, 0.01425661914460285, 0.2894472361809045, 0.6107578293091083, 0.10252365930599369, 0.0, 0.7080733860159897], 'AUROC Per Label': [0.5966335471526112, 0.6939633185024718, 0.7778049436600639, 0.6824595039924353, 0.8072572420924868, 0.745095656719075, 0.7889873990129926]}
}
for task in flat_tasks:
    print(f'\n\nDownstream Evaluation on {task}')
    task_tune = tune_data[task]
    label =  task_tune[0][4]
    if isinstance(label, list):
        label_size = len(label)
        multiclass = False
    else:
        label_size = len(set([p[4] for p in task_tune]))
        multiclass = True
        
    task_tune_data = ImageDataset(task_tune, config, 'cpu', patch_size=14, image_size = 320 if task == 'Chest X-Ray (CheXpert)' else 450 if task == 'Skin Lesion' else None, augment=False, downstream=True, multiclass=multiclass)
    task_tune_loader = DataLoader(task_tune_data, batch_size=config.downstream_batch_size, shuffle=True, num_workers=config.num_workers)
    task_test = test_data[task]
    task_test_data = ImageDataset(task_test, config, 'cpu', patch_size=14, image_size = 320 if task == 'Chest X-Ray (CheXpert)' else 450 if task == 'Skin Lesion' else None, augment=False, downstream=True, multiclass=multiclass)
    task_test_loader = DataLoader(task_test_data, batch_size=config.downstream_batch_size, shuffle=False, num_workers=config.num_workers)
    
    taskType = task_map[task]
    if taskType == 'Multi-Label Classification':
        loss_fn = torch.nn.BCELoss()
        train_activation = torch.nn.Sigmoid()
        test_activation = torch.nn.Sigmoid()
    elif taskType == 'Multi-Class Classification':
        loss_fn = torch.nn.CrossEntropyLoss()
        train_activation = torch.nn.Identity()
        test_activation = torch.nn.Softmax(dim=1)
    else:
        raise ValueError('Invalid task type')
    
    downstream = DownstreamModel(EMBEDDING_DIM, label_size).to(device)
    optimizer = torch.optim.Adam(downstream.parameters(), lr=config.downstream_lr)
    for epoch in tqdm(range(config.downstream_epochs), leave=False, desc=f'{task} Tuning'):
        batches_since_step = 0
        for batch_images, batch_labels in tqdm(task_tune_loader, desc=f'{task} Tuning Epoch {epoch+1}', leave=False):
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            with torch.no_grad():
                representations = model(batch_images)
            predictions = downstream(representations)
            predictions = train_activation(predictions)
            loss = loss_fn(predictions, batch_labels)
            loss = loss / batches_per_step
            loss.backward()
            batches_since_step += 1
            if batches_since_step == batches_per_step:
                optimizer.step()
                optimizer.zero_grad()
                batches_since_step = 0

    task_preds = []
    task_labels = []
    for batch_images, batch_labels in tqdm(task_test_loader, desc=f'{task} Testing', leave=False):
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        with torch.no_grad():
            representations = model(batch_images)
            predictions = downstream(representations)
            predictions = test_activation(predictions)
            task_preds.extend(predictions.cpu().tolist())
            task_labels.extend(batch_labels.cpu().tolist())
            
    if taskType == 'Multi-Label Classification':
        task_preds = np.array(task_preds)
        task_labels = np.array(task_labels)
        task_rounded_preds = np.round(task_preds)
        accPerLabel = [metrics.accuracy_score([label[i] for label in task_labels], [pred[i] for pred in task_rounded_preds]) for i in range(label_size)]
        f1PerLabel = [metrics.f1_score([label[i] for label in task_labels], [pred[i] for pred in task_rounded_preds]) for i in range(label_size)]
        aurocPerLabel = [metrics.roc_auc_score([label[i] for label in task_labels], [pred[i] for pred in task_preds]) for i in range(label_size)]
        overallAcc = metrics.accuracy_score(task_labels.flatten(), task_rounded_preds.flatten())
        overallF1 = metrics.f1_score(task_labels.flatten(), task_rounded_preds.flatten())
        overallAUROC = metrics.roc_auc_score(task_labels.flatten(), task_preds.flatten())
        taskResults = {'Accuracy': overallAcc, 'F1': overallF1, 'AUROC': overallAUROC, 'Accuracy Per Label': accPerLabel, 'F1 Per Label': f1PerLabel, 'AUROC Per Label': aurocPerLabel}
        print(taskResults)
    elif taskType == 'Multi-Class Classification':
        task_preds = np.array(task_preds)
        task_labels = np.array(task_labels)
        task_preds = np.argmax(task_preds, axis=1)
        acc = metrics.accuracy_score(task_labels, task_preds)
        f1 = metrics.f1_score(task_labels, task_preds, average='macro')
        taskResults = {'Accuracy': acc, 'F1': f1}
        print(taskResults)
        
    allResults[task] = taskResults
pickle.dump(allResults, open(f'{save_dir}/{model_key}_downstreamResults.pkl', 'wb'))