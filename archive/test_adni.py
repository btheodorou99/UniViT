import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from src.config import Config
from torch.utils.data import DataLoader
from src.data.adni_dataset import ImageDataset
from src.models.adni_cnn import CNNModel, LinearModel

model_key = 'testtest'
labels = [5, 6, 7, 8, 9, 10, 11, 15, 30, 31, 32, 57, 59]
label_cols = ['DIAGNOSIS', 'DXNORM', 'DXNODEP', 'DXMCI', 'DXMDES', 'DXMPTR1', 'DXMPTR2', 'DXMPTR3', 'DXMPTR4', 'DXMPTR5', 'DXMPTR6', 'DXMDUE', 'DXMOTHET', 'DXDSEV', 'DXAD', 'DXAPP', 'DXAPROB', 'DXAPOSS', 'DXPARK', 'DXPDES', 'DXPCOG', 'DXPATYP', 'DXOTHDEM', 'DXODES', 'DXCONFID', 'BCPREDX', 'BCADAS', 'BCMMSE', 'BCMMSREC', 'BCNMMMS', 'BCNEUPSY', 'BCNONMEM', 'BCFAQ', 'BCCDR', 'BCDEPRES', 'BCSTROKE', 'BCDELIR', 'BCEXTCIR', 'BCCORADL', 'BCCORCOG', 'AXNAUSEA', 'AXVOMIT', 'AXDIARRH', 'AXCONSTP', 'AXABDOMN', 'AXSWEATN', 'AXDIZZY', 'AXENERGY', 'AXDROWSY', 'AXVISION', 'AXHDACHE', 'AXDRYMTH', 'AXBREATH', 'AXCOUGH', 'AXPALPIT', 'AXCHEST', 'AXURNDIS', 'AXURNFRQ', 'AXANKLE', 'AXMUSCLE', 'AXRASH', 'AXINSOMN', 'AXDPMOOD', 'AXCRYING', 'AXELMOOD', 'AXWANDER', 'AXFALL', 'AXOTHER']

SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

config = Config()
cuda_num = 6
device = torch.device(f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)

config.downstream_batch_size = config.downstream_effective_batch_size
batches_per_step = config.downstream_effective_batch_size // config.downstream_batch_size
data_dir = '/shared/bpt3/data/ADNI_PRED/data'
save_dir = '/shared/bpt3/data/ADNI_PRED/save'

input_channels = 3
input_resolution = 192
tot_labels = 68
num_labels = len(labels)
ehr_labels = [i for i in range(tot_labels) if i not in labels and i > 50]
num_ehr = len(ehr_labels)
num_cnn_layers = 4

train_data = pickle.load(open(f'{data_dir}/mri_train.pkl', 'rb')) + pickle.load(open(f'{data_dir}/mri_val.pkl', 'rb'))
test_data = pickle.load(open(f'{data_dir}/mri_test.pkl', 'rb'))

task_train_data = ImageDataset(train_data, config, 'cpu', shape=(input_resolution,input_channels), label_idx=labels)
task_train_loader = DataLoader(task_train_data, batch_size=config.downstream_batch_size, shuffle=True, num_workers=config.num_workers)
task_test_data = ImageDataset(test_data, config, 'cpu', shape=(input_resolution,input_channels), label_idx=labels)
task_test_loader = DataLoader(task_test_data, batch_size=config.downstream_batch_size, shuffle=False, num_workers=config.num_workers)

loss_fn = torch.nn.BCELoss()
train_activation = torch.nn.Sigmoid()
test_activation = torch.nn.Sigmoid()

model = CNNModel(input_channels, input_resolution, num_labels, num_cnn_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.downstream_lr)
for epoch in tqdm(range(25)):
    batches_since_step = 0
    for batch_images, _, batch_labels in tqdm(task_train_loader, desc=f'Tuning Epoch {epoch+1}', leave=False):
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        predictions = model(batch_images)
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
for batch_images, _, batch_labels in tqdm(task_test_loader):
    batch_images = batch_images.to(device)
    batch_labels = batch_labels.to(device)
    with torch.no_grad():
        predictions = model(batch_images)
        predictions = test_activation(predictions)
        task_preds.extend(predictions.cpu().tolist())
        task_labels.extend(batch_labels.cpu().tolist())
        
model2 = LinearModel(num_ehr, num_labels, num_cnn_layers+2).to(device)
optimizer = torch.optim.Adam(model2.parameters(), lr=config.downstream_lr)
for epoch in tqdm(range(25)):
    batches_since_step = 0
    for _, batch_ehr, batch_labels in tqdm(task_train_loader, desc=f'Tuning Epoch {epoch+1}', leave=False):
        batch_ehr = batch_ehr.to(device)
        batch_labels = batch_labels.to(device)
        predictions = model2(batch_ehr)
        predictions = train_activation(predictions)
        loss = loss_fn(predictions, batch_labels)
        loss = loss / batches_per_step
        loss.backward()
        batches_since_step += 1
        if batches_since_step == batches_per_step:
            optimizer.step()
            optimizer.zero_grad()
            batches_since_step = 0

task_preds2 = []
task_labels2 = []
for _, batch_ehr, batch_labels in tqdm(task_test_loader):
    batch_ehr = batch_ehr.to(device)
    batch_labels = batch_labels.to(device)
    with torch.no_grad():
        predictions = model2(batch_ehr)
        predictions = test_activation(predictions)
        task_preds2.extend(predictions.cpu().tolist())
        task_labels2.extend(batch_labels.cpu().tolist())
   
task_preds = np.array(task_preds)
task_labels = np.array(task_labels)
task_rounded_preds = np.round(task_preds)
task_preds2 = np.array(task_preds2)
task_labels2 = np.array(task_labels2)
task_rounded_preds2 = np.round(task_preds2)

overallAcc = metrics.accuracy_score(task_labels.flatten(), task_rounded_preds.flatten())
overallF1 = metrics.f1_score(task_labels.flatten(), task_rounded_preds.flatten())
overallAUROC = metrics.roc_auc_score(task_labels.flatten(), task_preds.flatten())
overallAcc2 = metrics.accuracy_score(task_labels2.flatten(), task_rounded_preds2.flatten())
overallF12 = metrics.f1_score(task_labels2.flatten(), task_rounded_preds2.flatten())
overallAUROC2 = metrics.roc_auc_score(task_labels2.flatten(), task_preds2.flatten())
print(f"IMAGE RESULTS: {overallAcc}, {overallF1}, {overallAUROC}")
print(f"EHR RESULTS: {overallAcc2}, {overallF12}, {overallAUROC2}")
for i in range(len(labels)):
    print(f"IMAGE LABEL {label_cols[labels[i]]}: {metrics.accuracy_score(task_labels[:,i], task_rounded_preds[:,i])}, {metrics.f1_score(task_labels[:,i], task_rounded_preds[:,i])}, {metrics.roc_auc_score(task_labels[:,i], task_preds[:,i])}")
    print(f"EHR LABEL {label_cols[labels[i]]}: {metrics.accuracy_score(task_labels2[:,i], task_rounded_preds2[:,i])}, {metrics.f1_score(task_labels2[:,i], task_rounded_preds2[:,i])}, {metrics.roc_auc_score(task_labels2[:,i], task_preds2[:,i])}")