import torch
import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from src.config import Config
from src.models.univit import UniViT
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from src.baselines.temporal.data.test_dataset import ImageDataset

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

model_key = "univit"

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
cci_threshold1 = np.percentile(cci_values, 50)
cci_map1 = {k: 1 if v > cci_threshold1 else 0 for k, v in cci_map.items()}
cci_threshold2 = np.percentile(cci_values, 75)
cci_map2 = {k: 1 if v > cci_threshold2 else 0 for k, v in cci_map.items()}
cci_threshold3 = np.percentile(cci_values, 90)
cci_map3 = {k: 1 if v > cci_threshold3 else 0 for k, v in cci_map.items()}


temporal_tasks = ['ADNI PET', 'ADNI PET CN', 'ADNI PET AD', 'ADNI PET CCI 1', 'ADNI PET CCI 2', 'ADNI PET CCI 3', 
                'ADNI MRI', 'ADNI MRI CN', 'ADNI MRI AD', 'ADNI MRI CCI 1',  'ADNI MRI CCI 2', 'ADNI MRI CCI 3',
                'ACDC', 'ACDC NORMAL', 'ACDC Abnormality']
                # 'MIMIC-CXR 0', 'MIMIC-CXR 1', 'MIMIC-CXR 2', 'MIMIC-CXR 3', 'MIMIC-CXR 4', 'MIMIC-CXR 5', 'MIMIC-CXR 6',
                # 'CheXpert 0', 'CheXpert 1', 'CheXpert 2', 'CheXpert 3', 'CheXpert 4', 'CheXpert 5', 'CheXpert 6']
# temporal_tasks = ['ADNI PET CN', 'ADNI MRI CN', 'ACDC Abnormality', 'MIMIC-CXR Pneumothorax']
# tune_data['ADNI PET CN'] = [[(v[0], v[1], v[2], v[3], 1 if v[4] > 0 else 0) for v in p] for p in tune_data['ADNI PET']]
# test_data['ADNI PET CN'] = [[(v[0], v[1], v[2], v[3], 1 if v[4] > 0 else 0) for v in p] for p in test_data['ADNI PET']]
# task_map['ADNI PET CN'] = 'Multi-Class Classification'
# tune_data['ADNI MRI CN'] = [[(v[0], v[1], v[2], v[3], 1 if v[4] > 0 else 0) for v in p] for p in tune_data['ADNI MRI']]
# test_data['ADNI MRI CN'] = [[(v[0], v[1], v[2], v[3], 1 if v[4] > 0 else 0) for v in p] for p in test_data['ADNI MRI']]
# task_map['ADNI MRI CN'] = 'Multi-Class Classification'
# tune_data['ACDC Abnormality'] = [[(v[0], v[1], v[2], v[3], 1 if v[4] > 0 else 0) for v in p] for p in tune_data['ACDC']]
# test_data['ACDC Abnormality'] = [[(v[0], v[1], v[2], v[3], 1 if v[4] > 0 else 0) for v in p] for p in test_data['ACDC']]
# task_map['ACDC Abnormality'] = 'Multi-Class Classification'
# tune_data['MIMIC-CXR Pneumothorax'] = [[(v[0], v[1], v[2], v[3], v[4][5]) for v in p] for p in tune_data['MIMIC-CXR']]
# test_data['MIMIC-CXR Pneumothorax'] = [[(v[0], v[1], v[2], v[3], v[4][5]) for v in p] for p in test_data['MIMIC-CXR']]
# task_map['MIMIC-CXR Pneumothorax'] = 'Multi-Class Classification'

# tune_data['DeepLesion Set'] = [[(v[0], v[1], v[2], v[3], [1 if any([p[i][4][j] for i in range(len(p))]) else 0 for j in range(len(p[0][4]))]) for v in p] for p in tune_data['DeepLesion']]
# test_data['DeepLesion Set'] = [[(v[0], v[1], v[2], v[3], [1 if any([p[i][4][j] for i in range(len(p))]) else 0 for j in range(len(p[0][4]))]) for v in p] for p in test_data['DeepLesion']]
# task_map['DeepLesion Set'] = 'Multi-Label Classification'
# tune_data.pop('DeepLesion')
# test_data.pop('DeepLesion')
tune_data['ADNI PET CN'] = [[(v[0], v[1], v[2], v[3], 1 if v[4] > 0 else 0) for v in p] for p in tune_data['ADNI PET']]
test_data['ADNI PET CN'] = [[(v[0], v[1], v[2], v[3], 1 if v[4] > 0 else 0) for v in p] for p in test_data['ADNI PET']]
task_map['ADNI PET CN'] = 'Multi-Class Classification'
tune_data['ADNI PET AD'] = [[(v[0], v[1], v[2], v[3], 1 if v[4] == 2 else 0) for v in p] for p in tune_data['ADNI PET']]
test_data['ADNI PET AD'] = [[(v[0], v[1], v[2], v[3], 1 if v[4] == 2 else 0) for v in p] for p in test_data['ADNI PET']]
task_map['ADNI PET AD'] = 'Multi-Class Classification'
tune_data['ADNI PET CCI 1'] = [[(v[0], v[1], v[2], v[3], cci_map1[v[0].split('/')[-1].split('--')[0]]) for v in p] for p in tune_data['ADNI PET'] if p[0][0].split('/')[-1].split('--')[0] in cci_map1]
test_data['ADNI PET CCI 1'] = [[(v[0], v[1], v[2], v[3], cci_map1[v[0].split('/')[-1].split('--')[0]]) for v in p] for p in test_data['ADNI PET'] if p[0][0].split('/')[-1].split('--')[0] in cci_map1]
task_map['ADNI PET CCI 1'] = 'Multi-Class Classification'
tune_data['ADNI PET CCI 2'] = [[(v[0], v[1], v[2], v[3], cci_map2[v[0].split('/')[-1].split('--')[0]]) for v in p] for p in tune_data['ADNI PET'] if p[0][0].split('/')[-1].split('--')[0] in cci_map2]
test_data['ADNI PET CCI 2'] = [[(v[0], v[1], v[2], v[3], cci_map2[v[0].split('/')[-1].split('--')[0]]) for v in p] for p in test_data['ADNI PET'] if p[0][0].split('/')[-1].split('--')[0] in cci_map2]
task_map['ADNI PET CCI 2'] = 'Multi-Class Classification'
tune_data['ADNI PET CCI 3'] = [[(v[0], v[1], v[2], v[3], cci_map3[v[0].split('/')[-1].split('--')[0]]) for v in p] for p in tune_data['ADNI PET'] if p[0][0].split('/')[-1].split('--')[0] in cci_map3]
test_data['ADNI PET CCI 3'] = [[(v[0], v[1], v[2], v[3], cci_map3[v[0].split('/')[-1].split('--')[0]]) for v in p] for p in test_data['ADNI PET'] if p[0][0].split('/')[-1].split('--')[0] in cci_map3]
task_map['ADNI PET CCI 3'] = 'Multi-Class Classification'
tune_data['ADNI MRI CN'] = [[(v[0], v[1], v[2], v[3], 1 if v[4] > 0 else 0) for v in p] for p in tune_data['ADNI MRI']]
test_data['ADNI MRI CN'] = [[(v[0], v[1], v[2], v[3], 1 if v[4] > 0 else 0) for v in p] for p in test_data['ADNI MRI']]
task_map['ADNI MRI CN'] = 'Multi-Class Classification'
tune_data['ADNI MRI AD'] = [[(v[0], v[1], v[2], v[3], 1 if v[4] == 2 else 0) for v in p] for p in tune_data['ADNI MRI']]
test_data['ADNI MRI AD'] = [[(v[0], v[1], v[2], v[3], 1 if v[4] == 2 else 0) for v in p] for p in test_data['ADNI MRI']]
task_map['ADNI MRI AD'] = 'Multi-Class Classification'
tune_data['ADNI MRI CCI 1'] = [[(v[0], v[1], v[2], v[3], cci_map1[v[0].split('/')[-1].split('--')[0]]) for v in p] for p in tune_data['ADNI MRI'] if p[0][0].split('/')[-1].split('--')[0] in cci_map1]
test_data['ADNI MRI CCI 1'] = [[(v[0], v[1], v[2], v[3], cci_map1[v[0].split('/')[-1].split('--')[0]]) for v in p] for p in test_data['ADNI MRI'] if p[0][0].split('/')[-1].split('--')[0] in cci_map1]
task_map['ADNI MRI CCI 1'] = 'Multi-Class Classification'
tune_data['ADNI MRI CCI 2'] = [[(v[0], v[1], v[2], v[3], cci_map2[v[0].split('/')[-1].split('--')[0]]) for v in p] for p in tune_data['ADNI MRI'] if p[0][0].split('/')[-1].split('--')[0] in cci_map2]
test_data['ADNI MRI CCI 2'] = [[(v[0], v[1], v[2], v[3], cci_map2[v[0].split('/')[-1].split('--')[0]]) for v in p] for p in test_data['ADNI MRI'] if p[0][0].split('/')[-1].split('--')[0] in cci_map2]
task_map['ADNI MRI CCI 2'] = 'Multi-Class Classification'
tune_data['ADNI MRI CCI 3'] = [[(v[0], v[1], v[2], v[3], cci_map3[v[0].split('/')[-1].split('--')[0]]) for v in p] for p in tune_data['ADNI MRI'] if p[0][0].split('/')[-1].split('--')[0] in cci_map3]
test_data['ADNI MRI CCI 3'] = [[(v[0], v[1], v[2], v[3], cci_map3[v[0].split('/')[-1].split('--')[0]]) for v in p] for p in test_data['ADNI MRI'] if p[0][0].split('/')[-1].split('--')[0] in cci_map3]
task_map['ADNI MRI CCI 3'] = 'Multi-Class Classification'
# tune_data.pop('ADNI PET')
# test_data.pop('ADNI PET')
# tune_data.pop('ADNI MRI')
# test_data.pop('ADNI MRI')
tune_data['ACDC NORMAL'] = [[(v[0], v[1], v[2], v[3], 1 if v[4] > 0 else 0) for v in p] for p in tune_data['ACDC']]
test_data['ACDC NORMAL'] = [[(v[0], v[1], v[2], v[3], 1 if v[4] > 0 else 0) for v in p] for p in test_data['ACDC']]
task_map['ACDC NORMAL'] = 'Multi-Class Classification'
tune_data['ACDC Multi-Label'] = [[(v[0], v[1], v[2], v[3], [1 if i == v[4] else 0 for i in range(0,5)]) for v in p] for p in tune_data['ACDC']]
test_data['ACDC Multi-Label'] = [[(v[0], v[1], v[2], v[3], [1 if i == v[4] else 0 for i in range(0,5)]) for v in p] for p in test_data['ACDC']]
task_map['ACDC Multi-Label'] = 'Multi-Label Classification'
# tune_data.pop('ACDC')
# test_data.pop('ACDC')
# tune_data['MIMIC-CXR Set'] = [[(v[0], v[1], v[2], v[3], [1 if any([p[i][4][j] for i in range(len(p))]) else 0 for j in range(len(p[0][4]))]) for v in p] for p in tune_data['MIMIC-CXR']]
# test_data['MIMIC-CXR Set'] = [[(v[0], v[1], v[2], v[3], [1 if any([p[i][4][j] for i in range(len(p))]) else 0 for j in range(len(p[0][4]))]) for v in p] for p in test_data['MIMIC-CXR']]
# task_map['MIMIC-CXR Set'] = 'Multi-Label Classification'
# tune_data['MIMIC-CXR Positive Change'] = [[(p[i][0], p[i][1], p[i][2], p[i][3], p[i][4] if i == 0 else [1 if p[i][4][j] > p[i-1][4][j] else 0 for j in range(len(p[i][4]))]) for i in range(len(p))] for p in tune_data['MIMIC-CXR']]
# test_data['MIMIC-CXR Positive Change'] = [[(p[i][0], p[i][1], p[i][2], p[i][3], p[i][4] if i == 0 else [1 if p[i][4][j] > p[i-1][4][j] else 0 for j in range(len(p[i][4]))]) for i in range(len(p))] for p in test_data['MIMIC-CXR']]
# task_map['MIMIC-CXR Positive Change'] = 'Multi-Label Classification'
# tune_data['MIMIC-CXR Any Change'] = [[(p[i][0], p[i][1], p[i][2], p[i][3], p[i][4] if i == 0 else [1 if p[i][4][j] != p[i-1][4][j] else 0 for j in range(len(p[i][4]))]) for i in range(len(p))] for p in tune_data['MIMIC-CXR']]
# test_data['MIMIC-CXR Any Change'] = [[(p[i][0], p[i][1], p[i][2], p[i][3], p[i][4] if i == 0 else [1 if p[i][4][j] != p[i-1][4][j] else 0 for j in range(len(p[i][4]))]) for i in range(len(p))] for p in test_data['MIMIC-CXR']]
# task_map['MIMIC-CXR Any Change'] = 'Multi-Label Classification'
# tune_data['MIMIC-CXR Paired Set'] = [[(p[i][0], p[i][1], p[i][2], p[i][3], p[i][4] if i == 0 else [1 if p[i][4][j] or p[i-1][4][j] else 0 for j in range(len(p[i][4]))]) for i in range(len(p))] for p in tune_data['MIMIC-CXR']]
# test_data['MIMIC-CXR Paired Set'] = [[(p[i][0], p[i][1], p[i][2], p[i][3], p[i][4] if i == 0 else [1 if p[i][4][j] or p[i-1][4][j] else 0 for j in range(len(p[i][4]))]) for i in range(len(p))] for p in test_data['MIMIC-CXR']]
# task_map['MIMIC-CXR Paired Set'] = 'Multi-Label Classification'
tune_data['MIMIC-CXR 0'] = [[(v[0], v[1], v[2], v[3], v[4][0]) for v in p] for p in tune_data['MIMIC-CXR']]
test_data['MIMIC-CXR 0'] = [[(v[0], v[1], v[2], v[3], v[4][0]) for v in p] for p in test_data['MIMIC-CXR']]
task_map['MIMIC-CXR 0'] = 'Multi-Class Classification'
tune_data['MIMIC-CXR 1'] = [[(v[0], v[1], v[2], v[3], v[4][1]) for v in p] for p in tune_data['MIMIC-CXR']]
test_data['MIMIC-CXR 1'] = [[(v[0], v[1], v[2], v[3], v[4][1]) for v in p] for p in test_data['MIMIC-CXR']]
task_map['MIMIC-CXR 1'] = 'Multi-Class Classification'
tune_data['MIMIC-CXR 2'] = [[(v[0], v[1], v[2], v[3], v[4][2]) for v in p] for p in tune_data['MIMIC-CXR']]
test_data['MIMIC-CXR 2'] = [[(v[0], v[1], v[2], v[3], v[4][2]) for v in p] for p in test_data['MIMIC-CXR']]
task_map['MIMIC-CXR 2'] = 'Multi-Class Classification'
tune_data['MIMIC-CXR 3'] = [[(v[0], v[1], v[2], v[3], v[4][3]) for v in p] for p in tune_data['MIMIC-CXR']]
test_data['MIMIC-CXR 3'] = [[(v[0], v[1], v[2], v[3], v[4][3]) for v in p] for p in test_data['MIMIC-CXR']]
task_map['MIMIC-CXR 3'] = 'Multi-Class Classification'
tune_data['MIMIC-CXR 4'] = [[(v[0], v[1], v[2], v[3], v[4][4]) for v in p] for p in tune_data['MIMIC-CXR']]
test_data['MIMIC-CXR 4'] = [[(v[0], v[1], v[2], v[3], v[4][4]) for v in p] for p in test_data['MIMIC-CXR']]
task_map['MIMIC-CXR 4'] = 'Multi-Class Classification'
tune_data['MIMIC-CXR 5'] = [[(v[0], v[1], v[2], v[3], v[4][5]) for v in p] for p in tune_data['MIMIC-CXR']]
test_data['MIMIC-CXR 5'] = [[(v[0], v[1], v[2], v[3], v[4][5]) for v in p] for p in test_data['MIMIC-CXR']]
task_map['MIMIC-CXR 5'] = 'Multi-Class Classification'
tune_data['MIMIC-CXR 6'] = [[(v[0], v[1], v[2], v[3], v[4][6]) for v in p] for p in tune_data['MIMIC-CXR']]
test_data['MIMIC-CXR 6'] = [[(v[0], v[1], v[2], v[3], v[4][6]) for v in p] for p in test_data['MIMIC-CXR']]
task_map['MIMIC-CXR 6'] = 'Multi-Class Classification'
# tune_data.pop('MIMIC-CXR')
# test_data.pop('MIMIC-CXR')
# tune_data['CheXpert Set'] = [[(v[0], v[1], v[2], v[3], [1 if any([p[i][4][j] for i in range(len(p))]) else 0 for j in range(len(p[0][4]))]) for v in p] for p in tune_data['CheXpert']]
# test_data['CheXpert Set'] = [[(v[0], v[1], v[2], v[3], [1 if any([p[i][4][j] for i in range(len(p))]) else 0 for j in range(len(p[0][4]))]) for v in p] for p in test_data['CheXpert']]
# task_map['CheXpert Set'] = 'Multi-Label Classification'
# tune_data['CheXpert Positive Change'] = [[(p[i][0], p[i][1], p[i][2], p[i][3], p[i][4] if i == 0 else [1 if p[i][4][j] > p[i-1][4][j] else 0 for j in range(len(p[i][4]))]) for i in range(len(p))] for p in tune_data['CheXpert']]
# test_data['CheXpert Positive Change'] = [[(p[i][0], p[i][1], p[i][2], p[i][3], p[i][4] if i == 0 else [1 if p[i][4][j] > p[i-1][4][j] else 0 for j in range(len(p[i][4]))]) for i in range(len(p))] for p in test_data['CheXpert']]
# task_map['CheXpert Positive Change'] = 'Multi-Label Classification'
# tune_data['CheXpert Any Change'] = [[(p[i][0], p[i][1], p[i][2], p[i][3], p[i][4] if i == 0 else [1 if p[i][4][j] != p[i-1][4][j] else 0 for j in range(len(p[i][4]))]) for i in range(len(p))] for p in tune_data['CheXpert']]
# test_data['CheXpert Any Change'] = [[(p[i][0], p[i][1], p[i][2], p[i][3], p[i][4] if i == 0 else [1 if p[i][4][j] != p[i-1][4][j] else 0 for j in range(len(p[i][4]))]) for i in range(len(p))] for p in test_data['CheXpert']]
# task_map['CheXpert Any Change'] = 'Multi-Label Classification'
# tune_data['CheXpert Paired Set'] = [[(p[i][0], p[i][1], p[i][2], p[i][3], p[i][4] if i == 0 else [1 if p[i][4][j] or p[i-1][4][j] else 0 for j in range(len(p[i][4]))]) for i in range(len(p))] for p in tune_data['CheXpert']]
# test_data['CheXpert Paired Set'] = [[(p[i][0], p[i][1], p[i][2], p[i][3], p[i][4] if i == 0 else [1 if p[i][4][j] or p[i-1][4][j] else 0 for j in range(len(p[i][4]))]) for i in range(len(p))] for p in test_data['CheXpert']]
# task_map['CheXpert Paired Set'] = 'Multi-Label Classification'
tune_data['CheXpert 0'] = [[(v[0], v[1], v[2], v[3], int(v[4][0])) for v in p] for p in tune_data['CheXpert']]
test_data['CheXpert 0'] = [[(v[0], v[1], v[2], v[3], int(v[4][0])) for v in p] for p in test_data['CheXpert']]
task_map['CheXpert 0'] = 'Multi-Class Classification'
tune_data['CheXpert 1'] = [[(v[0], v[1], v[2], v[3], int(v[4][1])) for v in p] for p in tune_data['CheXpert']]
test_data['CheXpert 1'] = [[(v[0], v[1], v[2], v[3], int(v[4][1])) for v in p] for p in test_data['CheXpert']]
task_map['CheXpert 1'] = 'Multi-Class Classification'
tune_data['CheXpert 2'] = [[(v[0], v[1], v[2], v[3], int(v[4][2])) for v in p] for p in tune_data['CheXpert']]
test_data['CheXpert 2'] = [[(v[0], v[1], v[2], v[3], int(v[4][2])) for v in p] for p in test_data['CheXpert']]
task_map['CheXpert 2'] = 'Multi-Class Classification'
tune_data['CheXpert 3'] = [[(v[0], v[1], v[2], v[3], int(v[4][3])) for v in p] for p in tune_data['CheXpert']]
test_data['CheXpert 3'] = [[(v[0], v[1], v[2], v[3], int(v[4][3])) for v in p] for p in test_data['CheXpert']]
task_map['CheXpert 3'] = 'Multi-Class Classification'
tune_data['CheXpert 4'] = [[(v[0], v[1], v[2], v[3], int(v[4][4])) for v in p] for p in tune_data['CheXpert']]
test_data['CheXpert 4'] = [[(v[0], v[1], v[2], v[3], int(v[4][4])) for v in p] for p in test_data['CheXpert']]
task_map['CheXpert 4'] = 'Multi-Class Classification'
tune_data['CheXpert 5'] = [[(v[0], v[1], v[2], v[3], int(v[4][5])) for v in p] for p in tune_data['CheXpert']]
test_data['CheXpert 5'] = [[(v[0], v[1], v[2], v[3], int(v[4][5])) for v in p] for p in test_data['CheXpert']]
task_map['CheXpert 5'] = 'Multi-Class Classification'
tune_data['CheXpert 6'] = [[(v[0], v[1], v[2], v[3], int(v[4][6])) for v in p] for p in tune_data['CheXpert']]
test_data['CheXpert 6'] = [[(v[0], v[1], v[2], v[3], int(v[4][6])) for v in p] for p in test_data['CheXpert']]
task_map['CheXpert 6'] = 'Multi-Class Classification'
# tune_data.pop('CheXpert')
# test_data.pop('CheXpert')


model = UniViT(
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
    config.patch_prob,
    extra_cls=True,
).to(device)
print("Loading previous model")
model.load_state_dict(
    torch.load(f"{save_dir}/{model_key}.pt", map_location="cpu")["model"]
)
model.eval()
model.requires_grad_(False)

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
        print(f"Task {task}")
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
    X_temporal = []
    X_static = []
    y = []
    for batch_images_temporal, batch_dimensions_temporal, batch_images_static, batch_dimensions_static, batch_labels in tqdm(task_loader, desc=f"{task} Tuning", leave=False):
        batch_images_temporal = batch_images_temporal.to(device)
        batch_images_static = batch_images_static.to(device)
        with torch.no_grad():
            X_temporal.extend(model.embed(batch_images_temporal, batch_dimensions_temporal).cpu().tolist())
            X_static.extend(model.embed(batch_images_static, batch_dimensions_static).cpu().tolist())
            y.extend(batch_labels.cpu().tolist())
            
    temporalResults =  {"Accuracy": [], "F1": [], "AUROC": []}
    staticResults =  {"Accuracy": [], "F1": [], "AUROC": []}
    totData = len(y)
    foldSize = totData // config.downstream_folds
    for fold in tqdm(range(config.downstream_folds), desc=f"Evaluating {task} Folds", leave=False):
        X_temporal_train = np.array(X_temporal[:fold*foldSize] + X_temporal[(fold+1)*foldSize:])
        X_static_train = np.array(X_static[:fold*foldSize] + X_static[(fold+1)*foldSize:])
        y_train = np.array(y[:fold*foldSize] + y[(fold+1)*foldSize:])
        X_temporal_test = np.array(X_temporal[fold*foldSize:(fold+1)*foldSize])
        X_static_test = np.array(X_static[fold*foldSize:(fold+1)*foldSize])
        y_test = np.array(y[fold*foldSize:(fold+1)*foldSize])
        if len(set(y_test.flatten())) == 1:
            continue
        
        downstream_temporal = LogisticRegression(max_iter=10000)
        downstream_static = LogisticRegression(max_iter=10000)
        if taskType == "Multi-Label Classification":
            downstream_temporal = MultiOutputClassifier(downstream_temporal)
            downstream_static = MultiOutputClassifier(downstream_static)

        downstream_temporal.fit(X_temporal_train, y_train)
        downstream_static.fit(X_static_train, y_train)
        task_probs_temporal = downstream_temporal.predict_proba(X_temporal_test)
        task_probs_static = downstream_static.predict_proba(X_static_test)
        if taskType == "Multi-Label Classification":
            task_probs_temporal = np.array(task_probs_temporal).transpose(1, 0, 2)[:,:,1]
            task_probs_static = np.array(task_probs_static).transpose(1, 0, 2)[:,:,1]
        elif label_size == 2:
            task_probs_temporal = task_probs_temporal[:,1]
            task_probs_static = task_probs_static[:,1]
            
        if taskType == "Multi-Label Classification" or label_size == 2:
            task_preds_temporal = np.round(task_probs_temporal)
            acc_temporal = metrics.accuracy_score(y_test.flatten(), task_preds_temporal.flatten())
            f1_temporal = metrics.f1_score(y_test.flatten(), task_preds_temporal.flatten())
            auroc_temporal = metrics.roc_auc_score(y_test.flatten(), task_probs_temporal.flatten())
            task_preds_static = np.round(task_probs_static)
            acc_static = metrics.accuracy_score(y_test.flatten(), task_preds_static.flatten())
            f1_static = metrics.f1_score(y_test.flatten(), task_preds_static.flatten())
            auroc_static = metrics.roc_auc_score(y_test.flatten(), task_probs_static.flatten())
        elif taskType == "Multi-Class Classification":
            task_preds_temporal = np.argmax(task_probs_temporal, axis=1)
            task_preds_static = np.argmax(task_probs_static, axis=1)
            acc_temporal = metrics.accuracy_score(y_test, task_preds_temporal)
            f1_temporal = metrics.f1_score(y_test, task_preds_temporal, average="macro")
            acc_static = metrics.accuracy_score(y_test, task_preds_static)
            f1_static = metrics.f1_score(y_test, task_preds_static, average="macro")
            present_classes = np.unique(y_test)
            if len(present_classes) != label_size:
                task_probs_temporal = task_probs_temporal[:, present_classes]
                task_probs_temporal = task_probs_temporal / task_probs_temporal.sum(axis=1, keepdims=True)
                task_probs_static = task_probs_static[:, present_classes]
                task_probs_static = task_probs_static / task_probs_static.sum(axis=1, keepdims=True)
                label_mapping = {c: i for i, c in enumerate(present_classes)}
                y_test = np.array([label_mapping[c] for c in y_test])
            auroc_temporal = metrics.roc_auc_score(y_test, task_probs_temporal, average="macro", multi_class="ovr")
            auroc_static = metrics.roc_auc_score(y_test, task_probs_static, average="macro", multi_class="ovr")

        temporalResults["Accuracy"].append(acc_temporal)
        temporalResults["F1"].append(f1_temporal)
        temporalResults["AUROC"].append(auroc_temporal)
        staticResults["Accuracy"].append(acc_static)
        staticResults["F1"].append(f1_static)
        staticResults["AUROC"].append(auroc_static)
    taskResults_temporal = {k: np.mean(v) for k, v in temporalResults.items()}
    taskResults_static = {k: np.mean(v) for k, v in staticResults.items()}
    print('\tTEMPORAL:', taskResults_temporal)
    print('\tSTATIC:', taskResults_static)
