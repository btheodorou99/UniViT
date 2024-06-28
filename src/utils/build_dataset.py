from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import pandas as pd
import pickle
import os

# Set the output directory
data_dir = '/shared/bpt3/data/UniViT/data/'
os.makedirs(data_dir, exist_ok=True)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

# Initialize the dataset
dataset = {}

# Set the different directories
mimic_cxr_dir = '/srv/local/data/MIMIC-CXR/'
chexpert_dir = '/srv/local/data/CheXpert/'
isic_dir = '/srv/local/data/ISIC_SkinLesion/'
adni_dir = '/srv/local/data/ADNI/'
deep_lesion_dir = '/srv/local/data/DeepLesion/'
covid_cxr_dir = '/srv/local/data/COVID-QU-Ex/'
crc_he_dir1 = '/srv/local/data/NCT-CRC-HE-100K/'
crc_he_dir2 = '/srv/local/data/CRC-VAL-HE-7K/'

# Do some label processing
chestXrayLabelIdx = [0, 1, 3, 7, 8, 12, 13]

# Process the MIMIC-CXR dataset
df = pd.read_csv(mimic_cxr_dir + 'cxr-record-list.csv').merge(pd.read_csv(mimic_cxr_dir + 'mimic-cxr-report.csv'), on=['subject_id', 'study_id']).merge(pd.read_csv(mimic_cxr_dir + 'mimic-cxr-2.0.0-metadata.csv'), on=['subject_id', 'study_id', 'dicom_id']).merge(pd.read_csv(mimic_cxr_dir + 'mimic-cxr-2.0.0-chexpert.csv'), on=['subject_id', 'study_id'])
df = df.sort_values(by=['subject_id', 'study_id', 'StudyDate', 'StudyTime'], ascending=[True, True, True, True])[['subject_id', 'study_id', 'dicom_id', 'StudyDate', 'StudyTime', 'report', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia','Pneumothorax', 'Support Devices']]
df['report'] = [[t - 100 for t in tokenizer.encode(x)] for x in df['report']] # 100 is [UNK], 101 is [CLS], 102 is [SEP]
def labelMap(val):
    if val == 1:
        return 1
    else:
        return 0

for row in tqdm(df.itertuples(), total=len(df), desc='Processing MIMIC-CXR'):
    subject_id = row.subject_id
    study_id = row.study_id
    dicom_id = row.dicom_id
    path = f'{mimic_cxr_dir}resized/{dicom_id}.jpg'
    note = row.report
    labels = [row.Atelectasis, row.Cardiomegaly, row.Consolidation, row.Edema, row._11, row.Fracture, row._13, row._14, row._15, row._16, row._17, row.Pneumonia, row.Pneumothorax, row._20]
    labels = [labelMap(l) for i, l in enumerate(labels) if i in chestXrayLabelIdx]
    dimensions = (1, 256, 256)
    modality = 'Chest X-Ray (MIMIC)'
    data_point = (path, dimensions, note, modality, labels)
    if subject_id not in dataset:
        dataset[subject_id] = [data_point]
    else:
        dataset[subject_id].append(data_point)

# Process the CheXpert dataset
df = pd.read_csv(chexpert_dir + 'chexpert-meta.csv')
df = df[['imgpath', 'Sex', 'Age' ,'Frontal/Lateral', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']]
attr_map = {'Atelectasis': 'Atelectasis', 'Cardiomegaly': 'Cardiomegaly', 'Consolidation': 'Consolidation', 'Edema': 'Edema', '_9': 'Enlarged Cardiomediastinum', 'Fracture': 'Fracture', '_11': 'Lung Lesion', '_12': 'Lung Opacity', '_13': 'No Finding', '_14': 'Pleural Effusion', '_15': 'Pleural Other', 'Pneumonia': 'Pneumonia', 'Pneumothorax': 'Pneumothorax', '_18': 'Support Devices'}
for row in tqdm(df.itertuples(), total=len(df), desc='Processing CheXpert'):
    imgpath = row.imgpath
    subject_id = imgpath.split('/study')[0].split('/')[-1]
    path = f'{chexpert_dir}{imgpath.split("./")[1]}'
    findings = [attr_map[c] for c in row._fields if c != 'Index' and getattr(row, c) == 1]
    note = f'{row._4} Chest X-Ray of a {row.Age} year old {row.Sex}'
    if len(findings) > 0:
        if len(findings) == 1:
            note += f' with {findings[0]}'
        elif len(findings) == 2:
            note += f' with {findings[0]} and {findings[1]}'
        else:
            note += f' with {", ".join(findings[:-1])}, and {findings[-1]}'   
    note = [t - 100 for t in tokenizer.encode(note)]
    labels = [row.Atelectasis, row.Cardiomegaly, row.Consolidation, row.Edema, row._9, row.Fracture, row._11, row._12, row._13, row._14, row._15, row.Pneumonia, row.Pneumothorax, row._18]
    labels = [l for i, l in enumerate(labels) if i in chestXrayLabelIdx]
    dimensions = Image.open(path).size
    dimensions = (1, dimensions[1], dimensions[0])
    modality = 'Chest X-Ray (CheXpert)'
    data_point = (path, dimensions, note, modality, labels)
    if subject_id not in dataset:
        dataset[subject_id] = [data_point]
    else:
        dataset[subject_id].append(data_point)

# Process the skin lesion dataset
df = pd.read_csv(isic_dir + 'ISIC_2019_Training_GroundTruth.csv').merge(pd.read_csv(isic_dir + 'ISIC_2019_Training_Metadata.csv'), on=['image'])
condMap = {'MEL': 'Melanoma', 'NV': 'Melanocytic nevus', 'BCC': 'Basal cell carcinoma', 'AK': 'Actinic keratosis', 'BKL': 'Benign keratosis', 'DF': 'Dermatofibroma', 'VASC': 'Vascular lesion', 'SCC': 'Squamous cell carcinoma'}
for row in tqdm(df.itertuples(), total=len(df), desc='Processing ISIC'):
    imgpath = row.image
    subject_id = imgpath.split('_')[1]
    path = f'{isic_dir}ISIC_2019_Data/{imgpath}.jpg'
    findings = [condMap[c] for c in row._fields if getattr(row, c) == 1 and c in condMap]
    note = f'Skin lesion image from the {row.anatom_site_general} of a {row.age_approx} year old {row.sex}'
    if len(findings) > 0:
        if len(findings) == 1:
            note += f' showing {findings[0]}'
        elif len(findings) == 2:
            note += f' showing {findings[0]} and {findings[1]}'
        else:
            note += f' showing {", ".join(findings[:-1])}, and {findings[-1]}'
    note = [t - 100 for t in tokenizer.encode(note)]
    labels = [getattr(row, c) for c in row._fields if c in condMap]
    dimensions = Image.open(path).size
    dimensions = (1, dimensions[1], dimensions[0])
    modality = 'Skin Lesion'
    data_point = (path, dimensions, note, modality, labels)
    if subject_id not in dataset:
        dataset[subject_id] = [data_point]
    else:
        dataset[subject_id].append(data_point)

# Load the ADNI Labels
adni_labels = pd.read_csv(adni_dir + 'DXSUM_PDXCONV.csv')
adni_labels = {row.PTID: row.DIAGNOSIS for row in adni_labels.itertuples()}
adni_labels = {p: int(l - 1) if l == l else 0 for p, l in adni_labels.items()}
# LABELS: CN=0, MCI=1, AD=2

# Process the ADNI MRI dataset
data = pickle.load(open(adni_dir + 'mri_data.pkl', 'rb'))
for subject_id in tqdm(data, desc='Processing ADNI MRI'):
    for date in sorted(data[subject_id].keys()):
        path = f'{adni_dir}MRI/{data[subject_id][date]["filename"]}'
        dimensions = data[subject_id][date]['shape']
        dimensions = (dimensions[2], dimensions[0], dimensions[1])
        note = [t - 100 for t in tokenizer.encode(f'T1-weighted MRI of a brain')]
        labels = adni_labels[subject_id] if subject_id in adni_labels else None
        modality = 'MRI'
        if subject_id not in dataset:
            dataset[subject_id] = [(path, dimensions, note, modality, labels)]
        else:
            dataset[subject_id].append((path, dimensions, note, modality, labels))
            
# Process the ADNI Amyloid PET dataset
data = pickle.load(open(adni_dir + 'av45_pet_data.pkl', 'rb'))
for subject_id in tqdm(data, desc='Processing ADNI AV45 PET'):
    for date in sorted(data[subject_id].keys()):
        path = f'{adni_dir}AV45_PET/{data[subject_id][date]["filename"]}'
        dimensions = data[subject_id][date]['shape']
        dimensions = (dimensions[2], dimensions[0], dimensions[1])
        note = [t - 100 for t in tokenizer.encode(f'Amyloid PET scan of a brain')]
        labels = adni_labels[subject_id] if subject_id in adni_labels else None
        modality = 'Amyloid PET'
        if subject_id not in dataset:
            dataset[subject_id] = [(path, dimensions, note, modality, labels)]
        else:
            dataset[subject_id].append((path, dimensions, note, modality, labels))

# Process the ADNI FDG PET dataset
data = pickle.load(open(adni_dir + 'fdg_pet_data.pkl', 'rb'))
for subject_id in tqdm(data, desc='Processing ADNI FDG PET'):
    for date in sorted(data[subject_id].keys()):
        path = f'{adni_dir}FDG_PET/{data[subject_id][date]["filename"]}'
        dimensions = data[subject_id][date]['shape']
        dimensions = (dimensions[2], dimensions[0], dimensions[1])
        note = [t - 100 for t in tokenizer.encode(f'FDG PET scan of a brain')]
        labels = adni_labels[subject_id] if subject_id in adni_labels else None
        modality = 'FDG PET'
        if subject_id not in dataset:
            dataset[subject_id] = [(path, dimensions, note, modality, labels)]
        else:
            dataset[subject_id].append((path, dimensions, note, modality, labels))

# Process the DeepLesion dataset
data = pd.read_csv(deep_lesion_dir + 'DL_info.csv')
for row in tqdm(data.itertuples(), total=len(data), desc='Processing DeepLesion'):
    path = f"{deep_lesion_dir}{'_'.join(row.File_name.split('_')[:-1])}.npy"
    slices = row.Slice_range.split(', ')
    image_size = row.Image_size.split(', ')
    dimensions = (int(slices[1]) - int(slices[0]) + 1, int(image_size[0]), int(image_size[1]))
    note = [t - 100 for t in tokenizer.encode(f'CT scan of a {row.Patient_age} year old {"male" if row.Patient_gender == "M" else "female"} patient with a lesion')]
    labels = row.Coarse_lesion_type if row.Coarse_lesion_type != -1 else None
    subject_id = f'DL_{row.Patient_index}'
    modality = 'CT'
    if subject_id not in dataset:
        dataset[subject_id] = [(path, dimensions, note, modality, labels)]
    else:
        dataset[subject_id].append((path, dimensions, note, modality, labels))
        
# Process the COVID-QU-Ex dataset
for top_dir in os.listdir(covid_cxr_dir):
    if os.path.isdir(f'{covid_cxr_dir}{top_dir}'):
        for split in os.listdir(f'{covid_cxr_dir}{top_dir}/{top_dir}/'):
            for cls in os.listdir(f'{covid_cxr_dir}{top_dir}/{top_dir}/{split}/'):
                for img in os.listdir(f'{covid_cxr_dir}{top_dir}/{top_dir}/{split}/{cls}/images/'):
                    path = f'{covid_cxr_dir}{top_dir}/{top_dir}/{split}/{cls}/images/{img}'
                    dimensions = Image.open(path).size
                    dimensions = (1, dimensions[1], dimensions[0])
                    note = [t - 100 for t in tokenizer.encode(f'Chest X-Ray of a {cls} patient')]
                    labels = 0 if cls == 'normal' else 1 if cls == 'pneumonia' else 2
                    subject_id = path.split('/')[-1].split('_')[0] if 'sub-' in path else path 
                    modality = 'Chest X-Ray (COVID-QU-Ex)'
                    if subject_id not in dataset:
                        dataset[subject_id] = [(path, dimensions, note, modality, labels)]
                    else:
                        dataset[subject_id].append((path, dimensions, note, modality, labels))

# Process the CRC-HE dataset
clsMap = {'ADI': 'Adipose', 'BACK': 'Background', 'DEB': 'Debris', 'LYM': 'Lymphocytes', 'MUC': 'Mucus', 'MUS': 'Smooth Muscle', 'NORM': 'Normal Colon Mucosa', 'STR': 'Cancer-Associated Stroma', 'TUM': 'Colorectal Adenocarcinoma Epithelium'}
labelIdx = {'ADI': 0, 'BACK': 1, 'DEB': 2, 'LYM': 3, 'MUC': 4, 'MUS': 5, 'NORM': 6, 'STR': 7, 'TUM': 8}
for cls in os.listdir(crc_he_dir1):
    for img in os.listdir(f'{crc_he_dir1}{cls}/'):
        path = f'{crc_he_dir1}{cls}/{img}'
        dimensions = Image.open(path).size
        dimensions = (1, dimensions[1], dimensions[0])
        note = [t - 100 for t in tokenizer.encode(f'Histopathology image of a {clsMap[cls]}')]
        labels = labelIdx[cls]
        subject_id = path.split('/')[-1].split('-')[-1].split('.')[0]
        modality = 'Histopathology'
        if subject_id not in dataset:
            dataset[subject_id] = [(path, dimensions, note, modality, labels)]
        else:
            dataset[subject_id].append((path, dimensions, note, modality, labels))
            
for cls in os.listdir(crc_he_dir2):
    for img in os.listdir(f'{crc_he_dir2}{cls}/'):
        path = f'{crc_he_dir2}{cls}/{img}'
        dimensions = Image.open(path).size
        dimensions = (1, dimensions[1], dimensions[0])
        note = [t - 100 for t in tokenizer.encode(f'Histopathology image of a {clsMap[cls]}')]
        labels = labelIdx[cls]
        subject_id = path.split('/')[-1].split('-')[-1].split('.')[0]
        modality = 'Histopathology'
        if subject_id not in dataset:
            dataset[subject_id] = [(path, dimensions, note, modality, labels)]
        else:
            dataset[subject_id].append((path, dimensions, note, modality, labels))

# Split the dataset
dataset = list(dataset.values())
train, test = train_test_split(dataset, test_size=0.2, random_state=4)
tune, test = train_test_split(test, test_size=0.5, random_state=4)
visualization = tune + test
tune_temporal = [p for p in tune if len(p) >= 3]
test_temporal = [p for p in test if len(p) >= 3]
tune = [v for p in tune for v in p]
test = [v for p in test for v in p]

# Separate the tuning and testing datasets by modality
tune = {'Chest X-Ray (MIMIC)': [d for d in tune if d[3] == 'Chest X-Ray (MIMIC)'], 'Chest X-Ray (CheXpert)': [d for d in tune if d[3] == 'Chest X-Ray (CheXpert)'], 'Skin Lesion': [d for d in tune if d[3] == 'Skin Lesion'], 'MRI': [d for d in tune if d[3] == 'MRI'], 'Amyloid PET': [d for d in tune if d[3] == 'Amyloid PET'], 'FDG PET': [d for d in tune if d[3] == 'FDG PET'], 'CT': [d for d in tune if d[3] == 'CT'], 'Chest X-Ray (COVID-QU-Ex)': [d for d in tune if d[3] == 'Chest X-Ray (COVID-QU-Ex)']}
test = {'Chest X-Ray (MIMIC)': [d for d in test if d[3] == 'Chest X-Ray (MIMIC)'], 'Chest X-Ray (CheXpert)': [d for d in test if d[3] == 'Chest X-Ray (CheXpert)'], 'Skin Lesion': [d for d in test if d[3] == 'Skin Lesion'], 'MRI': [d for d in test if d[3] == 'MRI'], 'Amyloid PET': [d for d in test if d[3] == 'Amyloid PET'], 'FDG PET': [d for d in test if d[3] == 'FDG PET'], 'CT': [d for d in test if d[3] == 'CT'], 'Chest X-Ray (COVID-QU-Ex)': [d for d in test if d[3] == 'Chest X-Ray (COVID-QU-Ex)']}
tune_temporal = {'Chest X-Ray (MIMIC)': [p for p in tune_temporal if p[0][3] == 'Chest X-Ray (MIMIC)'], 'Chest X-Ray (CheXpert)': [p for p in tune_temporal if p[0][3] == 'Chest X-Ray (CheXpert)'], 'Skin Lesion': [p for p in tune_temporal if p[0][3] == 'Skin Lesion'], 'MRI': [p for p in tune_temporal if p[0][3] == 'MRI'], 'Amyloid PET': [p for p in tune_temporal if p[0][3] == 'Amyloid PET'], 'FDG PET': [p for p in tune_temporal if p[0][3] == 'FDG PET'], 'CT': [p for p in tune_temporal if p[0][3] == 'CT'], 'Chest X-Ray (COVID-QU-Ex)': [p for p in tune_temporal if p[0][3] == 'Chest X-Ray (COVID-QU-Ex)']}
test_temporal = {'Chest X-Ray (MIMIC)': [p for p in test_temporal if p[0][3] == 'Chest X-Ray (MIMIC)'], 'Chest X-Ray (CheXpert)': [p for p in test_temporal if p[0][3] == 'Chest X-Ray (CheXpert)'], 'Skin Lesion': [p for p in test_temporal if p[0][3] == 'Skin Lesion'], 'MRI': [p for p in test_temporal if p[0][3] == 'MRI'], 'Amyloid PET': [p for p in test_temporal if p[0][3] == 'Amyloid PET'], 'FDG PET': [p for p in test_temporal if p[0][3] == 'FDG PET'], 'CT': [p for p in test_temporal if p[0][3] == 'CT'], 'Chest X-Ray (COVID-QU-Ex)': [p for p in test_temporal if p[0][3] == 'Chest X-Ray (COVID-QU-Ex)']}
taskMap = {'Chest X-Ray (MIMIC)': 'Multi-Label Classification', 'Chest X-Ray (CheXpert)': 'Multi-Label Classification', 'Skin Lesion': 'Multi-Label Classification', 'MRI': 'Multi-Class Classification', 'Amyloid PET': 'Multi-Class Classification', 'FDG PET': 'Multi-Class Classification', 'CT': 'Multi-Class Classification', 'Chest X-Ray (COVID-QU-Ex)': 'Multi-Class Classification'}

pickle.dump(train, open(data_dir + 'trainingDataset.pkl', 'wb'))
pickle.dump(tune, open(data_dir + 'tuningDataset.pkl', 'wb'))
pickle.dump(test, open(data_dir + 'testingDataset.pkl', 'wb'))
pickle.dump(tune_temporal, open(data_dir + 'tuningTemporalDataset.pkl', 'wb'))
pickle.dump(test_temporal, open(data_dir + 'testingTemporalDataset.pkl', 'wb'))
pickle.dump(visualization, open(data_dir + 'visualizationDataset.pkl', 'wb'))
pickle.dump(taskMap, open(data_dir + 'taskMap.pkl', 'wb'))