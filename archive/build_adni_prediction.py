from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import pickle
import os

# Set the output directory
data_dir = '/shared/bpt3/data/ADNI_PRED/data/'
adni_dir = '/srv/local/data/ADNI/'
os.makedirs(data_dir, exist_ok=True)

# Load the ADNI Labels
label_cols1 = ['DIAGNOSIS', 'DXNORM', 'DXNODEP', 'DXMCI', 'DXMDES', 'DXMPTR1', 'DXMPTR2', 'DXMPTR3', 'DXMPTR4', 'DXMPTR5', 'DXMPTR6', 'DXMDUE', 'DXMOTHET', 'DXDSEV', 'DXAD', 'DXAPP', 'DXAPROB', 'DXAPOSS', 'DXPARK', 'DXPDES', 'DXPCOG', 'DXPATYP', 'DXOTHDEM', 'DXODES', 'DXCONFID']
adni_labels1 = pd.read_csv(adni_dir + 'DXSUM_PDXCONV.csv')
adni_labels1 = {row.PTID: [int(row[col] == 1) for col in label_cols1] for _, row in adni_labels1.iterrows()}
label_cols2 = ['BCPREDX', 'BCADAS', 'BCMMSE', 'BCMMSREC', 'BCNMMMS', 'BCNEUPSY', 'BCNONMEM', 'BCFAQ', 'BCCDR', 'BCDEPRES', 'BCSTROKE', 'BCDELIR', 'BCEXTCIR', 'BCCORADL', 'BCCORCOG']
adni_labels2 = pd.read_csv(adni_dir + 'BLCHANGE.csv')
adni_labels2 = {row.PTID: [int(row[col] == 1) for col in label_cols2] for _, row in adni_labels2.iterrows()}
label_cols3 = ['AXNAUSEA', 'AXVOMIT', 'AXDIARRH', 'AXCONSTP', 'AXABDOMN', 'AXSWEATN', 'AXDIZZY', 'AXENERGY', 'AXDROWSY', 'AXVISION', 'AXHDACHE', 'AXDRYMTH', 'AXBREATH', 'AXCOUGH', 'AXPALPIT', 'AXCHEST', 'AXURNDIS', 'AXURNFRQ', 'AXANKLE', 'AXMUSCLE', 'AXRASH', 'AXINSOMN', 'AXDPMOOD', 'AXCRYING', 'AXELMOOD', 'AXWANDER', 'AXFALL', 'AXOTHER']
adni_labels3 = pd.read_csv(adni_dir + 'ADSXLIST.csv')
adni_labels3 = {row.PTID: [int(row[col] == 2) for col in label_cols3] for _, row in adni_labels3.iterrows()}
label_cols = label_cols1 + label_cols2 + label_cols3
adni_labels = {k: v + adni_labels2[k] + adni_labels3[k] for k, v in adni_labels1.items() if k in adni_labels2 and k in adni_labels3}
print(len(adni_labels))
print(len(list(adni_labels.values())[0]))

# Process the ADNI MRI dataset
mri_dataset = []
data = pickle.load(open(adni_dir + 'mri_data.pkl', 'rb'))
for subject_id in tqdm(data, desc='Processing ADNI MRI'):
    if subject_id not in adni_labels:
        continue
      
    for date in sorted(data[subject_id].keys()):
        path = f'{adni_dir}MRI/{data[subject_id][date]["filename"]}'
        labels = adni_labels[subject_id] if subject_id in adni_labels else None
        mri_dataset.append((path, labels))
            
# Process the ADNI Amyloid PET dataset
pet_dataset = []
data = pickle.load(open(adni_dir + 'av45_pet_data.pkl', 'rb'))
for subject_id in tqdm(data, desc='Processing ADNI AV45 PET'):
    if subject_id not in adni_labels:
        continue
      
    for date in sorted(data[subject_id].keys()):
        path = f'{adni_dir}AV45_PET/{data[subject_id][date]["filename"]}'
        labels = adni_labels[subject_id] if subject_id in adni_labels else None
        pet_dataset.append((path, labels))
        
mri_train, mri_test = train_test_split(mri_dataset, test_size=0.2, random_state=42)
mri_train, mri_val = train_test_split(mri_train, test_size=0.1, random_state=42)
pet_train, pet_test = train_test_split(pet_dataset, test_size=0.2, random_state=42)
pet_train, pet_val = train_test_split(pet_train, test_size=0.1, random_state=42)
print(len(mri_train))
print(len(pet_train))
pickle.dump(mri_train, open(data_dir + 'mri_train.pkl', 'wb'))
pickle.dump(mri_val, open(data_dir + 'mri_val.pkl', 'wb'))
pickle.dump(mri_test, open(data_dir + 'mri_test.pkl', 'wb'))
pickle.dump(pet_train, open(data_dir + 'pet_train.pkl', 'wb'))
pickle.dump(pet_val, open(data_dir + 'pet_val.pkl', 'wb'))
pickle.dump(pet_test, open(data_dir + 'pet_test.pkl', 'wb'))

# 915
# 68
# 3276
# 554