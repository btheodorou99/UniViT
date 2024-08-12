from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from collections import Counter
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import os

# Set the output directory
data_dir = "/shared/eng/bpt3/data/UniViT/data/"
os.makedirs(data_dir, exist_ok=True)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# Initialize the dataset
dataset = {}

# Set the different directories
mimic_cxr_dir = "/srv/local/data/MIMIC-CXR/"
chexpert_dir = "/srv/local/data/CheXpert/"
isic_dir = "/srv/local/data/ISIC_SkinLesion/"
adni_dir = "/srv/local/data/ADNI/"
deep_lesion_dir = "/srv/local/data/DeepLesion/"
covid_cxr_dir = "/srv/local/data/COVID-QU-Ex/"
crc_he_dir1 = "/srv/local/data/NCT-CRC-HE-100K/"
crc_he_dir2 = "/srv/local/data/CRC-VAL-HE-7K/"
brats_dir = "/srv/local/data/BraTS/"
acdc_dir = "/srv/local/data/ACDC/processed/"

# Do some label processing
chestXrayLabelIdx = [0, 1, 3, 7, 8, 12, 13]

# Process the MIMIC-CXR dataset
df = (
    pd.read_csv(mimic_cxr_dir + "cxr-record-list.csv")
    .merge(
        pd.read_csv(mimic_cxr_dir + "mimic-cxr-report.csv"),
        on=["subject_id", "study_id"],
    )
    .merge(
        pd.read_csv(mimic_cxr_dir + "mimic-cxr-2.0.0-metadata.csv"),
        on=["subject_id", "study_id", "dicom_id"],
    )
    .merge(
        pd.read_csv(mimic_cxr_dir + "mimic-cxr-2.0.0-chexpert.csv"),
        on=["subject_id", "study_id"],
    )
)
df = df.sort_values(
    by=["subject_id", "study_id", "StudyDate", "StudyTime"],
    ascending=[True, True, True, True],
)[
    [
        "subject_id",
        "study_id",
        "dicom_id",
        "StudyDate",
        "StudyTime",
        "report",
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "Lung Opacity",
        "No Finding",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumonia",
        "Pneumothorax",
        "Support Devices",
    ]
]
df["report"] = [
    [t - 100 for t in tokenizer.encode(x)] for x in df["report"]
]  # 100 is [UNK], 101 is [CLS], 102 is [SEP]


def labelMap(val):
    if val == 1:
        return 1
    else:
        return 0


for row in tqdm(df.itertuples(), total=len(df), desc="Processing MIMIC-CXR"):
    subject_id = row.subject_id
    study_id = row.study_id
    dicom_id = row.dicom_id
    path = f"{mimic_cxr_dir}resized/{dicom_id}.jpg"
    note = row.report
    labels = [
        row.Atelectasis,
        row.Cardiomegaly,
        row.Consolidation,
        row.Edema,
        row._11,
        row.Fracture,
        row._13,
        row._14,
        row._15,
        row._16,
        row._17,
        row.Pneumonia,
        row.Pneumothorax,
        row._20,
    ]
    labels = [labelMap(l) for i, l in enumerate(labels) if i in chestXrayLabelIdx]
    dimensions = (1, 256, 256)
    modality = "Chest X-Ray (MIMIC)"
    data_point = (path, dimensions, note, modality, labels)
    if subject_id not in dataset:
        dataset[subject_id] = [data_point]
    else:
        dataset[subject_id].append(data_point)

# Process the CheXpert dataset
df = pd.read_csv(chexpert_dir + "chexpert-meta.csv")
df = df[
    [
        "imgpath",
        "Sex",
        "Age",
        "Frontal/Lateral",
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "Lung Opacity",
        "No Finding",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumonia",
        "Pneumothorax",
        "Support Devices",
    ]
]
attr_map = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "_9": "Enlarged Cardiomediastinum",
    "Fracture": "Fracture",
    "_11": "Lung Lesion",
    "_12": "Lung Opacity",
    "_13": "No Finding",
    "_14": "Pleural Effusion",
    "_15": "Pleural Other",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
    "_18": "Support Devices",
}
for row in tqdm(df.itertuples(), total=len(df), desc="Processing CheXpert"):
    imgpath = row.imgpath
    subject_id = imgpath.split("/study")[0].split("/")[-1]
    path = f'{chexpert_dir}{imgpath.split("./")[1]}'
    findings = [
        attr_map[c] for c in row._fields if c != "Index" and getattr(row, c) == 1
    ]
    note = f"{row._4} Chest X-Ray of a {row.Age} year old {row.Sex}"
    if len(findings) > 0:
        if len(findings) == 1:
            note += f" with {findings[0]}"
        elif len(findings) == 2:
            note += f" with {findings[0]} and {findings[1]}"
        else:
            note += f' with {", ".join(findings[:-1])}, and {findings[-1]}'
    note = [t - 100 for t in tokenizer.encode(note)]
    labels = [
        row.Atelectasis,
        row.Cardiomegaly,
        row.Consolidation,
        row.Edema,
        row._9,
        row.Fracture,
        row._11,
        row._12,
        row._13,
        row._14,
        row._15,
        row.Pneumonia,
        row.Pneumothorax,
        row._18,
    ]
    labels = [l for i, l in enumerate(labels) if i in chestXrayLabelIdx]
    dimensions = Image.open(path).size
    dimensions = (1, dimensions[1], dimensions[0])
    modality = "Chest X-Ray (CheXpert)"
    data_point = (path, dimensions, note, modality, labels)
    if subject_id not in dataset:
        dataset[subject_id] = [data_point]
    else:
        dataset[subject_id].append(data_point)

# Process the skin lesion dataset
df = pd.read_csv(isic_dir + "ISIC_2019_Training_GroundTruth.csv").merge(
    pd.read_csv(isic_dir + "ISIC_2019_Training_Metadata.csv"), on=["image"]
)
condMap = {
    "MEL": "Melanoma",
    "NV": "Melanocytic nevus",
    "BCC": "Basal cell carcinoma",
    "AK": "Actinic keratosis",
    "BKL": "Benign keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular lesion",
    "SCC": "Squamous cell carcinoma",
}
for row in tqdm(df.itertuples(), total=len(df), desc="Processing ISIC"):
    imgpath = row.image
    subject_id = imgpath.split("_")[1]
    path = f"{isic_dir}ISIC_2019_Data/{imgpath}.jpg"
    findings = [
        condMap[c] for c in row._fields if getattr(row, c) == 1 and c in condMap
    ]
    note = f"Skin lesion image from the {row.anatom_site_general} of a {row.age_approx} year old {row.sex}"
    if len(findings) > 0:
        if len(findings) == 1:
            note += f" showing {findings[0]}"
        elif len(findings) == 2:
            note += f" showing {findings[0]} and {findings[1]}"
        else:
            note += f' showing {", ".join(findings[:-1])}, and {findings[-1]}'
    note = [t - 100 for t in tokenizer.encode(note)]
    labels = [getattr(row, c) for c in row._fields if c in condMap]
    dimensions = Image.open(path).size
    dimensions = (1, dimensions[1], dimensions[0])
    modality = "Skin Lesion"
    data_point = (path, dimensions, note, modality, labels)
    if subject_id not in dataset:
        dataset[subject_id] = [data_point]
    else:
        dataset[subject_id].append(data_point)

# Load the ADNI Labels
adni_labels = pd.read_csv(adni_dir + "DXSUM_PDXCONV.csv")
adni_labels = {row.PTID: row.DIAGNOSIS for row in adni_labels.itertuples()}
adni_labels = {p: int(l - 1) if l == l else 0 for p, l in adni_labels.items()}
# LABELS: CN=0, MCI=1, AD=2

# Process the ADNI MRI dataset
data = pickle.load(open(adni_dir + "mri_data.pkl", "rb"))
for subject_id in tqdm(data, desc="Processing ADNI MRI"):
    for date in sorted(data[subject_id].keys()):
        path = f'{adni_dir}MRI/{data[subject_id][date]["filename"]}'
        dimensions = data[subject_id][date]["shape"]
        dimensions = (dimensions[2], dimensions[0], dimensions[1])
        note = [t - 100 for t in tokenizer.encode(f"T1-weighted MRI of a brain")]
        labels = adni_labels[subject_id] if subject_id in adni_labels else None
        modality = "MRI"
        if subject_id not in dataset:
            dataset[subject_id] = [(path, dimensions, note, modality, labels)]
        else:
            dataset[subject_id].append((path, dimensions, note, modality, labels))

# Process the ADNI Amyloid PET dataset
data = pickle.load(open(adni_dir + "av45_pet_data.pkl", "rb"))
for subject_id in tqdm(data, desc="Processing ADNI AV45 PET"):
    for date in sorted(data[subject_id].keys()):
        path = f'{adni_dir}AV45_PET/{data[subject_id][date]["filename"]}'
        dimensions = data[subject_id][date]["shape"]
        dimensions = (dimensions[2], dimensions[0], dimensions[1])
        note = [t - 100 for t in tokenizer.encode(f"Amyloid PET scan of a brain")]
        labels = adni_labels[subject_id] if subject_id in adni_labels else None
        modality = "Amyloid PET"
        if subject_id not in dataset:
            dataset[subject_id] = [(path, dimensions, note, modality, labels)]
        else:
            dataset[subject_id].append((path, dimensions, note, modality, labels))

# Process the ADNI FDG PET dataset
data = pickle.load(open(adni_dir + "fdg_pet_data.pkl", "rb"))
for subject_id in tqdm(data, desc="Processing ADNI FDG PET"):
    for date in sorted(data[subject_id].keys()):
        path = f'{adni_dir}FDG_PET/{data[subject_id][date]["filename"]}'
        dimensions = data[subject_id][date]["shape"]
        dimensions = (dimensions[2], dimensions[0], dimensions[1])
        note = [t - 100 for t in tokenizer.encode(f"FDG PET scan of a brain")]
        labels = adni_labels[subject_id] if subject_id in adni_labels else None
        modality = "FDG PET"
        if subject_id not in dataset:
            dataset[subject_id] = [(path, dimensions, note, modality, labels)]
        else:
            dataset[subject_id].append((path, dimensions, note, modality, labels))

# Process the DeepLesion dataset
data = pd.read_csv(deep_lesion_dir + "DL_info.csv")
data.File_name = [
    f"{deep_lesion_dir}{'_'.join(f.split('_')[:-1])}.npy" for f in data.File_name
]
labelMap = {f: [] for f in data.File_name}
for row in data.itertuples():
    if row.Coarse_lesion_type != -1:
        labelMap[row.File_name].append(row.Coarse_lesion_type - 1)
data = data.drop_duplicates(subset="File_name")
for row in tqdm(data.itertuples(), total=len(data), desc="Processing DeepLesion"):
    path = row.File_name
    dimensions = np.load(path).shape
    dimensions = (dimensions[2], dimensions[0], dimensions[1])
    note = [
        t - 100
        for t in tokenizer.encode(
            f'CT scan of a {row.Patient_age} year old {"male" if row.Patient_gender == "M" else "female"} patient with a lesion'
        )
    ]
    labels = labelMap[row.File_name]
    labels = [1 if i in labels else 0 for i in range(8)] if labels else None
    subject_id = f"DL_{row.Patient_index}"
    modality = "CT"
    if subject_id not in dataset:
        dataset[subject_id] = [(path, dimensions, note, modality, labels)]
    else:
        dataset[subject_id].append((path, dimensions, note, modality, labels))

# Process the COVID-QU-Ex dataset
for top_dir in tqdm(os.listdir(covid_cxr_dir), desc="Processing COVID-QU-Ex"):
    if os.path.isdir(f"{covid_cxr_dir}{top_dir}"):
        for split in os.listdir(f"{covid_cxr_dir}{top_dir}/{top_dir}/"):
            for cls in os.listdir(f"{covid_cxr_dir}{top_dir}/{top_dir}/{split}/"):
                for img in os.listdir(
                    f"{covid_cxr_dir}{top_dir}/{top_dir}/{split}/{cls}/images/"
                ):
                    path = (
                        f"{covid_cxr_dir}{top_dir}/{top_dir}/{split}/{cls}/images/{img}"
                    )
                    dimensions = Image.open(path).size
                    dimensions = (1, dimensions[1], dimensions[0])
                    note = [
                        t - 100
                        for t in tokenizer.encode(f"Chest X-Ray of a {cls} patient")
                    ]
                    labels = 0 if cls == "COVID-19" else 1 if cls == "Non-COVID" else 2
                    subject_id = (
                        path.split("/")[-1].split("_")[0] if "sub-" in path else path
                    )
                    modality = "Chest X-Ray (COVID-QU-Ex)"
                    if subject_id not in dataset:
                        dataset[subject_id] = [
                            (path, dimensions, note, modality, labels)
                        ]
                    else:
                        dataset[subject_id].append(
                            (path, dimensions, note, modality, labels)
                        )

# Process the CRC-HE dataset
clsMap = {
    "ADI": "Adipose",
    "BACK": "Background",
    "DEB": "Debris",
    "LYM": "Lymphocytes",
    "MUC": "Mucus",
    "MUS": "Smooth Muscle",
    "NORM": "Normal Colon Mucosa",
    "STR": "Cancer-Associated Stroma",
    "TUM": "Colorectal Adenocarcinoma Epithelium",
}
labelIdx = {
    "ADI": 0,
    "BACK": 1,
    "DEB": 2,
    "LYM": 3,
    "MUC": 4,
    "MUS": 5,
    "NORM": 6,
    "STR": 7,
    "TUM": 8,
}
for cls in tqdm(os.listdir(crc_he_dir1), desc="Processing CRC-HE (Part 1)"):
    for img in os.listdir(f"{crc_he_dir1}{cls}/"):
        path = f"{crc_he_dir1}{cls}/{img}"
        dimensions = Image.open(path).size
        dimensions = (1, dimensions[1], dimensions[0])
        note = [
            t - 100
            for t in tokenizer.encode(f"Histopathology image of a {clsMap[cls]}")
        ]
        labels = labelIdx[cls]
        subject_id = path.split("/")[-1].split("-")[-1].split(".")[0]
        modality = "Histopathology"
        if subject_id not in dataset:
            dataset[subject_id] = [(path, dimensions, note, modality, labels)]
        else:
            dataset[subject_id].append((path, dimensions, note, modality, labels))

for cls in tqdm(os.listdir(crc_he_dir2), desc="Processing CRC-HE (Part 2)"):
    for img in os.listdir(f"{crc_he_dir2}{cls}/"):
        path = f"{crc_he_dir2}{cls}/{img}"
        dimensions = Image.open(path).size
        dimensions = (1, dimensions[1], dimensions[0])
        note = [
            t - 100
            for t in tokenizer.encode(f"Histopathology image of a {clsMap[cls]}")
        ]
        labels = labelIdx[cls]
        subject_id = path.split("/")[-1].split("-")[-1].split(".")[0]
        modality = "Histopathology"
        if subject_id not in dataset:
            dataset[subject_id] = [(path, dimensions, note, modality, labels)]
        else:
            dataset[subject_id].append((path, dimensions, note, modality, labels))

# Process the BraTS dataset
subdir = "BraTS-Path"
clsMap = {
    "CT": "Cellular Tumor",
    "PN": "Pseudopalisading Necrosis",
    "MP": "Microvascular Proliferation",
    "NC": "Geographic Necrosis",
    "IC": "Cortex Infiltration",
    "WM": "White Matter Penetration",
    "Validation": "Validation",
}
labelIdx = {
    "CT": 0,
    "PN": 1,
    "MP": 2,
    "NC": 3,
    "IC": 4,
    "WM": 5,
}
for cls in tqdm(
    os.listdir(os.path.join(brats_dir, subdir)), desc="Processing BraTS-Path"
):
    if not os.path.isdir(os.path.join(brats_dir, subdir, cls)):
        continue
    for img in os.listdir(os.path.join(brats_dir, subdir, cls)):
        path = os.path.join(brats_dir, subdir, cls, img)
        dimensions = Image.open(path).size
        dimensions = (1, dimensions[1], dimensions[0])
        note = [
            t - 100
            for t in tokenizer.encode(f"Pathology image of a {clsMap[cls]} patient")
        ]
        labels = None if cls == "Validation" else labelIdx[cls]
        subject_id = img.split(".")[0]
        modality = "BraTS-Path"
        if subject_id not in dataset:
            dataset[subject_id] = [(path, dimensions, note, modality, labels)]
        else:
            dataset[subject_id].append((path, dimensions, note, modality, labels))

subdir = "BraTS-MEN-RT"
for split in tqdm(["Train", "Val"], desc="Processing BraTS-MEN-RT"):
    for subj_dir in os.listdir(os.path.join(brats_dir, subdir, split)):
        path = os.path.join(brats_dir, subdir, split, subj_dir, f"{subj_dir}_t1c.npy")
        dimensions = np.load(path).shape
        dimensions = (dimensions[2], dimensions[0], dimensions[1])
        note = [t - 100 for t in tokenizer.encode(f"T1 MRI image of a patient")]
        labels = None if split == "Validation" else path.replace("t1c", "gtv")
        subject_id = "-".join(subj_dir.split("-")[:-1])
        modality = "MRI (BraTS-MEN-RT)"
        if subject_id not in dataset:
            dataset[subject_id] = [(path, dimensions, note, modality, labels)]
        else:
            dataset[subject_id].append((path, dimensions, note, modality, labels))

for subdir in ["BraTS-GLI", "BraTS-GoAT", "BraTS-MET", "BraTS-PED"]:
    for split in tqdm(["Train", "Val"], desc=f"Processing {subdir}"):
        for subj_dir in tqdm(
            os.listdir(os.path.join(brats_dir, subdir, split)),
            desc=f"Processing {subdir} {split}",
            leave=False,
        ):
            for imgType in ["t1c", "t1n", "t2f", "t2w"]:
                path = os.path.join(
                    brats_dir, subdir, split, subj_dir, f"{subj_dir}-{imgType}.npy"
                )
                if ".DS_Store" in path:
                    continue
                dimensions = np.load(path).shape
                dimensions = (dimensions[2], dimensions[0], dimensions[1])
                note = [
                    t - 100
                    for t in tokenizer.encode(
                        f"{imgType.upper()} MRI image of a patient"
                    )
                ]
                labels = None if split == "Validation" else path.replace(imgType, "seg")
                subject_id = (
                    "-".join(subj_dir.split("-")[:-1])
                    if subdir != "BraTS-GoAT"
                    else subj_dir
                ) + f"-{imgType}"
                modality = f"{imgType.upper()} MRI ({subdir})"
                if subject_id not in dataset:
                    dataset[subject_id] = [(path, dimensions, note, modality, labels)]
                else:
                    dataset[subject_id].append(
                        (path, dimensions, note, modality, labels)
                    )

# Process the ACDC dataset
clsMap = {
    "NOR": "Normal",
    "MINF": "Myocardial Infarction",
    "DCM": "Dilated Cardiomyopathy",
    "HCM": "Hypertrophic Cardiomyopathy",
    "RV": "Abnormal Right Ventricle",
}
labelIdx = {"NOR": 0, "MINF": 1, "DCM": 2, "HCM": 3, "RV": 4}
for cls in tqdm(os.listdir(acdc_dir), desc="Processing ACDC"):
    for subj in os.listdir(os.path.join(acdc_dir, cls)):
        for img in os.listdir(os.path.join(acdc_dir, cls, subj)):
            path = os.path.join(acdc_dir, cls, subj, img)
            dimensions = np.load(path).shape
            dimensions = (dimensions[2], dimensions[0], dimensions[1])
            note = [
                t - 100
                for t in tokenizer.encode(
                    f"Cardiac MRI image of a {clsMap[cls]} patient"
                )
            ]
            labels = labelIdx[cls]
            subject_id = f"ACDC-{subj}"
            modality = "Cardiac MRI (ACDC)"
            if subject_id not in dataset:
                dataset[subject_id] = [(path, dimensions, note, modality, labels)]
            else:
                dataset[subject_id].append((path, dimensions, note, modality, labels))

dataset = list(dataset.values())
modalities = list(set([p[0][3] for p in dataset]))
for m in modalities:
    print(m)
    d = [p for p in dataset if p[0][3] == m]
    print(len(d))
    print(len([v for p in d for v in p]))
    print(max([len(p) for p in d]))
    print(Counter([v[1] for p in d for v in p]).most_common(5))
    print()

pickle.dump(dataset, open(data_dir + "intermediate_dataset.pkl", "wb"))

# Split the dataset
# if modality is less than 5000, then split 0.4 test, otherwise 0.2 test
splits = []
for m in modalities:
    d = [p for p in dataset if p[0][3] == m]
    unlabeled = [p for p in d if p[0][4] is None]
    labeled = [p for p in d if p[0][4] is not None]
    d_train, d_test = train_test_split(
        labeled, test_size=0.2 if len(labeled) >= 5000 else 0.4, random_state=4
    )
    d_train = d_train + unlabeled
    splits.append((d_train, d_test))
train = [p for d in splits for p in d[0]]
test = [p for d in splits for p in d[1]]
tune, test = train_test_split(test, test_size=0.5, random_state=4)
visualization = tune + test
tune_temporal = [p for p in tune if len(p) >= 3]
test_temporal = [p for p in test if len(p) >= 3]
tune = [v for p in tune for v in p]
test = [v for p in test for v in p]

# Separate the tuning and testing datasets by modality
tune = {m: [d for d in tune if d[3] == m] for m in modalities}
test = {m: [d for d in test if d[3] == m] for m in modalities}
tune_temporal = {m: [p for p in tune_temporal if p[0][3] == m] for m in modalities}
test_temporal = {m: [p for p in test_temporal if p[0][3] == m] for m in modalities}
taskMap = {
    "Chest X-Ray (MIMIC)": "Multi-Label Classification",
    "Chest X-Ray (CheXpert)": "Multi-Label Classification",
    "Skin Lesion": "Multi-Label Classification",
    "MRI": "Multi-Class Classification",
    "Amyloid PET": "Multi-Class Classification",
    "FDG PET": "Multi-Class Classification",
    "CT": "Multi-Label Classification",
    "Chest X-Ray (COVID-QU-Ex)": "Multi-Class Classification",
    "Histopathology": "Multi-Class Classification",
    "BraTS-Path": "Multi-Class Classification",
    "Cardiac MRI (ACDC)": "Multi-Class Classification",
    "T1C MRI (BraTS-GoAT)": "Segmentation",
    "T1C MRI (BraTS-GLI)": "Segmentation",
    "T1C MRI (BraTS-MET)": "Segmentation",
    "T1C MRI (BraTS-PED)": "Segmentation",
    "T1N MRI (BraTS-GoAT)": "Segmentation",
    "T1N MRI (BraTS-GLI)": "Segmentation",
    "T1N MRI (BraTS-MET)": "Segmentation",
    "T1N MRI (BraTS-PED)": "Segmentation",
    "T2W MRI (BraTS-GoAT)": "Segmentation",
    "T2W MRI (BraTS-GLI)": "Segmentation",
    "T2W MRI (BraTS-MET)": "Segmentation",
    "T2W MRI (BraTS-PED)": "Segmentation",
    "T2F MRI (BraTS-GoAT)": "Segmentation",
    "T2F MRI (BraTS-GLI)": "Segmentation",
    "T2F MRI (BraTS-MET)": "Segmentation",
    "T2F MRI (BraTS-PED)": "Segmentation",
    "MRI (BraTS-MEN-RT)": "Segmentation",
}
flatTasks = set(
    [
        "Chest X-Ray (MIMIC)",
        "Chest X-Ray (CheXpert)",
        "Skin Lesion",
        "Chest X-Ray (COVID-QU-Ex)",
        "Histopathology",
        "BraTS-Path",
    ]
)

pickle.dump(train, open(data_dir + "trainingDataset.pkl", "wb"))
pickle.dump(tune, open(data_dir + "tuningDataset.pkl", "wb"))
pickle.dump(test, open(data_dir + "testingDataset.pkl", "wb"))
pickle.dump(tune_temporal, open(data_dir + "tuningTemporalDataset.pkl", "wb"))
pickle.dump(test_temporal, open(data_dir + "testingTemporalDataset.pkl", "wb"))
pickle.dump(visualization, open(data_dir + "visualizationDataset.pkl", "wb"))
pickle.dump(taskMap, open(data_dir + "taskMap.pkl", "wb"))
pickle.dump(flatTasks, open(data_dir + "flatTasks.pkl", "wb"))

# Chest X-Ray (MIMIC)
# 65379
# 377095
# 174
# [((1, 256, 256), 377095)]

# Chest X-Ray (CheXpert)
# 64540
# 223414
# 92
# [((1, 320, 390), 134659), ((1, 320, 389), 21876), ((1, 320, 320), 21792), ((1, 390, 320), 5835), ((1, 369, 320), 4934)]

# Chest X-Ray (COVID-QU-Ex)
# 34854
# 39746
# 35
# [((1, 256, 256), 39746)]

# Skin Lesion
# 25331
# 25331
# 1
# [((1, 1024, 1024), 12414), ((1, 450, 600), 10015), ((1, 680, 1024), 1121), ((1, 768, 1024), 774), ((1, 682, 1024), 173)]

# Histopathology
# 107180
# 107180
# 1
# [((1, 224, 224), 107180)]

# CT
# 4427
# 14601
# 37
# [((61, 512, 512), 3397), ((13, 512, 512), 2628), ((122, 512, 512), 624), ((26, 512, 512), 608), ((31, 512, 512), 328)]

# BraTS-Path
# 160301
# 160301
# 1
# [((1, 512, 512), 160301)]

# MRI
# 2507
# 15948
# 29
# [((1, 192, 192), 3140), ((5, 256, 256), 1452), ((47, 128, 128), 1434), ((63, 128, 128), 1337), ((3, 256, 256), 1266)]

# FDG PET
# 35
# 121
# 6
# [((47, 128, 128), 43), ((128, 128, 128), 21), ((81, 168, 168), 21), ((63, 128, 128), 11), ((90, 128, 128), 9)]

# Amyloid PET
# 53
# 166
# 9
# [((90, 128, 128), 49), ((81, 336, 336), 37), ((81, 256, 256), 23), ((47, 128, 128), 20), ((81, 168, 168), 18)]

# Cardiac MRI (ACDC)
# 150
# 3972
# 35
# [((9, 216, 256), 389), ((10, 216, 256), 328), ((8, 216, 256), 308), ((10, 256, 216), 253), ((9, 256, 216), 191)]

# MRI (BraTS-MEN-RT)
# 570
# 570
# 1
# [((124, 256, 256), 105), ((136, 512, 512), 46), ((172, 256, 256), 39), ((176, 256, 256), 22), ((120, 512, 512), 14)]

# T1C MRI (BraTS-GLI)
# 700
# 1538
# 10
# [((182, 182, 218), 1538)]

# T2W MRI (BraTS-GLI)
# 700
# 1538
# 10
# [((182, 182, 218), 1538)]

# T2W MRI (BraTS-GoAT)
# 1802
# 1802
# 1
# [((155, 240, 240), 1802)]

# T2F MRI (BraTS-GoAT)
# 1802
# 1802
# 1
# [((155, 240, 240), 1802)]

# T1N MRI (BraTS-PED)
# 351
# 351
# 1
# [((155, 240, 240), 351)]

# T2W MRI (BraTS-MET)
# 639
# 740
# 7
# [((155, 240, 240), 416), ((126, 256, 256), 46), ((104, 320, 320), 37), ((90, 400, 400), 33), ((106, 256, 256), 30)]

# T1N MRI (BraTS-GoAT)
# 1802
# 1802
# 1
# [((155, 240, 240), 1802)]

# T1C MRI (BraTS-GoAT)
# 1802
# 1802
# 1
# [((155, 240, 240), 1802)]

# T2F MRI (BraTS-PED)
# 351
# 351
# 1
# [((155, 240, 240), 351)]

# T2F MRI (BraTS-MET)
# 639
# 740
# 7
# [((155, 240, 240), 416), ((126, 256, 256), 46), ((104, 320, 320), 37), ((90, 400, 400), 33), ((106, 256, 256), 30)]

# T1C MRI (BraTS-MET)
# 639
# 740
# 7
# [((155, 240, 240), 416), ((126, 256, 256), 46), ((104, 320, 320), 37), ((90, 400, 400), 33), ((106, 256, 256), 30)]

# T1C MRI (BraTS-PED)
# 351
# 351
# 1
# [((155, 240, 240), 351)]

# T2W MRI (BraTS-PED)
# 351
# 351
# 1
# [((155, 240, 240), 351)]

# T1N MRI (BraTS-MET)
# 639
# 740
# 7
# [((155, 240, 240), 416), ((126, 256, 256), 46), ((104, 320, 320), 37), ((90, 400, 400), 33), ((106, 256, 256), 30)]

# T2F MRI (BraTS-GLI)
# 700
# 1538
# 10
# [((182, 182, 218), 1538)]

# T1N MRI (BraTS-GLI)
# 700
# 1538
# 10
# [((182, 182, 218), 1538)]
