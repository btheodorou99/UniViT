import os
import pickle
import numpy as np
from tqdm import tqdm
from skimage.filters import threshold_otsu

data_dir = "/shared/eng/bpt3/data/UniViT/data"
BUFFER = 1

train = pickle.load(open(f"{data_dir}/trainingDataset.pkl", "rb"))
tune = pickle.load(open(f"{data_dir}/tuningDataset.pkl", "rb"))
test = pickle.load(open(f"{data_dir}/testingDataset.pkl", "rb"))
tune_temporal = pickle.load(open(f"{data_dir}/tuningTemporalDataset.pkl", "rb"))
test_temporal = pickle.load(open(f"{data_dir}/testingTemporalDataset.pkl", "rb"))

imgs = (
        [i[0] for p in train for i in p if i[0].endswith(".npy")] + 
        [i[0] for m in tune for i in tune[m] if i[0].endswith(".npy")] + 
        [i[0] for m in test for i in test[m] if i[0].endswith(".npy")] + 
        [i[0] for m in tune_temporal for p in tune_temporal[m] for i in p if i[0].endswith(".npy")] + 
        [i[0] for m in test_temporal for p in test_temporal[m] for i in p if i[0].endswith(".npy")]
)
imgs = list(set(imgs))
shapes = {}
for f in tqdm(imgs):
    processed_path = f.replace(".npy", "_processed.npy")
    if os.path.exists(processed_path):
        img = np.load(processed_path)
        img = img.transpose(2, 0, 1)
        shapes[f] = tuple(img.shape)
    else:
        img = np.load(f)
        if len(img.shape) == 4:
            img = img[:, :, :, 0]
        img = img.transpose(2, 0, 1)
        threshold = threshold_otsu(img)
        mask = img >= threshold
        coords = np.argwhere(mask)
        min_coords = np.maximum(np.min(coords, axis=0) - BUFFER, 0)
        max_coords = np.minimum(np.max(coords, axis=0) + BUFFER + 1, img.shape)
        img = img[
                    min_coords[0]:max_coords[0], 
                    min_coords[1]:max_coords[1], 
                    min_coords[2]:max_coords[2]
                ]
        shapes[f] = tuple(img.shape)
        img = img.transpose(1, 2, 0)
        np.save(processed_path, img)
    
# train_processed = [[(i[0].replace(".npy", "_processed.npy"), shapes[i[0]], i[2], i[3], i[4]) if i[0].endswith(".npy") else i for i in p] for p in train]
# tune_processed = {m: [(i[0].replace(".npy", "_processed.npy"), shapes[i[0]], i[2], i[3], i[4]) if i[0].endswith(".npy") else i for i in tune[m]] for m in tune}
# test_processed = {m: [(i[0].replace(".npy", "_processed.npy"), shapes[i[0]], i[2], i[3], i[4]) if i[0].endswith(".npy") else i for i in test[m]] for m in test}
# tune_temporal_processed = {m: [[(i[0].replace(".npy", "_processed.npy"), shapes[i[0]], i[2], i[3], i[4]) if i[0].endswith(".npy") else i for i in p] for p in tune_temporal[m]] for m in tune_temporal}
# test_temporal_processed = {m: [[(i[0].replace(".npy", "_processed.npy"), shapes[i[0]], i[2], i[3], i[4]) if i[0].endswith(".npy") else i for i in p] for p in test_temporal[m]] for m in test_temporal}
# pickle.dump(train_processed, open(f"{data_dir}/trainingDataset_processed.pkl", "wb"))
# pickle.dump(tune_processed, open(f"{data_dir}/tuningDataset_processed.pkl", "wb"))
# pickle.dump(test_processed, open(f"{data_dir}/testingDataset_processed.pkl", "wb"))
# pickle.dump(tune_temporal_processed, open(f"{data_dir}/tuningTemporalDataset_processed.pkl", "wb"))
# pickle.dump(test_temporal_processed, open(f"{data_dir}/testingTemporalDataset_processed.pkl", "wb"))