import os
import ants
import numpy as np
from tqdm import tqdm

acdc_dir = "/srv/local/data/ACDC/"
os.makedirs(acdc_dir + "processed/", exist_ok=True)


def get_group_from_cfg(file_path):
    group_value = None
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("Group:"):
                group_value = line.split(":")[1].strip()
                break

    return group_value


for subdir in tqdm(["training", "testing"], desc="Processing ACDC"):
    for subj_dir in tqdm(
        os.listdir(acdc_dir + subdir), desc=f"Processing {subdir}", leave=False
    ):
        if not os.path.isdir(os.path.join(acdc_dir, subdir, subj_dir)):
            continue
        cls = get_group_from_cfg(os.path.join(acdc_dir, subdir, subj_dir, "Info.cfg"))
        if os.path.exists(os.path.join(acdc_dir, "processed", cls, subj_dir)):
            continue
        os.makedirs(os.path.join(acdc_dir, "processed", cls, subj_dir), exist_ok=True)
        img = ants.image_read(
            os.path.join(acdc_dir, subdir, subj_dir, f"{subj_dir}_4d.nii.gz")
        )
        img = img.numpy()
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        for i in range(img.shape[-1]):
            np.save(
                os.path.join(
                    acdc_dir, "processed", cls, subj_dir, f"{subj_dir}_{i}.npy"
                ),
                img[:, :, :, i],
            )
