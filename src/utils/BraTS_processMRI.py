import os
import ants
import numpy as np
from tqdm import tqdm

brats_dir = "/srv/local/data/BraTS"
dirMap = {"Training": "Train", "Validation": "Val"}
for subdir in tqdm(
    ["BraTS-GLI", "BraTS-GoAT", "BraTS-MET", "BraTS-PED", "BraTS-MEN-RT"],
    desc="Processing BraTS",
):
    for split in tqdm(
        ["Training", "Validation"], desc=f"Processing {subdir}", leave=False
    ):
        os.makedirs(os.path.join(brats_dir, subdir, dirMap[split]), exist_ok=True)
        for subj_dir in tqdm(
            os.listdir(os.path.join(brats_dir, subdir, split)), leave=False
        ):
            if not os.path.isdir(os.path.join(brats_dir, subdir, split, subj_dir)):
                continue
            os.makedirs(
                os.path.join(brats_dir, subdir, dirMap[split], subj_dir), exist_ok=True
            )
            for nifty_file in os.listdir(
                os.path.join(brats_dir, subdir, split, subj_dir)
            ):
                ouput_file = os.path.join(
                    brats_dir,
                    subdir,
                    dirMap[split],
                    subj_dir,
                    nifty_file.replace(".nii.gz", ".npy"),
                )
                if nifty_file.endswith(".nii.gz") and not os.path.exists(ouput_file):
                    # print(nifty_file)
                    img = ants.image_read(
                        os.path.join(brats_dir, subdir, split, subj_dir, nifty_file)
                    )
                    img = img.numpy()
                    img = (img - np.min(img)) / (np.max(img) - np.min(img))
                    np.save(ouput_file, img)
