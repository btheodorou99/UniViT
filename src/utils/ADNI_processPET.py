# ADNI PET SEARCH:
# Modality = PET
# Radiopharmaceutical = "18F-AV45"
# Frames?

# DIRECTORY STRUCTURE:
# ADNI/
#   subjectID/
#     seriesName/
#       date_otherStuff/
#         someIDs/
#           dicomFiles

import os
import ants
import pickle
import subprocess
import numpy as np
from tqdm import tqdm
from pypet2bids.ecat import Ecat

pet_dir = "/srv/local/data/ADNI/FDG_PET_Orig/"
output_dir = "/srv/local/data/ADNI/FDG_PET/"
data_dict = {}
# missing = []

# Create a temporary directory
tempdir = "/srv/local/data/ADNI/temp/"
if not os.path.exists(tempdir):
    os.makedirs(tempdir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for subject_id in tqdm(os.listdir(pet_dir)):
    subject_path = os.path.join(pet_dir, subject_id)
    if not os.path.isdir(subject_path):
        continue

    for series_name in os.listdir(subject_path):
        series_path = os.path.join(subject_path, series_name)
        if not os.path.isdir(series_path):
            continue
        
        for date_other in os.listdir(series_path):
            date = date_other.split('_')[0]
            date_path = os.path.join(series_path, date_other)
            if not os.path.isdir(date_path):
                continue
            
            for some_ids in os.listdir(date_path):
                some_ids_path = os.path.join(date_path, some_ids)
                if not os.path.isdir(some_ids_path):
                    continue
                
                files = os.listdir(some_ids_path)
                if not files:
                    continue
                
                npy_filename = f"{subject_id}--{date}--{some_ids}.npy"
                output_filename = os.path.join(output_dir, npy_filename)
                if os.path.exists(output_filename):
                    if subject_id not in data_dict:
                        data_dict[subject_id] = {}
                    data_dict[subject_id][date] = {'shape': np.load(output_filename).shape, 'filename': npy_filename}
                    continue
                else:
                    continue

                file = files[0]
                try:
                    if file.lower().endswith('.dcm'):
                        # Convert DICOM to NIfTI
                        subprocess.run(["dcm2niix", "-o", tempdir, some_ids_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    elif file.lower().endswith('.v'):
                        # Convert ECAT to NIfTI
                        file_path = os.path.join(some_ids_path, file)
                        ecat = Ecat(ecat_file=file_path, nifti_file=os.path.join(tempdir, 'output.nii'), collect_pixel_data=True)
                        ecat.make_nifti()
                    elif file.lower().endswith('.i.hdr') or file.lower().endswith('.i'):
                        continue
                    else:
                        continue

                    # Process the NIfTI file
                    nifti_files = [f for f in os.listdir(tempdir) if f.endswith(('.nii', '.nii.gz'))]
                    if nifti_files:
                        img = ants.image_read(os.path.join(tempdir, nifti_files[0]))
                        if len(img.shape) == 4:
                            num_time_points = img.shape[3]
                            reference_frame = ants.from_numpy(img[:, :, :, 0])
                            registered_frames = [reference_frame.numpy()]
                            for i in range(1, num_time_points):
                                moving_frame = img[:, :, :, i]
                                registration = ants.registration(fixed=reference_frame, moving=ants.from_numpy(moving_frame), type_of_transform='Rigid')
                                registered_frames.append(registration['warpedmovout'].numpy())

                            img = ants.from_numpy(np.mean(np.array(registered_frames), axis=0))
                        
                        data = img.numpy()
                        data = (data - data.min()) / (data.max() - data.min())
                        
                        # Save processed data
                        np.save(output_filename, data)
                        
                        # Remove the current directory files
                        for f in os.listdir(some_ids_path):
                            os.remove(os.path.join(some_ids_path, f))

                        # Update dictionary
                        if subject_id not in data_dict:
                            data_dict[subject_id] = {}
                        data_dict[subject_id][date] = {'shape': data.shape, 'filename': npy_filename}
                except Exception as e:
                    print(e)
                    # missing.append((subject_id, date, some_ids))
                    pass
                

                for f in os.listdir(tempdir):
                    os.remove(os.path.join(tempdir, f))


pickle.dump(data_dict, open('/srv/local/data/ADNI/fdg_pet_data.pkl', 'wb'))
os.rmdir(tempdir)
os.rmdir(pet_dir)
# pickle.dump(missing, open('./src/data/missing_pet.pkl', 'wb'))