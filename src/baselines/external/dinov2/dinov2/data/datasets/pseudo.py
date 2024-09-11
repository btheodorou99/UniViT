import random
import pickle
import math
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder

class PseudoImageFolder(ImageFolder):
    def __init__(self, data_list, transform=None):
        """
        Initialize with a data list, where each item is a list of tuples: 
        (image_path, dimensions, _, label, _)
        and an optional transform function.
        """
        self.data_list = pickle.load(open(data_list, "rb"))
        self.transform = transform
        self.label_to_index = self.create_label_mapping()

    def __len__(self):
        # Return the total number of patients (each patient has multiple images)
        return len(self.data_list)

    def __getitem__(self, index):
        # Randomly pick an image tuple from the data list of the selected patient
        patient_data = self.data_list[index]
        image_tuple = random.choice(patient_data)  # Randomly pick an image tuple
        image_path, _, _, label, _ = image_tuple
        label_index = self.label_to_index[label]
        
        # Load the image (assumes image_path is a valid file path to a 3D or 2D image)
        image = self.load_image(image_path)
        
        # Apply transformations (if any)
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label_index
    
    def create_label_mapping(self):
        """
        Creates a mapping from string labels to integer indices.
        """
        labels = set([label for patient in self.data_list for (_, _, _, label, _) in patient])
        return {label: idx for idx, label in enumerate(sorted(labels))}
    
    def load_image(self, image_path):
        if image_path.endswith(".npy"):
            img = np.load(image_path)
            if len(img.shape) == 4:
                img = img[:, :, :, 0]
            if True: # For now take middle slice
                img = img[:, :, img.shape[2] // 2]
            else: # Possibly take random slice
                img = img[:, :, random.randint(0, img.shape[2] - 1)]
            img = Image.fromarray(img).convert("RGB")
        elif (
            image_path.endswith(".jpg")
            or image_path.endswith(".png")
            or image_path.endswith(".tif")
        ):
            img = Image.open(image_path).convert("RGB")
        else:
            raise ValueError("Invalid image format")
        return img