# Dataset.py

import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        image = cv2.imread(img_name)
        label_name = os.path.join(self.label_dir, self.images[idx].replace('.jpg', '.png'))  # Assuming labels are in PNG format
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image = self.transform(image)

        return image, label