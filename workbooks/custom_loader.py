import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from skimage import io
from PIL import Image

# custom dataset object used in the dataloaders
class ShoeDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])

        image = io.imread((img_path))

        label = int(self.annotations.iloc[index, 0])

        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)

        return image, label
