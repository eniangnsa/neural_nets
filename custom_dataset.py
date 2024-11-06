import os
import pandas as pd 
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader

class EniangDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_label = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_label)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.img_dir, self.img_label.iloc[index, 0])
        image = read_image(image_path) # converts image to tensor
        label = self.img_label.iloc[index,1] # retrieves corresponding label
        if self.transform:
            image = self.transform(image) # apply normalization to the tensor so the values are between [0,1]
        if self.target_transform:
            label = self.target_transform(label) # apply lambda function for one-hot-encoding
        return image, label