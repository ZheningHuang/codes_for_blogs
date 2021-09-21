import torch
from glob import glob
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

class CrackDataset(Dataset):

    def __init__(self, image_dir, mask_dir, mode = "train" ):
        self.image_files=sorted(glob(image_dir))
        self.mask_files=sorted(glob(mask_dir))

        # train, val, test = 8: 1: 1
        self.image_files_train = self.image_files[:len(self.image_files)//5*4]
        self.image_files_val = self.image_files[len(self.image_files)//5*4:len(self.image_files)//10*9]
        self.image_files_test = self.image_files[len(self.image_files)//10*9 :]

        self.mask_files_train = self.mask_files[:len(self.mask_files)//5*4]
        self.mask_files_val = self.mask_files[len(self.mask_files)//5*4:len(self.mask_files)//10*9]
        self.mask_files_test = self.mask_files[len(self.mask_files)//10*9 :]

        self.mode = mode

    def __len__(self):
        if self.mode == "train":
            return len(self.image_files_train)
        if self.mode == "val":
            return len(self.image_files_val)
        if self.mode == "test":
            return len(self.image_files_test)
            
    def __getitem__(self, idx):
        if self.mode == "train":
            itemfile=self.image_files_train[idx]
            item=cv2.imread(itemfile)
            label= cv2.imread(self.mask_files_train[idx], cv2.IMREAD_GRAYSCALE)
            label = np.expand_dims(label, axis=0)
            item= self.swapdim (item)
            return torch.cuda.FloatTensor(item), torch.cuda.FloatTensor(label)

        if self.mode == "val":
            itemfile=self.image_files_val[idx]
            item=cv2.imread(itemfile)
            label= cv2.imread(self.image_files_val[idx], cv2.IMREAD_GRAYSCALE)
            label = np.expand_dims(label, axis=0)
            item= self.swapdim (item)
            return torch.cuda.FloatTensor(item), torch.cuda.FloatTensor(label)

        if self.mode == "test":
            itemfile=self.image_files_test[idx]
            item=cv2.imread(itemfile)
            label= cv2.imread(self.image_files_test[idx], cv2.IMREAD_GRAYSCALE)
            label = np.expand_dims(label, axis=0)
            item= self.swapdim (item)
            return torch.cuda.FloatTensor(item), torch.cuda.FloatTensor(label)
            
    def swapdim(self, image):
        new_image = np.moveaxis(image, [-1, 0], [0, 1])
        return new_image


image_dir= "/home/zh340/github/210829_week1/UNet_with_PL/data/Image/*"
mask_dir= "/home/zh340/github/210829_week1/UNet_with_PL/data/mask/*"

training_dataset = CrackDataset(image_dir, mask_dir)
training_loader = DataLoader(training_dataset, batch_size=20)


first_data= training_dataset[0]
img, mask = first_data
print (img.shape, mask.shape)