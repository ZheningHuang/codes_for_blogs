import torch
from torch.utils.data import Dataset, DataLoader
import cv2

from glob import glob

class CrackImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        image_files=sorted(glob(image_dir))
        mask_files=sorted(glob(mask_dir))
        self.image_files=image_files
        self.mask_files=mask_files
        
    def __len__(self):
        return len(self.mask_files)
    def __getitem__(self,idx):
        itemfile=self.image_files[idx]
        item=cv2.imread(itemfile)
        label= cv2.imread(self.mask_files[idx])
        return item, label 

image_dir="./BenchStudy_Dataset/SegmentationMask/*"
mask_dir="./BenchStudy_Dataset/RawImage/*"

dataset=CrackImageDataset(image_dir,mask_dir)
first_data= dataset[0]
image,label= first_data
cv2.imshow("image",image)
cv2.imshow("label",label)
cv2.waitKey(0)

