import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image 
from random import shuffle
import landmark as Landmark
import cv2 as cv2
import numpy as np
from alignment.align import AlignDlib

class DataSetTrain(Dataset):

    def __init__(self, data_root,isPatch=False, factor=0):
        # super(self).__init__()
        self.isPatch = isPatch
        self.imgs = []
        self.totensor=transforms.Compose([
                            transforms.Resize(size=(128, 128)),
                            #transforms.CenterCrop(128),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5]),
                        ])
        self.toTensorEye=transforms.Compose([
                            transforms.Resize(size=(40, 40)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5]),
                        ])
        self.toTensorNose=transforms.Compose([
                            transforms.Resize(size=(40, 32)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5]),
                        ])
        self.toTensorMouth=transforms.Compose([
                            transforms.Resize(size=(32, 48)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5]),
                        ])

        self.alignment = AlignDlib("alignment/landmarks.dat")
        # Return Patch
        self.get_patch = Landmark.Landmark(factor)
        
        for register in os.listdir(data_root):
            # print(register)
            img_profile = os.path.join(data_root, register)

            self.imgs.append((register, img_profile))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        item = {}
        data = self.imgs[index]
        data_profile = Image.open(data[1])

        item["name"] = data[0]
        item["profile"] = self.totensor(data_profile.convert(mode='RGB'))

        if self.isPatch:
           
            cv_profile = cv2.cvtColor(np.array(data_profile), cv2.COLOR_RGB2BGR)
            leye_profile, reye_profile, nose_profile, mouth_profile = self.get_patch.getPatches(cv_profile)
            
            item["leye_p"] = self.toTensorEye(Image.fromarray(leye_profile))
            item["reye_p"] = self.toTensorEye(Image.fromarray(reye_profile))
            item["nose_p"] = self.toTensorNose(Image.fromarray(nose_profile))
            item["mouth_p"] = self.toTensorMouth(Image.fromarray(mouth_profile))
           
        return item

# Example
# import torch
# data = DatasetTrain('/home/liz/Documents/Data/cfp-dataset/Data/Images/')

# dataloader = torch.utils.data.DataLoader( data, shuffle=True, batch_size=20)
# print(len(data))
# print(len(dataloader))
# f=None
# p=None
# for data in dataloader:
#    n,f,p = data 
#    break 

# print(f.shape)
# print(p.shape)