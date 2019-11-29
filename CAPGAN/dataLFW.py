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
                            #transforms.Resize(size=(128, 128)),
                            transforms.CenterCrop(128),
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
        self.tobasictensor=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # Return Patch
        self.get_patch = Landmark.Landmark(factor)
        # this dataset have list of name
        self.classes = os.listdir(data_root)
        self.dic_classes = {x:i for i,x in enumerate(self.classes)}

        for register in os.listdir(data_root):
            # print(register)
            register_folder = os.path.join(data_root, register)

            # Get frontal image
            frontal_file = register_folder
            name_frontals = []
            for pfile in os.listdir(frontal_file):
                path_frontal = os.path.join(frontal_file, pfile)
                name_frontals.append(path_frontal)

            ## Get profile image an to asociate it with frontal images
            profile_file = register_folder
            shuffle(name_frontals) #select a random but the same for all images

            for pfile in os.listdir(profile_file):
                file_profile = os.path.join(profile_file, pfile)

                file_frontal = name_frontals[0]
                self.imgs.append((register, file_profile, file_frontal))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        item = {}
        data = self.imgs[index]
        data_profile = Image.open(data[1])
        data_frontal = Image.open(data[2])

        item["name"] = data[0]
        item["label"] = self.dic_classes[data[0]]
        item["profile"] = self.totensor(data_profile.convert(mode='RGB'))
        item["frontal"] = self.totensor(data_frontal.convert(mode='RGB'))
        #item["frontal"] = self.totensor(Image.fromarray(data_frontal_align))
        item["frontal32"] = self.tobasictensor(data_frontal.resize((32,32), Image.ANTIALIAS))
        item["frontal64"] = self.tobasictensor(data_frontal.resize((64,64), Image.ANTIALIAS))

        if self.isPatch:
           
            cv_profile = cv2.cvtColor(np.array(data_profile), cv2.COLOR_RGB2BGR)
            cv_frontal = cv2.cvtColor(np.array(data_frontal), cv2.COLOR_RGB2BGR)
            leye_profile, reye_profile, nose_profile, mouth_profile = self.get_patch.getPatches(cv_profile)
            leye_frontal, reye_frontal, nose_frontal, mouth_frontal = self.get_patch.getPatches(cv_frontal)
            
            item["leye_p"] = self.toTensorEye(Image.fromarray(leye_profile))
            item["reye_p"] = self.toTensorEye(Image.fromarray(reye_profile))
            item["nose_p"] = self.toTensorNose(Image.fromarray(nose_profile))
            item["mouth_p"] = self.toTensorMouth(Image.fromarray(mouth_profile))
            item["leye_f"] = self.toTensorEye(Image.fromarray(leye_frontal))
            item["reye_f"] = self.toTensorEye(Image.fromarray(reye_frontal))
            item["nose_f"] = self.toTensorNose(Image.fromarray(nose_frontal))
            item["mouth_f"] = self.toTensorMouth(Image.fromarray(mouth_frontal))
        
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