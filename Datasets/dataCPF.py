import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image 
from random import shuffle
import Datasets.FaceLadmark.landmark as Landmark
import cv2 as cv2
import numpy as np
from math import exp
import matplotlib.pyplot as plt



class DataSetTrain(Dataset):

    def __init__(self, data_root,isPatch="None", factor=0):
        # super(self).__init__()
        self.data_root = data_root 
        self.classes = []
        self.isPatch = isPatch
        self.imgs = []
        self.totensor=transforms.Compose([
                            transforms.Resize(size=(128, 128)),
                            # transforms.CenterCrop(128),
                            transforms.ToTensor(),
                            # transforms.Normalize([0.5], [0.5]),
                        ])
        self.tobasictensor=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.toTensorEye=transforms.Compose([
                            transforms.Resize(size=(40, 40)),
                            transforms.ToTensor(),
                            # transforms.Normalize([0.5], [0.5]),
                        ])
        self.toTensorNose=transforms.Compose([
                            transforms.Resize(size=(40, 32)),
                            transforms.ToTensor(),
                            # transforms.Normalize([0.5], [0.5]),
                        ])
        self.toTensorMouth=transforms.Compose([
                            transforms.Resize(size=(32, 48)),
                            transforms.ToTensor(),
                            # transforms.Normalize([0.5], [0.5]),
                        ])
        self.scaledGaussian = lambda x : exp(-(1/8)*(x**2))
        
        self.__classes__()

        self.get_patch = Landmark.Landmark(factor)
        for register in os.listdir(self.data_root + "Images/"):
            # print(register)
            register_folder = os.path.join(self.data_root + "Images/", register)

            # Get frontal image
            frontal_file = os.path.join(register_folder, 'frontal')
            profile_file = os.path.join(register_folder, 'profile')
            
            for pfile in os.listdir(profile_file):
                file_profile = os.path.join(profile_file, pfile)

                # shuffle(name_frontals)
                # file_frontal = name_frontals[0]
                file_frontal = frontal_file + "/01.jpg"
                self.imgs.append((register, file_profile, file_frontal))


    def __len__(self):
        return len(self.imgs)

    def __classes__(self):

        # Load txt
        with open(self.data_root + "list_name.txt", 'r') as fp:
            line = fp.readline()
            line = fp.readline() # not first line
            while line:
                self.classes.append(line.strip())
                line = fp.readline()

    def __getitem__(self, index):
        item = {}
        data = self.imgs[index]
        data_profile = Image.open(data[1]).convert(mode='RGB')
        data_frontal = Image.open(data[2]).convert(mode='RGB')

        # Dictionary
        item["name"] = data[0]
        item["label"] = int(data[1].split('/')[-3])
        # print(data[1].split('/')[-3], item["label"])
        item["profile"] = self.totensor(data_profile)
        item["profile32"] = self.tobasictensor(data_profile.resize((32,32), Image.ANTIALIAS))
        item["profile64"] = self.tobasictensor(data_profile.resize((64,64), Image.ANTIALIAS))
        item["frontal"] = self.totensor(data_frontal)
        item["frontal32"] = self.tobasictensor(data_frontal.resize((32,32), Image.ANTIALIAS))
        item["frontal64"] = self.tobasictensor(data_frontal.resize((64,64), Image.ANTIALIAS))

        # Change for switch patch/only point as capgan/ all points 
        
        if self.isPatch == "patch":
           
            cv_profile = cv2.cvtColor(np.array(data_profile), cv2.COLOR_RGB2BGR)
            cv_frontal = cv2.cvtColor(np.array(data_frontal), cv2.COLOR_RGB2BGR)
            leye_profile, reye_profile, nose_profile, mouth_profile = self.get_patch.getPatches(cv_profile)
            leye_frontal, reye_frontal, nose_frontal, mouth_frontal = self.get_patch.getPatches(cv_frontal)
            #a = torch.from_numpy(leye_frontal.transpose(2, 1 , 0))
            #b = self.toTensorEye(Image.fromarray(leye_frontal))
            # item["leye_p"] = self.toTensorEye(Image.fromarray(leye_profile))
            # item["reye_p"] = self.toTensorEye(Image.fromarray(reye_profile))
            # item["nose_p"] = self.toTensorNose(Image.fromarray(nose_profile))
            # item["mouth_p"] = self.toTensorMouth(Image.fromarray(mouth_profile))
            item["leye_p"] = self.toTensorEye(Image.fromarray(leye_profile))
            item["reye_p"] = self.toTensorEye(Image.fromarray(reye_profile))
            item["nose_p"] = self.toTensorNose(Image.fromarray(nose_profile))
            item["mouth_p"] = self.toTensorMouth(Image.fromarray(mouth_profile))
            item["leye_f"] = self.toTensorEye(Image.fromarray(leye_frontal))
            item["reye_f"] = self.toTensorEye(Image.fromarray(reye_frontal))
            item["nose_f"] = self.toTensorNose(Image.fromarray(nose_frontal))
            item["mouth_f"] = self.toTensorMouth(Image.fromarray(mouth_frontal))
        
        if self.isPatch == "all-frontal":
            width = 128
            height = 128
            # Convert array to opencv
            cv_frontal = (item["frontal"].permute(1,2,0).numpy()*255).astype(np.uint8)
           
            # Get 5 coord lefteye, right eye, nose, left mouth , right mouth
            listCoord = self.get_patch.getAllPointFace(cv_frontal)
            item["frontal"] = self.create_heap(listCoord, width, height)

        return item

    def create_heap(self, listCoord, width,height):
        img_heapmap = np.zeros((width,height),np.uint8)
        heap = None
        flag = 0
        for point in listCoord:
            if point is not None:
                x, y = point
                for i in range(width):
                    for j in range(height):
                        distancePoint = np.linalg.norm(np.array([i-y,j-x]))
                        scaledGaussianProb = self.scaledGaussian(distancePoint)
                        img_heapmap[i,j] = np.clip(scaledGaussianProb*255,0,255)
                if flag == 0:
                    heap = torch.Tensor(img_heapmap)
                    heap = heap.unsqueeze(0)
                    flag = 1
                else:
                    tensor = torch.Tensor(img_heapmap)
                    tensor = tensor.unsqueeze(0)
                    heap = torch.cat([heap, tensor], dim=0)
        if heap is None:
            heap = torch.zeros(5, 128, 128)
        return heap

# # Example
# import torch
# data = DataSetTrain('/home/liz/Documents/Data/cfp-dataset/Data/Images/', isPatch="all-frontal")

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

