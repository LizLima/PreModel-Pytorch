import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import torchvision.utils as vutils
import pickle
import torchvision.models as models

device = torch.device('cpu')
################################################
# DATASET 
################################################
path = '/media/liz/Files/data/VGGface'
path_lfw = "/home/liz/Documents/Data/lfw"
path_result= '/home/liz/Documents/Model-Pretrained/results'
# Create the dataset
train_dataset = datasets.ImageFolder(root=path + '/vggface2_train/train',
                           transform=transforms.Compose([
                               transforms.Resize(128),
                               transforms.CenterCrop(128),
                               transforms.ToTensor(),
                           ]))

test_dataset = datasets.ImageFolder(root=path + '/vggface2_test/test',
                           transform=transforms.Compose([
                               transforms.Resize(128),
                               transforms.CenterCrop(128),
                               transforms.ToTensor(),
                           ]))                         

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
testloader  = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

# config VGGface
class_train = train_dataset.classes
class_test  = test_dataset.classes
num_classes = len(class_train) + len(class_test)
# Plot some training images
real_batch = next(iter(trainloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()

model = models.vgg19(pretrained=True)
num_ftrs = model.classifier[6].in_features
# convert all the layers to list and remove the last one
features = list(model.classifier.children())[:-1]
## Add the last layer based on the num of classes in our dataset
features.extend([nn.Linear(num_ftrs, len(class_train))])
## convert it into container and add it to our model class.
model.classifier = nn.Sequential(*features)

loss = nn.CrossEntropyLoss()
# Freeze training for all layers
for param in model.features.parameters():
    param.require_grad = False

model.to(device)

for i, x in enumerate(trainloader):

    print("data")
    data    = x[0].to(device)
    label   = x[1].to(device)

    output = model(data)

    # calculate rank 1
    dist = loss(output, label)

    # dictionarry
    result = {} # id = value / value classid

    for idclass, clasdes in enumerate(class_train):
        print("")
        result
    
    for idclass, value in enumerate(output[0]):
        result[value] = idclass


    

