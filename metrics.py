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
import Datasets.dataCPF as datacpfs

device = torch.device('cpu')
################################################
# DATASET 
################################################
path_cpf = "/home/liz/Documents/Data/cfp-dataset/Data/"

# Create the dataset
data = datacpfs.DataSetTrain(path_cpf, isPatch="none", factor=0)

train_size = int(0.7 * len(data))
test_size = len(data) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
testloader  = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

# config VGGface
list_class = data.classes
num_classes = len(list_class)


model = models.vgg19(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Sequential(
                      nn.Linear(num_ftrs, 256), 
                      nn.ReLU(), 
                      nn.Dropout(0.4),
                      nn.Linear(256, num_classes),                   
                      nn.Softmax(dim=1))

loss = nn.CrossEntropyLoss()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 250
for epoch in range(epochs):

    train_loss  = 0
    count_true  = 0
    count_total = 0
    for i, x in enumerate(trainloader):

        data    = x['frontal'].to(device)
        label   = x['label'].to(device)
        name    = x['name']
        optimizer.zero_grad()
        output      = model(data)
        value_loss  = loss(output, label)
        value_loss.backward()
        optimizer.step()

        train_loss += value_loss.item()

        # Calculate rank for each image
        results     = {}
        # calculate rank 1
        output      = model(data[0].unsqueeze(0))
        for idx, n_class in enumerate(list_class):

            # Calculate loss class idx and ouput
            lbl = torch.tensor([idx])
            value = loss(output, lbl).item()
            results[value] = n_class
        
        list_items = []
        for key in results:
            list_items.append((key, results[key]))
        list_items = sorted(list_items)

        if list_class[label[0]]==list_items[0][1]:
            count_true +=1
        count_total += 1
    
    print("Epoch: %d, Ce: %.5f " % (epoch, train_loss/len(trainloader)) )
    print("Example ")
    print("Real label : ", list_class[label[0]], "\tPred. label : " , list_items[0][1])
    print("True: ", count_true,  "Total: ", count_total)
    # print("Real value : ", value_loss.item(), "\tPred. value : ", list_items[0][0])
    print("-----------------------------------------------------------------------------------")
    # plt.imshow(data[0].permute(1,2,0))




    

