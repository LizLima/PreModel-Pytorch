# Model pretrained 
# Model   : VGG19n
# Dataser : LFW
# epoch   : 200
# z dim   : 256

import utils as utils
import modelv2 as modelGen
import discriminator as modelDis
import Datasets.dataCPF as datacpfs

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
import pickle
from tqdm.autonotebook import tqdm
import os

device = torch.device('cpu')


################################################
# CONFIG 
################################################

laten_sapce = 256
lr          = 0.001
num_epochs  = 200
batch_size  = 64
image_size  = 128
print_epoch = 25

################################################
# DATASET 
################################################
path = '/home/liz/Documents/Data/VGGFace'
path_lfw = "/content/gdrive/My Drive/Maestria/Face/lfw"
path_cpf = "/home/liz/Documents/Data/cfp-dataset/Data/"
path_result= '/media/liz/Files/Model-Pretrained/GAN_64batch'
# Create the dataset
# train_dataset = datasets.ImageFolder(root=path + '/vggface2_train/train',
#                            transform=transforms.Compose([
#                                transforms.Resize(image_size),
#                                transforms.CenterCrop(image_size),
#                                transforms.ToTensor(),
#                            ]))

# test_dataset = datasets.ImageFolder(root=path + '/vggface2_test/test',
#                            transform=transforms.Compose([
#                                transforms.Resize(image_size),
#                                transforms.CenterCrop(image_size),
#                                transforms.ToTensor(),
#                            ]))                         
# Create the dataloader
data = datacpfs.DataSetTrain(path_cpf, isPatch="none", factor=0)

train_size = int(0.7 * len(data))
test_size = len(data) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Plot some training images


################################################
# MODEL 
################################################
# config VGGface
# class_train = train_dataset.classes
# class_test  = test_dataset.classes
# num_classes = len(class_train) + len(class_test)
class_data = data.classes

num_classes = len(class_data)
model       = modelGen.Generator(num_classes).to(device)
model_dis   = modelDis.Discriminator().to(device)
model_dis.apply(utils.weights_init)

optimizer   = optim.Adam(model.parameters(), lr=lr)
optimizer_d = optim.Adam(model_dis.parameters(), lr=lr)

lossMse     = nn.MSELoss()
lossBCE     = nn.BCELoss()

################################################
# CONFIGURATION MODEL 
################################################
flag = True
start_epoch = 0
if flag:
  model, optimizer, start_epoch = utils.load_checkpoint(model, optimizer, "/media/liz/Files/Model-Pretrained/GAN_64batch/checkpoint199.pth.tar")


################################################
# TRAIN 
################################################

def train(epoch):
    model.train()
    trainG_loss = 0
    trainD_loss = 0
    progress = tqdm(enumerate(trainloader), desc="Train", total=len(trainloader))
    for x in progress:
        data    = x[1]
        image   = data[0].to(device)
        label       = data[1].to(device)
        
        ######################
        # Train discriminator
        ######################
        optimizer_d.zero_grad()

        d_real      = model_dis(image)
        l_real      = torch.ones(batch_size).to(device)
        e_real      = lossBCE(d_real, l_real)

        # Generate image
        image_synt  = model(image)
        d_syn       = model_dis(image_syn)
        l_syn       = torch.zeros(batch_size).to(device)
        e_syn       = lossBCE(d_syn, lg_real)

        error_dis   = e_real + e_syn
        error_dis.backward()
        optimizer_d.step()

        ######################
        # Train generator
        ######################
        optimizer.zero_grad()
        image_synt  = model(image)
        dg_syn      = model_dis(image_size)
        lg_real     = torch.ones(batch_size).to(device)
        eg_syn      = lossBCE(dg_syn, lg_real)

        # Generator error
        error_gen   = eg_syn
        error_gen.backward()
        optimizer.step()
        
        trainG_loss += error_gen.item()
        trainD_loss += error_dis.item()

        progress.set_description("Epoch: %d, G: %.3f G: %.3f  " % (epoch, error_dis.item(), error_gen.item()))
    
    # Save model
    # if (epoch + 1) % print_epoch == 0:

    #     # Save the model
    #     # https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/2
    #     state = {   
    #                 'epoch': epoch + 1, 
    #                 'state_dict_encoder': model.encoder.state_dict(),
    #                 'state_dict_decoder': model.decoder.state_dict(),
    #                 'optimizer': optimizer.state_dict()
    #             }
    #     torch.save(state, path_result + "/resnet50_checkpoint" + str(epoch) +"_" + ".pth.tar" ) 

    return trainG_loss, trainD_loss
################################################
# TEST 
################################################
def test(epoch):
    model.eval()
    model_dis.eval()
    testG_loss   = 0
    testD_loss   = 0
    image       = None
    image_synt  = None
    # progress = tqdm(enumerate(testloader), desc="Train", total=len(testloader))
    with torch.no_grad():
        for x in enumerate(testloader):
            data    = x[1]
            image   = data[0].to(device)

            ######################
            # Train discriminator
            ######################
            
            d_real      = model_dis(image)
            l_real      = torch.ones(batch_size).to(device)
            e_real      = lossBCE(d_real, l_real)
            # Generate image
            image_synt  = model(image)
            d_syn       = model_dis(image_syn)
            l_syn       = torch.zeros(batch_size).to(device)
            e_syn       = lossBCE(d_syn, lg_)

            error_dis   = e_real + e_syn
            
            ######################
            # Train generator
            ######################
            image_synt  = model(image)
            dg_syn      = model_dis(image_size)
            lg_real     = torch.ones(batch_size).to(device)
            eg_syn      = lossBCE(dg_syn, lg_real)

            # Generator error
            error_gen   = eg_syn
            
            testG_loss += error_gen.item()
            testD_loss += error_dis.item()

            # progress.set_description("Test epoch: %d, MSE: %.5f , CE: %.5f , T: %.5f  " % (epoch, loss_mse, loss_ce, train_loss))
        if (epoch + 1) % print_epoch == 0:
          vutils.save_image(image_synt.data, path_result + '/resnet50_synt_%03d.jpg' % epoch, normalize=True)
          vutils.save_image(image.data, path_result + '/resnet50_input_%03d.jpg' % epoch, normalize=True)
    return testG_loss, testD_loss


best_test_loss = float('inf')
patience_counter = 0

loss_Train = []
loss_Test = []
for e in range(num_epochs):

    trainG_loss, trainD_loss = train(e)
    testG_loss, testD_loss = test(e)

    trainG_loss /= len(trainloader)
    trainD_loss /= len(trainloader)

    testG_loss /= len(testloader)
    testD_loss /= len(testloader)

    loss_Train.append((trainG_loss, trainD_loss))
    loss_Test.append((testG_loss, testD_loss))

fichero = open(path_result + '/files_vgg19_train.pckl', 'wb')
pickle.dump(loss_Train, fichero)
fichero.close()
fichero = open(path_result + '/files_vgg19_test.pckl', 'wb')
pickle.dump(loss_Test, fichero)
fichero.close()