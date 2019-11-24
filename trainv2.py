# Model pretrained 
# Model   : VGG1919n
# Dataser : LFW
# epoch   : 200
# z dim   : 256
# Type    : GAN
# Loss    : MinMax
import utils as utils
import Models.modelv2 as modelGen
import Models.discriminator as modelDis
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
from torchvision import datasets, transforms

device = torch.device('cuda')


################################################
# CONFIG 
################################################

laten_sapce = 256
lr          = 0.0002
lr_d        = 0.0002
num_epochs  = 500
batch_size  = 32
image_size  = 128
print_epoch = 5

################################################
# DATASET 
################################################

path_lfw        = "/home/liz/Documents/Data/lfw"
path_cpf        = "/home/liz/Documents/Data/cfp-dataset/Data/"
path_result     = "/media/liz/Files/Model-Pretrained/GAN_64batch"
path_pretrained = "/media/liz/Files/Model-Pretrained/PreTrained_VGG19bn_b64_lfw/vgg19_checkpoint199.pth.tar"
name_checkpoint = "vgg19_gan_checkpoint"                      
# Create the dataloader CPF
data = datacpfs.DataSetTrain(path_cpf, isPatch="none", factor=0)

# Create the dataloader LFW
# data = datasets.ImageFolder(root=path_lfw,
#                            transform=transforms.Compose([
#                                transforms.Resize(image_size),
#                                transforms.CenterCrop(image_size),
#                                transforms.ToTensor(),
#                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                            ]))

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
optimizer_d = optim.Adam(model_dis.parameters(), lr=lr_d)

criterion = nn.BCELoss()
pixel_wise = nn.L1Loss(reduction='mean')
################################################
# CONFIGURATION MODEL 
################################################
flag = True
start_epoch = 0
if flag:
  model, optimizer, start_epoch = utils.load_checkpoint(model, optimizer, path_pretrained)


################################################
# TRAIN 
################################################

def train(epoch):
    model.train()
    trainG_loss = 0
    trainD_loss = 0
    progress = tqdm(enumerate(trainloader), desc="Train", total=len(trainloader))
    for x in progress:
        data = x[1]
        image   = data['profile'].to(device)
        frontal   = data['frontal'].to(device)
        # image = data[0].to(device)
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        model_dis.zero_grad()
        b_size = image.size(0)
        label = torch.ones(b_size, 1, 1, 1).to(device)
        output = model_dis(frontal).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward(retain_graph=True)
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate fake image batch with G
        fake = model(image)
        label.fill_(0)
        output = model_dis(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward(retain_graph=True)
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        # Update D
        optimizer_d.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        model.zero_grad()
        label.fill_(1)  # fake labels are real for generator cost
        output = model_dis(fake).view(-1)
        errG_g = criterion(output, label)
        errG_g.backward(retain_graph=True)
        D_G_z2 = output.mean().item()

        # Pixel wise loss
        errG_wise = 10*pixel_wise(fake, frontal)
        errG_wise.backward()
        errG = errG_g + errG_wise
        # Update G
        optimizer.step()



        progress.set_description("Epoch: %d, G: %.3f D: %.3f  " % (epoch, errG.item(), errD.item()))
    
    # if (epoch + 1) % print_epoch == 0:
    #     vutils.save_image(fake.data, path_result + '/synt_%03d.jpg' % epoch, normalize=True)
    #     vutils.save_image(image.data, path_result + '/input_%03d.jpg' % epoch, normalize=True)
    # Save model
    if (epoch + 1) % print_epoch == 0:

        # Save the model
        # https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/2
        state = {   
                    'epoch': epoch + 1, 
                    'state_dict_disc': model_dis.state_dict(),
                    'state_dict_gen': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'optimizer_dis': optimizer_d.state_dict()
                }
        torch.save(state, path_result + "/" + name_checkpoint + str(epoch) + ".pth.tar" ) 

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
            
            data = x[1]
            # image   = data['profile'].to(device)
            image = data[0].to(device)

            ######################
            # Train discriminator
            ######################
            
            d_real      = model_dis(image)
            l_real      = torch.ones(d_real.size(0), 1, 1, 1).to(device)
            e_real      = lossBCE(d_real, l_real)
            # Generate image
            image_synt  = model(image)
            d_syn       = model_dis(image_synt)
            l_syn       = torch.zeros(d_syn.size(0), 1, 1, 1).to(device)
            e_syn       = lossBCE(d_syn, l_syn)

            errD   = e_real + e_syn
            
            ######################
            # Train generator
            ######################
            dg_syn      = model_dis(image_synt)
            lg_real     = torch.ones(dg_syn.size(0), 1, 1, 1).to(device)
            errG_g      = lossBCE(dg_syn, lg_real)

            # Generator error

            # Pixel wise loss
            errG_wise = 10*pixel_wise(fake, frontal)
            errG = errG_g + errG_wise

            testG_loss += errG.item()
            testD_loss += error_dis.item()

            progress.set_description("Epoch: %d, G: %.3f D: %.3f  " % (epoch, errG.item(), errD.item()))
        if (epoch + 1) % print_epoch == 0:
            vutils.save_image(fake.data, path_result + '/synt_%03d.jpg' % epoch, normalize=True)
            vutils.save_image(image.data, path_result + '/input_%03d.jpg' % epoch, normalize=True)

    return testG_loss, testD_loss


best_test_loss = float('inf')
patience_counter = 0

loss_Train = []
loss_Test = []

loss_Train.append(("trainG_loss", "trainD_loss"))
loss_Test.append(("testG_loss", "testD_loss"))
for e in range(num_epochs):

    trainG_loss, trainD_loss = train(e)
    testG_loss, testD_loss = test(e)

    trainG_loss /= len(trainloader)
    trainD_loss /= len(trainloader)

    testG_loss /= len(testloader)
    testD_loss /= len(testloader)

    loss_Train.append((trainG_loss, trainD_loss))
    loss_Test.append((testG_loss, testD_loss))

fichero = open(path_result + '/files_gan_train.pckl', 'wb')
pickle.dump(loss_Train, fichero)
fichero.close()
fichero = open(path_result + '/files_gan_test.pckl', 'wb')
pickle.dump(loss_Test, fichero)
fichero.close()