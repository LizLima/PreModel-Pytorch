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
lr_d        = 0.0001
num_epochs  = 500
batch_size  = 32
image_size  = 128
print_epoch = 25

################################################
# DATASET 
################################################

path_lfw        = "/home/liz/Documents/Data/lfw"
path_cpf        = "/home/liz/Documents/Data/cfp-dataset/Data/"
path_result     = "/media/liz/Files/Model-Pretrained/GAN_64batch"
# path_pretrained = "/media/liz/Files/Model-Pretrained/PreTrained_VGG19bn_b64_lfw/vgg19_checkpoint199.pth.tar"
path_pretrained = '/media/liz/Files/Model-Pretrained/resnet50_b128_vggface/resnet_checkpoint3_.pth.tar'
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
model_G       = modelGen.Generator(num_classes).to(device)
model_D       = modelDis.Discriminator().to(device)
model_D.apply(utils.weights_init)

optimizer_G = optim.Adam(model_G.parameters(), lr=lr)
optimizer_D = optim.Adam(model_D.parameters(), lr=lr_d)

criterion   = nn.BCELoss()
loss_pix  = nn.L1Loss(reduction='mean')

################################################
# CONFIGURATION MODEL 
################################################
flag = False
start_epoch = 0
if flag:
  model_G, optimizer_G, start_epoch = utils.load_checkpoint(model_G, optimizer_G, path_pretrained)


################################################
# TRAIN 
################################################

def train(epoch):

    model_G.train()
    model_D.train()

    trainG_loss = 0
    trainD_loss = 0

    image       = None
    fake        = None

    progress = tqdm(enumerate(trainloader), desc="Train", total=len(trainloader))
    for x in progress:

        data = x[1]
        image   = data['profile'].to(device)
        frontal   = data['frontal'].to(device)
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        optimizer_D.zero_grad()

        label = torch.ones(image.size(0), 1, 1, 1).to(device)
        output = model_D(frontal)
        errD_real = criterion(output, label)
        errD_real.backward()
       
        ## Train with fake image
        fake = model_G(image)
        label.fill_(0)
        output = model_D(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        
        errD = errD_real + errD_fake
        optimizer_D.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        optimizer_G.zero_grad()
        label.fill_(1) 
        output = model_D(fake)
        err_bce = criterion(output, label)
        # err_bce.backward(retain_graph=True)

        err_pix = loss_pix(fake, frontal)
        # err_pix.backward(retain_graph=True)

        errG = err_bce + err_pix
        errG.backward()
        optimizer_G.step()
        
        trainG_loss += errG.item()
        trainD_loss += errD.item()


        progress.set_description("Epoch: %d, G: %.3f D: %.3f  " % (epoch, errG.item(), errD.item()))
    
    if (epoch + 1) % print_epoch == 0:
        vutils.save_image(fake.data, path_result + '/train/synt_%03d.jpg' % epoch, normalize=True)
        vutils.save_image(image.data, path_result + '/train/input_%03d.jpg' % epoch, normalize=True)
    
    # Save model
    if (epoch + 1) % print_epoch == 0:

        # Save the model
        # https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/2
        state = {   
                    'epoch': epoch + 1, 
                    'state_dict_D': model_D.state_dict(),
                    'state_dict_G': model_G.state_dict(),
                    'optimizer_G': optimizer_G.state_dict(),
                    'optimizer_D': optimizer_D.state_dict()
                }
        torch.save(state, path_result + "/" + name_checkpoint + str(epoch) + ".pth.tar" ) 

    return trainG_loss, trainD_loss

################################################
# TEST 
################################################
def test(epoch):

    model_G.eval()
    model_D.eval()

    testG_loss   = 0
    testD_loss   = 0

    image       = None
    fake        = None

    progress = tqdm(enumerate(testloader), desc="Train", total=len(testloader))

    with torch.no_grad():
        for x in progress:
            
            data = x[1]
            image   = data['profile'].to(device)
            frontal = data['frontal'].to(device)

            # DISCRIMINATOR
            label       = torch.ones(image.size(0), 1, 1, 1).to(device)
            output      = model_D(frontal)
            errD_real   = criterion(output, label)
           
            # Generate fake image
            fake = model_G(image)
            label.fill_(0)
            output      = model_D(fake.detach())
            errD_fake   = criterion(output, label)
           
            errD = errD_real + errD_fake
            
            # GENERATOR
            label.fill_(1)  # fake labels are real for generator cost
            output = model_D(fake)
            err_bce = criterion(output, label)
            err_pix = loss_pix(fake, frontal)

            errG = err_bce + err_pix
            
            testG_loss += errG.item()
            testD_loss += errD.item()

            progress.set_description("Test: %d, G: %.3f D: %.3f  " % (epoch, errG.item(), errD.item()))

        if (epoch + 1) % print_epoch == 0:
            vutils.save_image(fake.data, path_result + '/test/synt_%03d.jpg' % epoch, normalize=True)
            vutils.save_image(image.data, path_result + '/test/input_%03d.jpg' % epoch, normalize=True)

    return testG_loss, testD_loss



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

    if (e + 1) % print_epoch == 0:
        fichero = open(path_result + '/files_gan_train_' + str(e) + '.pckl', 'wb')
        pickle.dump(loss_Train, fichero)
        fichero.close()
        fichero = open(path_result + '/files_gan_test_' + str(e) + '.pckl', 'wb')
        pickle.dump(loss_Test, fichero)
        fichero.close()