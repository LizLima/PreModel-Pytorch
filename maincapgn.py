import torch
import torch.nn as nn
import pickle
import Models.capggan as model_cap

########################################################################
## TRAIN
########################################################################
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils

# Model
import Datasets.dataCPF as datasets
import utils.utils as utils
from tqdm.autonotebook import tqdm


################################################
# CONFIG 
################################################
lambda1 = 10
lambda2 = 0.1
lambda3 = 0.1
lambda4 = 0.02
lambda5 = 0.0001
lr      = 0.0002


num_epochs  = 100
batch_size  = 8
print_epoch = 5
load_checkpoint = True

################################################
# DATASET 
################################################
path = '/home/invitado/Documents/Liz/data/cfp-dataset/Data/'
path_result= '/home/invitado/Documents/Liz/test_capggan'
# Create the dataset
dataset = datasets.DataSetTrain(path, isPatch="heap-map", factor=0)

train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print(len(trainloader))
print(len(testloader))
# Plot some training images
real_batch = next(iter(trainloader))


################################################
# MODEL 
################################################
device = torch.device('cuda')
model_gen = model_cap.Generator().to(device)
model_dii = model_cap.Discriminator(6).to(device)
model_dpe = model_cap.Discriminator(8).to(device)

################################################
# COST FUNCTIONS 
################################################
loss_pixel = nn.L1Loss(reduction='sum')
loss_adv = nn.BCELoss()

params = list(model_gen.parameters()) + list(model_dii.parameters()) +\
         list(model_dpe.parameters())

optimizer = torch.optim.Adam(params, lr=lr)
###
# epoch = 11
# state = {'epoch': epoch + 1, 'state_dict': model_gen.state_dict(),
# 'optimizer': optimizer.state_dict()}
# torch.save(state, path_result + "/model/checkpoint_gen" + str(epoch) + ".pth.tar" )

# state = {'epoch': epoch + 1, 'state_dict': model_dii.state_dict(),
# 'optimizer': optimizer.state_dict()}
# torch.save(state, path_result + "/model/checkpoint_dii" + str(epoch) + ".pth.tar" )

# state = {'epoch': epoch + 1, 'state_dict': model_dpe.state_dict(),
# 'optimizer': optimizer.state_dict()}
# torch.save(state, path_result + "/model/checkpoint_dp" + str(epoch) + ".pth.tar" ) 
###
def train(epoch):
    model_gen.train()
    model_dii.train()
    model_dpe.train()
    value_loss  = 0

    progress = tqdm(enumerate(trainloader), desc="Train", total=len(trainloader))
    for i,x in progress:
        optimizer.zero_grad()

        source = x['profile'].to(device)
        source32 = x['profile32'].to(device)
        source64 = x['profile64'].to(device)
        target = x['frontal'].to(device)
        target32 = x["frontal32"].to(device)
        target64 = x["frontal64"].to(device)
        # Generate heap map
        pa_profile = x["hm_p"].to(device)
        pb_target = x["hm_f"].to(device)
        img128, img64, img32 = model_gen(source, pa_profile, pb_target, source64, source32)
        
        Lpix_128 = loss_pixel(img128, target) # 128*128*3 -> 0.000020345
        Lpix_64 = loss_pixel(img64, target64) # 64*64*3 -> 0,00008138
        Lpix_32 = loss_pixel(img32, target32) # 32*32*3 -> 0,000325521

        Lpix = (Lpix_128*0.00002 + Lpix_64*0.00008 + Lpix_32*0.00032)/3

        # Disc II
        output_t = model_dii(target, source)
        output_s = model_dii(img128, source)
        label_real = torch.ones(output_t.size()).to(device)
        label_fake = torch.zeros(output_s.size()).to(device)

        Lii_real = loss_adv(output_t, label_real)
        Lii_fake = loss_adv(output_s, label_fake)

        Lii = Lii_real + Lii_fake

        # Disc PE
        output_pt = model_dpe(target, pb_target)
        output_ps = model_dpe(img128, pb_target)
        label_real = torch.ones(output_t.size()).to(device)
        label_fake = torch.zeros(output_s.size()).to(device)

        Lpe_real = loss_adv(output_pt, label_real)
        Lpe_fake = loss_adv(output_ps, label_fake)

        Lpe = Lpe_real + Lpe_fake

        L = lambda1*Lpix + lambda2*Lii + lambda3*Lpe
        L.backward()
        optimizer.step()

        value_loss += L.item()

        progress.set_description("Epoch: %d, Loss: %.3f " % (epoch, L.item()))

    if (epoch + 1) % print_epoch == 0:
        state = {'epoch': epoch + 1, 'state_dict': model_gen.state_dict(),
        'optimizer': optimizer.state_dict()}
        torch.save(state, path_result + "/model/checkpoint_gen" + str(epoch) + ".pth.tar" )

        state = {'epoch': epoch + 1, 'state_dict': model_dii.state_dict(),
        'optimizer': optimizer.state_dict()}
        torch.save(state, path_result + "/model/checkpoint_dii" + str(epoch) + ".pth.tar" )

        state = {'epoch': epoch + 1, 'state_dict': model_dpe.state_dict(),
        'optimizer': optimizer.state_dict()}
        torch.save(state, path_result + "/model/checkpoint_dp" + str(epoch) + ".pth.tar" ) 
        
    return value_loss

def test(epoch):

    model_gen.eval()
    model_dii.eval()
    model_dpe.eval()
    value_loss  = 0
    
    progress = tqdm(enumerate(testloader), desc="Train", total=len(testloader))

    for i, x in progress:
        source = x['profile'].to(device)
        source32 = x['profile32'].to(device)
        source64 = x['profile64'].to(device)
        target = x['frontal'].to(device)
        target32 = x["frontal32"].to(device)
        target64 = x["frontal64"].to(device)
        # Generate heap map
        pa_profile = x["hm_p"].to(device)
        pb_target = x["hm_f"].to(device)
        img128, img64, img32 = model_gen(source, pa_profile, pb_target, source64, source32)
        
        Lpix_128 = loss_pixel(img128, target) # 128*128*3 -> 0.000020345
        Lpix_64 = loss_pixel(img64, target64) # 64*64*3 -> 0,00008138
        Lpix_32 = loss_pixel(img32, target32) # 32*32*3 -> 0,000325521

        Lpix = (Lpix_128*0.00002 + Lpix_64*0.00008 + Lpix_32*0.00032)/3

        # Disc II
        output_t = model_dii(target, source)
        output_s = model_dii(img128, source)
        label_real = torch.ones(output_t.size()).to(device)
        label_fake = torch.zeros(output_s.size()).to(device)

        Lii_real = loss_adv(output_t, label_real)
        Lii_fake = loss_adv(output_s, label_fake)

        Lii = Lii_real + Lii_fake

        # Disc PE
        output_pt = model_dpe(target, pb_target)
        output_ps = model_dpe(img128, pb_target)
        label_real = torch.ones(output_t.size()).to(device)
        label_fake = torch.zeros(output_s.size()).to(device)

        Lpe_real = loss_adv(output_pt, label_real)
        Lpe_fake = loss_adv(output_ps, label_fake)

        Lpe = Lpe_real + Lpe_fake

        L = lambda1*Lpix + lambda2*Lii + lambda3*Lpe
       
        value_loss += L.item()

        progress.set_description("Test: %d, Loss: %.3f  " % (epoch, L.item()))

    if (epoch + 1) % print_epoch == 0:

        vutils.save_image(source.data, path_result + '/test/input_%03d.jpg' % epoch, normalize=True)
        vutils.save_image(target.data, path_result + '/test/frontal_%03d.jpg' % epoch, normalize=True)
        vutils.save_image(img128.data, path_result + '/test/gen_%03d.jpg' % epoch, normalize=True)
        
    return value_loss

Loss_train = []
Loss_test = []

start_epochs = 0
# TODO
def load_model(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    last_epoch = checkpoint['epoch']
    
    return model, optimizer, last_epoch


if(load_checkpoint == True):

    path_chck = path_result + "/model/checkpoint_"
    epoch = 14
    model_gen, optimizer, last_epoch = load_model(path_chck + "gen" + str(epoch) + ".pth.tar",model_gen, optimizer)
    model_dii, optimizer, last_epoch = load_model(path_chck + "dii" + str(epoch) + ".pth.tar",model_dii, optimizer)
    model_dpe, optimizer, last_epoch = load_model(path_chck + "dp" + str(epoch) + ".pth.tar",model_dpe, optimizer)
    
    start_epochs = last_epoch


for e in range(start_epochs, num_epochs):
    train_loss = train(e)
    print("Train : " + str(e) + "Loss: " + str(train_loss/len(trainloader)))
    test_loss = test(e)
    print("Test : " + str(e) + "Loss: " + str(test_loss/len(testloader)))

    Loss_train.append(train_loss/len(trainloader))
    Loss_test.append(test_loss/len(testloader))
    
    if(e + 1) % print_epoch == 0:
      #  x = np.arange(e + 1)
       # fig = plt.figure()
       # ax = plt.subplot(111)
      #  ax.plot(x, Loss_train, 'mediumvioletred', label='Generator Training')
      #  ax.plot(x, Loss_test, 'pink', label='Generator Test')

        # ax.plot(x, Loss_Disc, 'steelblue', label='Discriminator Training')
        # ax.plot(x, Loss_Disc_Test, 'lightskyblue', label='Discriminator Test')

      #  plt.title('Function loss')
      #  ax.legend()
      #  fig.savefig(path_result + '/plot' + str(e) + '.png')
        # plt.show()
       # plt.close(fig)

        # Save results loss
        fichero = open(path_result + '/files_gan_train_' + str(e) + '.pckl', 'wb')
        pickle.dump(Loss_train, fichero)
        fichero.close()
        fichero = open(path_result + '/files_gan_test_' + str(e) + '.pckl', 'wb')
        pickle.dump(Loss_test, fichero)
        fichero.close()
