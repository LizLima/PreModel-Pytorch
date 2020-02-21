import torch
import torch.nn as nn
import utils.residual as residual
import utils.maxout as maxout

class Encoder(nn.Module):
    # pass
    def __init__(self):
        super().__init__()
        self.bias = False
        # ENCODER
        self.conv0 =  nn.Conv2d(in_channels=13, out_channels=64, kernel_size=7, stride=1, padding=3, bias=self.bias)
        self.bn0 = nn.BatchNorm2d(64)
        self.r_conv0 = residual.BasicBlock(64, 64)

        self.conv1 =  nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2, bias=self.bias)
        self.bn1 = nn.BatchNorm2d(64)
        self.r_conv1 = residual.BasicBlock(64, 64)

        self.conv2 =  nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=self.bias)
        self.bn2 = nn.BatchNorm2d(128)
        self.r_conv2 = residual.BasicBlock(128, 128)

        self.conv3 =  nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=self.bias)
        self.bn3 = nn.BatchNorm2d(256)
        self.r_conv3 = residual.BasicBlock(256, 256)

        self.conv4 =  nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=self.bias)
        self.bn4 = nn.BatchNorm2d(512)
        self.r_conv4 = residual.BasicBlock(512, 512)
        

        # Flatten
        self.fc1 = nn.Linear(in_features=32768, out_features=512)
        self.bnfc1 = nn.BatchNorm1d(num_features=512)

        self.maxout = maxout.Maxout(512, 256, 64)

        self.relu = nn.ReLU()

    def forward(self, ia, pa, pb):

        input = torch.cat([ia, pa, pb], dim=1)
        conv0 = self.relu(self.bn0(self.conv0(input)))
        r_conv0 = self.r_conv0(conv0)

        conv1 = self.relu(self.bn1(self.conv1(r_conv0)))
        r_conv1 = self.r_conv1(conv1)

        conv2 = self.relu(self.bn2(self.conv2(r_conv1)))
        r_conv2 = self.r_conv2(conv2)

        conv3 = self.relu(self.bn3(self.conv3(r_conv2)))
        r_conv3 = self.r_conv3(conv3)

        conv4 = self.relu(self.bn4(self.conv4(r_conv3)))
        r_conv4 = self.r_conv4(conv4)

        # Flatten
        fc1 = self.fc1(r_conv4.view(r_conv4.size(0), -1))

        # Maxout
        ls_vector = self.maxout(fc1) # return 256


        return ls_vector, r_conv4, r_conv3, r_conv2, r_conv1, r_conv0

class Decoder(nn.Module):
    # pass
    def __init__(self):
        super().__init__()
        self.bias = False
        # DECODER
        self.fc2 = nn.Linear(in_features=256, out_features=4096)
        self.bnfc2 = nn.BatchNorm1d(num_features=4096)
        # Reshape
        # upsample
        self.dc0_1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=4, padding=0, bias=self.bias)
        self.dbn0_1 = nn.BatchNorm2d(32)

        self.dc0_2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0, bias=self.bias)
        self.dbn0_2 = nn.BatchNorm2d(16)

        self.dc0_3 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2, padding=0, bias=self.bias)
        self.dbn0_3 = nn.BatchNorm2d(8)

        # reconstruction
        # fc2 (8x8x64) conv4(8x8x512)
        self.dc1 = nn.ConvTranspose2d(in_channels=576, out_channels=512, kernel_size=2, stride=2, padding=0, bias=self.bias)
        self.dbn1 = nn.BatchNorm2d(512)

        # dc1(16x16x512) conv3(16x16x256)
        self.dc2 = nn.ConvTranspose2d(in_channels=768, out_channels=256, kernel_size=2, stride=2, padding=0, bias=self.bias)
        self.dbn2 = nn.BatchNorm2d(256)

        # dc2(32x32x256) conv2(32x32x128) Ia(128x128x3) dc0_1(32x32x32)
        self.dc3 = nn.ConvTranspose2d(in_channels=419, out_channels=128, kernel_size=2, stride=2, padding=0, bias=self.bias)
        self.dbn3 = nn.BatchNorm2d(128)

        # dc3(64x64x128) conv1(64x64x64) Ia(128x128x3) dc0_2(64x64x16)
        self.dc4 = nn.ConvTranspose2d(in_channels=211, out_channels=64, kernel_size=2, stride=2, padding=0, bias=self.bias)
        self.dbn4 = nn.BatchNorm2d(64)

        # Scale
        self.conv5 =  nn.Conv2d(in_channels=256, out_channels=3, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.bn5 = nn.BatchNorm2d(3)

        self.conv6 =  nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.bn6 = nn.BatchNorm2d(3)

        # dc4(128x128x64) conv0(128x128x64) Ia(128x128x3) dc0_3(128x128x8)
        self.conv7 =  nn.Conv2d(in_channels=139, out_channels=64, kernel_size=5, stride=1, padding=2, bias=self.bias)
        self.bn7 = nn.BatchNorm2d(64)

        self.conv8 =  nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.bn8 = nn.BatchNorm2d(32)

        self.conv9 =  nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1, bias=self.bias)
        self.bn9 = nn.BatchNorm2d(3)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
    def forward(self, ls_vector, r_conv4, r_conv3, r_conv2, r_conv1, r_conv0, ia_32, ia_64, ia):

       
        # input es maxout = ls_vector of decoder
        fc2 = self.fc2(ls_vector)

        # reshape
        out_fc2 = fc2.reshape(fc2.size(0), 64, 8,  8) #N x C x H x W

        ## Upsample
        dc0_1 = self.relu(self.dbn0_1(self.dc0_1(out_fc2)))
        dc0_2 = self.relu(self.dbn0_2(self.dc0_2(dc0_1)))
        dc0_3 = self.relu(self.dbn0_3(self.dc0_3(dc0_2)))

        ## Reconstruction
        concat = torch.cat([out_fc2, r_conv4], dim=1)
        dc1 = self.relu(self.dbn1(self.dc1(concat)))

        concat = torch.cat([dc1, r_conv3], dim=1)
        dc2 = self.relu(self.dbn2(self.dc2(concat)))

        concat = torch.cat([dc2, r_conv2, ia_32, dc0_1], dim=1)
        dc3 = self.relu(self.dbn3(self.dc3(concat)))

        concat = torch.cat([dc3, r_conv1, ia_64, dc0_2], dim=1)
        dc4 = self.relu(self.dbn4(self.dc4(concat)))

        # Scale
        conv5 = self.tanh(self.conv5(dc2)) # 32 x 32 x 3 conv5 = self.relu(self.bn5(self.conv5(dc2)))
        conv6 = self.tanh(self.conv6(dc3)) # 64 x 64 x 3 conv5 = self.relu(self.bn5(self.conv5(dc3)))
        concat = torch.cat([dc4, r_conv0, ia, dc0_3], dim=1)
        conv7 = self.relu(self.bn7(self.conv7(concat)))
        conv8 = self.relu(self.bn8(self.conv8(conv7)))
        conv9 = self.tanh(self.conv9(conv8)) # 128 x 128 x 3

        # Return 128 x 128 / 64 x 64 / 32 x 32
        return conv9, conv6, conv5

class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, ia, pa, pb, ia_64, ia_32):

        ls_vector, r_conv4, r_conv3, r_conv2, r_conv1, r_conv0  = self.encoder(ia, pa, pb) 
        conv9, conv6, conv5 =  self.decoder(ls_vector, r_conv4, r_conv3, r_conv2, r_conv1, r_conv0, ia_32, ia_64, ia)

        return conv9, conv6, conv5

    


class Discriminator(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.bias = False
        self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias = self.bias)
        self.bn0 = nn.BatchNorm2d(64)

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias = self.bias)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias = self.bias)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias = self.bias)
        self.bn3 = nn.BatchNorm2d(512)

        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=1, padding=1, bias = self.bias)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias = self.bias)
        self.bn5 = nn.BatchNorm2d(1)

        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, in_1, in_2):

        input = torch.cat([in_1, in_2], dim=1)

        conv0 =  self.lrelu(self.bn0(self.conv0(input)))
        conv1 =  self.lrelu(self.bn1(self.conv1(conv0)))
        conv2 =  self.lrelu(self.bn2(self.conv2(conv1)))
        conv3 =  self.lrelu(self.bn3(self.conv3(conv2)))
        conv4 =  self.lrelu(self.bn4(self.conv4(conv3)))
        conv5 = self.sigmoid(self.conv5(conv4))

        return conv5


# ########################################################################
# ## TRAIN
# ########################################################################
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import torch.nn.functional as F
# import numpy as np
# import torchvision.utils as vutils

# # Model
# import Datasets.dataCFP as datasets
# import utils.utils as utils
# from tqdm.autonotebook import tqdm


# ################################################
# # CONFIG 
# ################################################
# lambda1 = 10
# lambda2 = 0.1
# lambda3 = 0.1
# lambda4 = 0.02
# lambda5 = 0.0001
# lr      = 0.0002


# num_epochs  = 100
# batch_size  = 8
# print_epoch = 1


# ################################################
# # DATASET 
# ################################################
# path = '/home/liz/Documents/Data/cfp-dataset/Data/Images'
# path_result= '/media/liz/Files/TestTPGan'
# # Create the dataset
# dataset = datasets.DataSetTrain(path, isPatch=False, factor=0)

# train_size = int(0.7 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
# print(len(trainloader))
# print(len(testloader))
# # Plot some training images
# real_batch = next(iter(trainloader))


# ################################################
# # MODEL 
# ################################################
# device = torch.device('cpu')
# model_gen = Generator().to(device)
# model_dii = Discriminator(6).to(device)
# model_dpe = Discriminator(8).to(device)

# ################################################
# # COST FUNCTIONS 
# ################################################
# loss_pixel = nn.L1Loss(reduction='sum')
# loss_adv = nn.BCELoss()

# params = list(model_gen.parameters()) + list(model_dii.parameters()) +\
#          list(model_dpe.parameters())

# optimizer = torch.optim.Adam(params, lr=lr)
# ###
# # epoch = 11
# # state = {'epoch': epoch + 1, 'state_dict': model_gen.state_dict(),
# # 'optimizer': optimizer.state_dict()}
# # torch.save(state, path_result + "/model/checkpoint_gen" + str(epoch) + ".pth.tar" )

# # state = {'epoch': epoch + 1, 'state_dict': model_dii.state_dict(),
# # 'optimizer': optimizer.state_dict()}
# # torch.save(state, path_result + "/model/checkpoint_dii" + str(epoch) + ".pth.tar" )

# # state = {'epoch': epoch + 1, 'state_dict': model_dpe.state_dict(),
# # 'optimizer': optimizer.state_dict()}
# # torch.save(state, path_result + "/model/checkpoint_dp" + str(epoch) + ".pth.tar" ) 
# ###
# def train(epoch):
#     model_gen.train()
#     model_dii.train()
#     model_dpe.train()
#     value_loss  = 0

#     progress = tqdm(enumerate(trainloader), desc="Train", total=len(trainloader))
#     for i,x in progress:
#         optimizer.zero_grad()

#         source = x['profile'].to(device)
#         source32 = x['profile32'].to(device)
#         source64 = x['profile64'].to(device)
#         target = x['frontal'].to(device)
#         target32 = x["frontal32"].to(device)
#         target64 = x["frontal64"].to(device)
#         # Generate heap map
#         pa_profile = x["hm_p"].to(device)
#         pb_target = x["hm_f"].to(device)
#         img128, img64, img32 = model_gen(source, pa_profile, pb_target, source64, source32)
        
#         Lpix_128 = loss_pixel(img128, target) # 128*128*3 -> 0.000020345
#         Lpix_64 = loss_pixel(img64, target64) # 64*64*3 -> 0,00008138
#         Lpix_32 = loss_pixel(img32, target32) # 32*32*3 -> 0,000325521

#         Lpix = (Lpix_128*0.00002 + Lpix_64*0.00008 + Lpix_32*0.00032)/3

#         # Disc II
#         output_t = model_dii(target, source)
#         output_s = model_dii(img128, source)
#         label_real = torch.ones(output_t.size()).to(device)
#         label_fake = torch.zeros(output_s.size()).to(device)

#         Lii_real = loss_adv(output_t, label_real)
#         Lii_fake = loss_adv(output_s, label_fake)

#         Lii = Lii_real + Lii_fake

#         # Disc PE
#         output_pt = model_dpe(target, pb_target)
#         output_ps = model_dpe(img128, pb_target)
#         label_real = torch.ones(output_t.size()).to(device)
#         label_fake = torch.zeros(output_s.size()).to(device)

#         Lpe_real = loss_adv(output_pt, label_real)
#         Lpe_fake = loss_adv(output_ps, label_fake)

#         Lpe = Lpe_real + Lpe_fake

#         L = lambda1*Lpix + lambda2*Lii + lambda3*Lpe
#         L.backward()
#         optimizer.step()

#         value_loss += L.item()

#         progress.set_description("Epoch: %d, Loss: %.3f " % (epoch, L.item()))

#     if (epoch + 1) % print_epoch == 0:
#         state = {'epoch': epoch + 1, 'state_dict': model_gen.state_dict(),
#         'optimizer': optimizer.state_dict()}
#         torch.save(state, path_result + "/model/checkpoint_gen" + str(epoch) + ".pth.tar" )

#         state = {'epoch': epoch + 1, 'state_dict': model_dii.state_dict(),
#         'optimizer': optimizer.state_dict()}
#         torch.save(state, path_result + "/model/checkpoint_dii" + str(epoch) + ".pth.tar" )

#         state = {'epoch': epoch + 1, 'state_dict': model_dpe.state_dict(),
#         'optimizer': optimizer.state_dict()}
#         torch.save(state, path_result + "/model/checkpoint_dp" + str(epoch) + ".pth.tar" ) 
        
#     return value_loss

# def test(epoch):

#     model_gen.eval()
#     model_dii.eval()
#     model_dpe.eval()
#     value_loss  = 0
    
#     progress = tqdm(enumerate(testloader), desc="Train", total=len(testloader))

#     for i, x in progress:
#         source = x['profile'].to(device)
#         source32 = x['profile32'].to(device)
#         source64 = x['profile64'].to(device)
#         target = x['frontal'].to(device)
#         target32 = x["frontal32"].to(device)
#         target64 = x["frontal64"].to(device)
#         # Generate heap map
#         pa_profile = x["hm_p"].to(device)
#         pb_target = x["hm_f"].to(device)
#         img128, img64, img32 = model_gen(source, pa_profile, pb_target, source64, source32)
        
#         Lpix_128 = loss_pixel(img128, target) # 128*128*3 -> 0.000020345
#         Lpix_64 = loss_pixel(img64, target64) # 64*64*3 -> 0,00008138
#         Lpix_32 = loss_pixel(img32, target32) # 32*32*3 -> 0,000325521

#         Lpix = (Lpix_128*0.00002 + Lpix_64*0.00008 + Lpix_32*0.00032)/3

#         # Disc II
#         output_t = model_dii(target, source)
#         output_s = model_dii(img128, source)
#         label_real = torch.ones(output_t.size()).to(device)
#         label_fake = torch.zeros(output_s.size()).to(device)

#         Lii_real = loss_adv(output_t, label_real)
#         Lii_fake = loss_adv(output_s, label_fake)

#         Lii = Lii_real + Lii_fake

#         # Disc PE
#         output_pt = model_dpe(target, pb_target)
#         output_ps = model_dpe(img128, pb_target)
#         label_real = torch.ones(output_t.size()).to(device)
#         label_fake = torch.zeros(output_s.size()).to(device)

#         Lpe_real = loss_adv(output_pt, label_real)
#         Lpe_fake = loss_adv(output_ps, label_fake)

#         Lpe = Lpe_real + Lpe_fake

#         L = lambda1*Lpix + lambda2*Lii + lambda3*Lpe
       
#         value_loss += L.item()

#         progress.set_description("Test: %d, Loss: %.3f  " % (epoch, L.item()))

#     if (epoch + 1) % print_epoch == 0:

#         vutils.save_image(source.data, path_result + '/test/input_%03d.jpg' % epoch, normalize=True)
#         vutils.save_image(target.data, path_result + '/test/frontal_%03d.jpg' % epoch, normalize=True)
#         vutils.save_image(img128.data, path_result + '/test/gen_%03d.jpg' % epoch, normalize=True)
        
#     return value_loss

# Loss_train = []
# Loss_test = []
# for e in range(num_epochs):
#     train_loss = train(e)
#     print("Train : " + str(e) + "Loss: " + str(train_loss/len(trainloader)))
#     test_loss = test(e)
#     print("Test : " + str(e) + "Loss: " + str(test_loss/len(testloader)))

#     Loss_train.append(train_loss/len(trainloader))
#     Loss_test.append(test_loss/len(testloader))
    
#     if(e + 1) % print_epoch == 0:
#         x = np.arange(e + 1)
#         fig = plt.figure()
#         ax = plt.subplot(111)
#         ax.plot(x, Loss_train, 'mediumvioletred', label='Generator Training')
#         ax.plot(x, Loss_test, 'pink', label='Generator Test')

#         # ax.plot(x, Loss_Disc, 'steelblue', label='Discriminator Training')
#         # ax.plot(x, Loss_Disc_Test, 'lightskyblue', label='Discriminator Test')

#         plt.title('Function loss')
#         ax.legend()
#         fig.savefig(path_result + '/plot' + str(e) + '.png')
#         # plt.show()
#         plt.close(fig)

#         # Save results loss
#         fichero = open(path_result + '/files_gan_train_' + str(e) + '.pckl', 'wb')
#         pickle.dump(Loss_train, fichero)
#         fichero.close()
#         fichero = open(path_result + '/files_gan_test_' + str(e) + '.pckl', 'wb')
#         pickle.dump(Loss_test, fichero)
#         fichero.close()